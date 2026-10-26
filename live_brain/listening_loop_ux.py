"""
Live Brain — Listening Loop UX Components

Implements directives/listening_loop_ux.md:
1. Avatar Idle & Listening state management (AVATAR_IDLE_LISTENING)
2. VAD-Triggered Micro-Reactions (NOD_SUBTLE, TILT_ENGAGED, BROW_RAISE_BRIEF)
   with throttling (min 6s cooldown, max 2/turn, suppression on silence/barge-in)
3. Conversational Response Pacer (Variable response timing / Jitter Buffer:
   Base + RandomJitter + ComplexityWeight) allowing parallel LLM/TTS generation
4. VAD-Aligned Live Caption Smoother (flicker-free interim word clustering,
   VAD turn-completion finalization, interviewer 12-word bounded chunking)

All components are designed as non-blocking FrameProcessors conforming to
Pipecat standards, maintaining zero latency overhead on the core STT->LLM->TTS path.
"""

import asyncio
import enum
import inspect
import random
import re
import time
from typing import Any, Callable, Dict, List, Optional

from loguru import logger

from pipecat.frames.frames import (
    AudioRawFrame,
    CancelFrame,
    EndFrame,
    Frame,
    InterimTranscriptionFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    StartFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TextFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class AvatarState(str, enum.Enum):
    AVATAR_IDLE_LISTENING = "AVATAR_IDLE_LISTENING"
    AVATAR_THINKING = "AVATAR_THINKING"
    AVATAR_SPEAKING = "AVATAR_SPEAKING"
    AVATAR_MICRO_REACTION = "AVATAR_MICRO_REACTION"
    AVATAR_FILLER = "AVATAR_FILLER"


class MicroReactionType(str, enum.Enum):
    NOD_SUBTLE = "NOD_SUBTLE"            # 0.8s subtle nod
    TILT_ENGAGED = "TILT_ENGAGED"        # 1.2s engaged head tilt
    BROW_RAISE_BRIEF = "BROW_RAISE_BRIEF"# 0.4s affirmative brow raise
    THINKING_DEEP = "THINKING_DEEP"      # 1.5s deeper thinking/contemplative cue
    STILL_PROCESSING = "STILL_PROCESSING"# 1.0s subtle filler micro-cue when latency ceiling reached


# Durations in seconds for micro-reactions
MICRO_REACTION_DURATIONS: Dict[MicroReactionType, float] = {
    MicroReactionType.NOD_SUBTLE: 0.8,
    MicroReactionType.TILT_ENGAGED: 1.2,
    MicroReactionType.BROW_RAISE_BRIEF: 0.4,
    MicroReactionType.THINKING_DEEP: 1.5,
    MicroReactionType.STILL_PROCESSING: 1.0,
}


# Named latency ceiling configuration (4000ms before triggering filler cue)
FILLER_CUE_TIMEOUT_SECONDS: float = 4.0


class AvatarMicroReactionController(FrameProcessor):

    @property
    def task_manager(self):
        if not self._task_manager:
            try:
                from pipecat.utils.asyncio.task_manager import TaskManager
                self._task_manager = TaskManager()
            except Exception:
                pass
        return super().task_manager

    """
    Non-blocking tap processor monitoring candidate voice activity and
    orchestrating the Live Brain "feels alive" turn-taking & latency-masking loop:

    1. Candidate Speech: Monitors active voice duration and triggers natural
       micro-reactions (nod, head tilt, brow raise) with cooldown guardrails.
    2. End-of-Speech (VAD): Immediately enters looping AVATAR_THINKING animation state.
    3. Latency Absorption: Absorbs 1.2s - 3.0s Gemini TTFB variance while maintaining
       alive thinking animation.
    4. First Audio Chunk Sync: Cancels filler timer and transitions to AVATAR_SPEAKING
       strictly in sync with first TTS audio arrival.
    5. Latency Ceiling (4000ms): If no TTS audio arrives within ttfb_ceiling_seconds, triggers
       a subtle filler cue (STILL_PROCESSING / AVATAR_FILLER) without interrupting the stream.
    """

    def __init__(
        self,
        on_reaction: Optional[Callable[[Dict[str, Any]], Any]] = None,
        on_state_change: Optional[Callable[[AvatarState], Any]] = None,
        min_speech_duration: float = 2.5,
        cooldown_seconds: float = 6.0,
        max_per_turn: int = 2,
        ttfb_ceiling_seconds: float = FILLER_CUE_TIMEOUT_SECONDS,
        reaction_durations: Optional[Dict[MicroReactionType, float]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._on_reaction = on_reaction
        self._on_state_change = on_state_change
        self._min_speech_duration = min_speech_duration
        self._cooldown_seconds = cooldown_seconds
        self._max_per_turn = max_per_turn
        self._ttfb_ceiling_seconds = ttfb_ceiling_seconds
        self._reaction_durations = reaction_durations or dict(MICRO_REACTION_DURATIONS)

        self._state: AvatarState = AvatarState.AVATAR_IDLE_LISTENING
        self._speech_start_time: Optional[float] = None
        self._is_candidate_speaking: bool = False
        self._turn_index: int = 0
        self._reactions_in_current_turn: int = 0
        self._last_reaction_time: float = -999.0
        self._reaction_timer_task: Optional[asyncio.Task] = None
        self._ceiling_timer_task: Optional[asyncio.Task] = None
        self._active_reaction: Optional[MicroReactionType] = None
        self._llm_tokens_received_in_turn: bool = False

    @property
    def current_state(self) -> AvatarState:
        return self._state

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        if isinstance(frame, StartFrame):
            self._reset_session()
            await self._set_state(AvatarState.AVATAR_IDLE_LISTENING)

        elif isinstance(frame, UserStartedSpeakingFrame):
            # Candidate began speaking -> cancel pending ceiling/thinking tasks, transition to idle/listening
            self._handle_user_started_speaking()
            await self._set_state(AvatarState.AVATAR_IDLE_LISTENING)

        elif isinstance(frame, (UserStoppedSpeakingFrame, TranscriptionFrame)):
            # Candidate finished speaking (VAD detected) -> immediately trigger looping thinking state and arm 4000ms timer
            await self._handle_user_stopped_speaking()

        elif isinstance(frame, TTSAudioRawFrame):
            # FIRST TTS audio chunk arriving at output -> cancel filler timer, immediate cut-to-speaking in sync with audio
            self._cancel_ceiling_task()
            if self._state != AvatarState.AVATAR_SPEAKING:
                self._abort_active_reaction()
                await self._set_state(AvatarState.AVATAR_SPEAKING)

        elif isinstance(frame, (InterimTranscriptionFrame, AudioRawFrame)):
            # Check continuous speech duration for micro-reactions while candidate is speaking
            self._check_speech_duration()

        elif isinstance(frame, (LLMFullResponseStartFrame, LLMTextFrame, TextFrame)):
            # LLM tokens flowing: hold thinking state until TTS audio arrives
            self._handle_llm_token_arrival()

        elif isinstance(frame, TTSStartedFrame):
            # TTS synthesis in progress: hold thinking state until actual audio arrives
            pass

        elif isinstance(frame, TTSStoppedFrame):
            # Interviewer speech complete -> return to idle listening
            self._cancel_ceiling_task()
            await self._set_state(AvatarState.AVATAR_IDLE_LISTENING)

        elif isinstance(frame, (CancelFrame, EndFrame)):
            self._abort_active_reaction()
            self._cancel_ceiling_task()
            self._reset_session()

        # Non-blocking tap: pass frame downstream immediately
        await self.push_frame(frame, direction)

    def _reset_session(self) -> None:
        self._speech_start_time = None
        self._is_candidate_speaking = False
        self._turn_index = 0
        self._reactions_in_current_turn = 0
        self._last_reaction_time = -999.0
        self._llm_tokens_received_in_turn = False
        self._abort_active_reaction()
        self._cancel_ceiling_task()

    def _handle_user_started_speaking(self) -> None:
        now = time.monotonic()
        self._cancel_ceiling_task()
        self._llm_tokens_received_in_turn = False
        if not self._is_candidate_speaking:
            self._is_candidate_speaking = True
            self._turn_index += 1
            self._reactions_in_current_turn = 0
            self._speech_start_time = now

    async def _handle_user_stopped_speaking(self) -> None:
        self._is_candidate_speaking = False
        self._speech_start_time = None
        self._abort_active_reaction()

        # Enter looping thinking animation state immediately on VAD turn boundary
        if self._state not in (AvatarState.AVATAR_SPEAKING, AvatarState.AVATAR_THINKING):
            await self._set_state(AvatarState.AVATAR_THINKING)

        # Start hard ceiling timer (4000ms default) to surface filler cue if TTS audio is delayed
        if self._ceiling_timer_task is None or self._ceiling_timer_task.done():
            self._ceiling_timer_task = asyncio.create_task(
                self._ttfb_ceiling_watcher(self._ttfb_ceiling_seconds)
            )

    def _handle_llm_token_arrival(self) -> None:
        if not self._llm_tokens_received_in_turn:
            self._llm_tokens_received_in_turn = True

    async def _ttfb_ceiling_watcher(self, timeout_secs: float) -> None:
        try:
            await asyncio.sleep(timeout_secs)
            if self._state == AvatarState.AVATAR_THINKING:
                logger.warning(
                    f"[FILLER_CUE_FIRED] No TTS audio received within {timeout_secs * 1000:.0f}ms of user silence. "
                    "Triggering latency filler cue."
                )
                await self._set_state(AvatarState.AVATAR_FILLER)
                self._emit_reaction({
                    "type": "avatar_latency_filler",
                    "reaction": MicroReactionType.STILL_PROCESSING.value,
                    "reason": "ttfb_ceiling_exceeded",
                    "duration_seconds": self._reaction_durations[MicroReactionType.STILL_PROCESSING],
                    "turn": self._turn_index,
                    "timestamp": time.monotonic(),
                })
        except asyncio.CancelledError:
            pass

    def _cancel_ceiling_task(self) -> None:
        if self._ceiling_timer_task and not self._ceiling_timer_task.done():
            self._ceiling_timer_task.cancel()
            self._ceiling_timer_task = None

    def _check_speech_duration(self) -> None:
        if not self._is_candidate_speaking or self._speech_start_time is None:
            return

        now = time.monotonic()
        speech_duration = now - self._speech_start_time

        # Check conditions: sustained speech, cooldown, max per turn, avatar listening
        if (
            speech_duration >= self._min_speech_duration
            and (now - self._last_reaction_time) >= self._cooldown_seconds
            and self._reactions_in_current_turn < self._max_per_turn
            and self._active_reaction is None
            and self._state in (AvatarState.AVATAR_IDLE_LISTENING, AvatarState.AVATAR_MICRO_REACTION)
        ):
            self._trigger_micro_reaction(now)

    def _trigger_micro_reaction(self, now: float) -> None:
        reaction_choices = [
            MicroReactionType.NOD_SUBTLE,
            MicroReactionType.TILT_ENGAGED,
            MicroReactionType.BROW_RAISE_BRIEF,
        ]
        chosen_reaction = random.choice(reaction_choices)
        duration = self._reaction_durations[chosen_reaction]

        self._active_reaction = chosen_reaction
        self._last_reaction_time = now
        self._reactions_in_current_turn += 1

        logger.debug(
            f"[listening_loop_ux] Micro-reaction triggered: {chosen_reaction.value} "
            f"(turn {self._turn_index}, count {self._reactions_in_current_turn})"
        )

        event_payload = {
            "type": "avatar_micro_reaction",
            "reaction": chosen_reaction.value,
            "duration_seconds": duration,
            "turn": self._turn_index,
            "timestamp": now,
        }

        self._emit_reaction(event_payload)
        self._reaction_timer_task = asyncio.create_task(
            self._reaction_duration_watcher(chosen_reaction, duration)
        )

    async def _reaction_duration_watcher(
        self, reaction: MicroReactionType, duration: float
    ) -> None:
        try:
            await self._set_state(AvatarState.AVATAR_MICRO_REACTION)
            await asyncio.sleep(duration)
            if self._active_reaction == reaction:
                self._active_reaction = None
                if self._state == AvatarState.AVATAR_MICRO_REACTION:
                    await self._set_state(AvatarState.AVATAR_IDLE_LISTENING)
        except asyncio.CancelledError:
            self._active_reaction = None
            if self._state == AvatarState.AVATAR_MICRO_REACTION:
                await self._set_state(AvatarState.AVATAR_IDLE_LISTENING)

    def _abort_active_reaction(self) -> None:
        if self._reaction_timer_task and not self._reaction_timer_task.done():
            self._reaction_timer_task.cancel()
            self._reaction_timer_task = None
        self._active_reaction = None

    async def _set_state(self, new_state: AvatarState) -> None:
        if self._state != new_state:
            self._state = new_state
            if self._on_state_change:
                try:
                    res = self._on_state_change(new_state)
                    if inspect.isawaitable(res):
                        await res
                except Exception:
                    logger.exception("[listening_loop_ux] on_state_change callback error")

    def _emit_reaction(self, event: Dict[str, Any]) -> None:
        if self._on_reaction:
            try:
                res = self._on_reaction(event)
                if inspect.isawaitable(res):
                    asyncio.create_task(res)
            except Exception:
                logger.exception("[listening_loop_ux] on_reaction callback error")


class ConversationalResponsePacer(FrameProcessor):

    @property
    def task_manager(self):
        if not self._task_manager:
            try:
                from pipecat.utils.asyncio.task_manager import TaskManager
                self._task_manager = TaskManager()
            except Exception:
                pass
        return super().task_manager

    """
    Conversational Pacing & Jitter Buffer Processor.

    Solves the uncanny 'zero-latency' instant response problem while eliminating
    buffer stalls. Pre-computation of LLM tokens and TTS audio streams concurrently
    during the pacing window:

      Δt_pause = Base + RandomJitter + ComplexityWeight

      - Short answers / Confirmations: 300ms - 500ms delay
      - Complex answers / Rubric follow-ups: 600ms - 900ms delay

    Audio frames arriving from TTS during the pause window are queued in an internal
    buffer. The audio stream is released smoothly the instant the jitter timer expires.
    """

    def __init__(
        self,
        base_short_ms: int = 300,
        max_short_jitter_ms: int = 200,
        base_complex_ms: int = 600,
        max_complex_jitter_ms: int = 300,
        short_turn_word_threshold: int = 10,
        on_pacing_delay: Optional[Callable[[Dict[str, Any]], Any]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._base_short_ms = base_short_ms
        self._max_short_jitter_ms = max_short_jitter_ms
        self._base_complex_ms = base_complex_ms
        self._max_complex_jitter_ms = max_complex_jitter_ms
        self._short_turn_word_threshold = short_turn_word_threshold
        self._on_pacing_delay = on_pacing_delay

        self._candidate_turn_ended_at: Optional[float] = None
        self._target_release_time: Optional[float] = None
        self._current_pacing_delay_ms: float = 0.0
        self._is_buffering_audio: bool = False
        self._audio_frame_buffer: List[Frame] = []
        self._release_task: Optional[asyncio.Task] = None
        self._last_candidate_word_count: int = 15

    def calculate_pause_delay_ms(self, candidate_word_count: int) -> float:
        """Computes Δt_pause based on candidate utterance complexity and jitter."""
        if candidate_word_count <= self._short_turn_word_threshold:
            # Short / confirmation turn
            jitter = random.uniform(0, self._max_short_jitter_ms)
            return self._base_short_ms + jitter
        else:
            # Complex architecture / rubric explanation
            jitter = random.uniform(0, self._max_complex_jitter_ms)
            return self._base_complex_ms + jitter

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        if isinstance(frame, StartFrame):
            self._reset()
            await self.push_frame(frame, direction)

        elif isinstance(frame, UserStartedSpeakingFrame):
            # User barge-in or new speech: abort any pending audio release
            self._reset()
            await self.push_frame(frame, direction)

        elif isinstance(frame, TranscriptionFrame):
            # Candidate turn completed
            text = (frame.text or "").strip()
            words = re.findall(r"[\w']+", text)
            self._last_candidate_word_count = len(words)
            self._candidate_turn_ended_at = time.monotonic()

            self._current_pacing_delay_ms = self.calculate_pause_delay_ms(
                self._last_candidate_word_count
            )
            self._target_release_time = (
                self._candidate_turn_ended_at + (self._current_pacing_delay_ms / 1000.0)
            )
            self._is_buffering_audio = True

            logger.debug(
                f"[listening_loop_ux] Candidate turn ended ({self._last_candidate_word_count} words). "
                f"Conversational pacing delay set to {self._current_pacing_delay_ms:.1f} ms"
            )

            if self._on_pacing_delay:
                try:
                    res = self._on_pacing_delay({
                        "type": "pacing_delay_calculated",
                        "delay_ms": round(self._current_pacing_delay_ms, 1),
                        "candidate_word_count": self._last_candidate_word_count,
                    })
                    if inspect.isawaitable(res):
                        await res
                except Exception:
                    pass

            await self.push_frame(frame, direction)

        elif isinstance(frame, (TTSAudioRawFrame, AudioRawFrame)):
            # Downstream TTS audio
            now = time.monotonic()
            if self._is_buffering_audio and self._target_release_time is not None:
                if now < self._target_release_time:
                    # Enqueue frame in jitter buffer
                    self._audio_frame_buffer.append(frame)
                    remaining_wait = max(0.0, self._target_release_time - now)
                    if self._release_task is None or self._release_task.done():
                        self._release_task = asyncio.create_task(
                            self._flush_buffer_after(remaining_wait, direction)
                        )
                    return
                else:
                    # Timer already expired, release buffering
                    self._is_buffering_audio = False
                    await self._flush_buffer(direction)
                    await self.push_frame(frame, direction)
            else:
                await self.push_frame(frame, direction)

        elif isinstance(frame, (CancelFrame, EndFrame)):
            self._reset()
            await self.push_frame(frame, direction)

        else:
            await self.push_frame(frame, direction)

    async def _flush_buffer_after(self, wait_secs: float, direction: FrameDirection) -> None:
        try:
            if wait_secs > 0:
                await asyncio.sleep(wait_secs)
            self._is_buffering_audio = False
            await self._flush_buffer(direction)
        except asyncio.CancelledError:
            self._audio_frame_buffer.clear()
            self._is_buffering_audio = False

    async def _flush_buffer(self, direction: FrameDirection) -> None:
        while self._audio_frame_buffer:
            frame = self._audio_frame_buffer.pop(0)
            await self.push_frame(frame, direction)

    def _reset(self) -> None:
        if self._release_task and not self._release_task.done():
            self._release_task.cancel()
            self._release_task = None
        self._audio_frame_buffer.clear()
        self._is_buffering_audio = False
        self._candidate_turn_ended_at = None
        self._target_release_time = None


class LiveCaptionSmoother(FrameProcessor):

    @property
    def task_manager(self):
        if not self._task_manager:
            try:
                from pipecat.utils.asyncio.task_manager import TaskManager
                self._task_manager = TaskManager()
            except Exception:
                pass
        return super().task_manager

    """
    VAD-Aligned Live Caption Smoother.

    Eliminates flickering and visual jumping caused by raw interim STT partials:
      1. Candidate Input: Buffers STT interim hypotheses, commits stable word
         clusters, and finalizes with a clean opacity/turn transition on VAD
         utterance completion (stop_secs = 0.3s).
      2. Interviewer Output: Synchronized with TTS word-level alignment timestamps,
         rendered in bounded sentence chunks (max 12 words per line) matching
         avatar lip-sync.

    Emits structured caption events via on_caption callback without blocking frames.
    """

    def __init__(
        self,
        on_caption: Optional[Callable[[Dict[str, Any]], Any]] = None,
        max_words_per_line: int = 12,
        min_interim_cluster_words: int = 3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._on_caption = on_caption
        self._max_words_per_line = max_words_per_line
        self._min_interim_cluster_words = min_interim_cluster_words

        self._candidate_interim_words: List[str] = []
        self._candidate_last_emitted_text: str = ""
        self._candidate_turn_index: int = 0
        self._interviewer_words: List[str] = []
        self._interviewer_turn_index: int = 0

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        try:
            await super().process_frame(frame, direction)
        except Exception:
            pass

        if isinstance(frame, StartFrame):
            self._reset()

        elif isinstance(frame, InterimTranscriptionFrame):
            # Candidate live interim hypothesis
            self._handle_interim_transcription(frame)

        elif isinstance(frame, (TranscriptionFrame, UserStoppedSpeakingFrame)):
            # Candidate utterance committed / finalized on VAD boundary
            if isinstance(frame, TranscriptionFrame):
                self._handle_final_transcription(frame)
            elif isinstance(frame, UserStoppedSpeakingFrame):
                self._finalize_candidate_caption()

        elif isinstance(frame, TTSStartedFrame):
            self._interviewer_turn_index += 1
            self._interviewer_words = []

        elif isinstance(frame, (CancelFrame, EndFrame)):
            self._reset()

        # Non-blocking pass-through
        await self.push_frame(frame, direction)

    def _handle_interim_transcription(self, frame: InterimTranscriptionFrame) -> None:
        text = (frame.text or "").strip()
        if not text:
            return

        words = re.findall(r"\S+", text)
        self._candidate_interim_words = words

        # Commit stable word cluster when length is sufficient and changed
        if (
            len(words) >= self._min_interim_cluster_words
            and text != self._candidate_last_emitted_text
        ):
            self._candidate_last_emitted_text = text
            self._emit_caption({
                "speaker": "candidate",
                "status": "interim",
                "text": text,
                "turn": self._candidate_turn_index,
                "is_final": False,
            })

    def _handle_final_transcription(self, frame: TranscriptionFrame) -> None:
        text = (frame.text or "").strip()
        if not text:
            return

        self._candidate_turn_index += 1
        self._candidate_last_emitted_text = text
        self._candidate_interim_words = []

        self._emit_caption({
            "speaker": "candidate",
            "status": "final",
            "text": text,
            "turn": self._candidate_turn_index,
            "is_final": True,
        })

    def _finalize_candidate_caption(self) -> None:
        if self._candidate_interim_words:
            final_text = " ".join(self._candidate_interim_words)
            self._candidate_turn_index += 1
            self._candidate_interim_words = []
            self._candidate_last_emitted_text = ""

            self._emit_caption({
                "speaker": "candidate",
                "status": "final",
                "text": final_text,
                "turn": self._candidate_turn_index,
                "is_final": True,
            })

    def handle_interviewer_words(self, words: List[str]) -> None:
        """Chunk interviewer output words into lines of max_words_per_line."""
        for i in range(0, len(words), self._max_words_per_line):
            chunk = " ".join(words[i : i + self._max_words_per_line])
            self._emit_caption({
                "speaker": "interviewer",
                "status": "spoken",
                "text": chunk,
                "turn": self._interviewer_turn_index,
                "is_final": (i + self._max_words_per_line) >= len(words),
            })

    def _emit_caption(self, payload: Dict[str, Any]) -> None:
        if self._on_caption:
            try:
                res = self._on_caption(payload)
                if inspect.isawaitable(res):
                    asyncio.create_task(res)
            except Exception:
                logger.exception("[listening_loop_ux] on_caption callback error")

    def _reset(self) -> None:
        self._candidate_interim_words.clear()
        self._candidate_last_emitted_text = ""
        self._candidate_turn_index = 0
        self._interviewer_words.clear()
        self._interviewer_turn_index = 0

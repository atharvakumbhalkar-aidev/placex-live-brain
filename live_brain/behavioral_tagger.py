"""
BehavioralSignalTagger — passive coaching-signal tap for the Live Brain pipeline.

Owner: behavioral coaching signal subsystem
(see Design Architecture -> Behavioral Guardrails).

WHAT THIS IS
Sits between STT and the user context aggregator in bot.py's pipeline:

    stt -> behavioral_tagger -> user_aggregator -> llm -> ...

It is a *tap*, not a gate: every frame that arrives is forwarded downstream
unchanged, in the same direction it arrived in. Analysis happens on the
side, and a bug or slow callback in analysis must never stall or break the
candidate's interview. If you extend this file, preserve that property.

WHAT IT TAGS (v1 - content-based only)
Signals are computed from finalized transcription text only. Interim/partial
STT results are ignored - they're provisional and would just produce noisy,
duplicate signals as STT revises them. Per finalized candidate turn, this
looks for:

  - filler_words       e.g. "um", "uh", "like", "you know"
  - hedging_language   e.g. "I think", "probably", "not sure"
  - repeated_words      immediate word-level repetition ("the the")
  - short_response / long_response   word count relative to a threshold
  - turn_gap            best-effort wall-clock time since this candidate's
                         previous finalized turn (see caveat below)

COMPLIANCE / FRAMING - READ BEFORE ADDING SIGNAL TYPES
Per Design Architecture: "Behavioral signals are candidate-facing
coaching, never hidden scoring." Concretely: every signal dict below is
*descriptive* (what was observed) and never *evaluative* (what it means
about the candidate). No "confidence_score", "nervousness_level", or
pass/fail judgment belongs in this file - that interpretation is the
Scoring/Feedback Brain's job, downstream of on_signal(). Keep it that way.

KNOWN LIMITATION - turn_gap
This tap sits before the LLM/TTS stages, so it never sees the interviewer's
side of the conversation - it has no idea when a question ends. turn_gap is
therefore "time since this candidate's last finalized utterance," which
conflates two different things: the candidate thinking in silence, and the
interviewer still talking. Treat it as a rough signal, not a clean
"hesitation" measurement. If that distinction turns out to matter, the real
fix is an event from the LLM/TTS stage ("assistant finished speaking"), not
a cleverer calculation here.

ON on_signal
May be sync or async - both are supported, since bot.py's own comment on
log_behavioral_signal anticipates swapping it for "a queue push, a
websocket to Antigravity, or a DB write" without touching this file. A
failing on_signal callback is caught and logged, never raised, and never
stops other signals for the same turn from being emitted.

VERSION NOTE
Written against pipecat's FrameProcessor / TranscriptionFrame API as used
elsewhere in this repo's bot.py. As bot.py's own docstring says: confirm
signatures against https://docs.pipecat.ai if pipecat has moved since.
"""

import inspect
import re
import time
from typing import Any, Callable, List, Optional, Tuple

from loguru import logger

from pipecat.frames.frames import Frame, StartFrame, TranscriptionFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

# Tune these per subsystem learnings - a natural home once it exists is
# directives/live-brain/behavioral_tagger.md. Matching is case-insensitive.
DEFAULT_FILLER_WORDS: Tuple[str, ...] = (
    "um", "uh", "uhh", "umm", "er", "ah",
    "like", "you know", "i mean", "sort of", "kind of",
    "basically", "actually", "literally",
)
DEFAULT_HEDGES: Tuple[str, ...] = (
    "i think", "i guess", "i suppose", "probably", "maybe",
    "not sure", "i don't know", "i'm not sure",
)

SHORT_TURN_WORDS = 5     # at or below this many words -> "short_response"
LONG_TURN_WORDS = 150    # at or above this many words -> "long_response"


def _compile_phrase_pattern(phrases: Tuple[str, ...]) -> re.Pattern:
    """One alternation regex, longest phrases first so multi-word phrases
    match before a shorter phrase they contain (e.g. "not sure" before a
    hypothetical standalone "sure")."""
    ordered = sorted(phrases, key=len, reverse=True)
    escaped = [re.escape(p) for p in ordered]
    return re.compile(r"\b(" + "|".join(escaped) + r")\b", re.IGNORECASE)


class BehavioralSignalTagger(FrameProcessor):

    @property
    def task_manager(self):
        if not self._task_manager:
            try:
                from pipecat.utils.asyncio.task_manager import TaskManager
                self._task_manager = TaskManager()
            except Exception:
                pass
        return super().task_manager

    """Passive tap: forwards every frame unchanged; emits coaching signals
    on the side for each finalized candidate transcription turn."""

    def __init__(
        self,
        on_signal: Callable[[dict], Any],
        filler_words: Tuple[str, ...] = DEFAULT_FILLER_WORDS,
        hedge_phrases: Tuple[str, ...] = DEFAULT_HEDGES,
        short_turn_words: int = SHORT_TURN_WORDS,
        long_turn_words: int = LONG_TURN_WORDS,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._on_signal = on_signal
        self._filler_pattern = _compile_phrase_pattern(filler_words)
        self._hedge_pattern = _compile_phrase_pattern(hedge_phrases)
        self._short_turn_words = short_turn_words
        self._long_turn_words = long_turn_words

        self._turn_index = 0
        self._last_turn_ended_at: Optional[float] = None

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        try:
            await super().process_frame(frame, direction)
        except Exception:
            pass

        if isinstance(frame, StartFrame):
            # Fresh session (or a defensive reset if this instance is ever
            # reused) - don't carry turn count/timing across sessions.
            self._turn_index = 0
            self._last_turn_ended_at = None

        elif isinstance(frame, TranscriptionFrame):
            # A bug here must never take down the interview - log and move on.
            try:
                await self._handle_final_transcription(frame)
            except Exception:
                logger.exception("[behavioral] tagger failed on a turn, skipping")

        # Tap contract: every frame - including interim transcriptions,
        # audio, and anything else - is forwarded unchanged, same direction
        # it arrived in. This line must always run.
        await self.push_frame(frame, direction)

    async def _handle_final_transcription(self, frame: TranscriptionFrame) -> None:
        text = (frame.text or "").strip()
        if not text:
            return

        now = time.monotonic()
        self._turn_index += 1
        turn = self._turn_index

        words = re.findall(r"[\w']+", text)
        word_count = len(words)

        fillers = self._filler_pattern.findall(text)
        if fillers:
            await self._emit({
                "type": "filler_words",
                "turn": turn,
                "count": len(fillers),
                "words": [f.lower() for f in fillers],
            })

        hedges = self._hedge_pattern.findall(text)
        if hedges:
            await self._emit({
                "type": "hedging_language",
                "turn": turn,
                "count": len(hedges),
                "phrases": [h.lower() for h in hedges],
            })

        repeats = self._find_repeated_words(words)
        if repeats:
            await self._emit({
                "type": "repeated_words",
                "turn": turn,
                "words": repeats,
            })

        if word_count <= self._short_turn_words:
            await self._emit({
                "type": "short_response",
                "turn": turn,
                "word_count": word_count,
            })
        elif word_count >= self._long_turn_words:
            await self._emit({
                "type": "long_response",
                "turn": turn,
                "word_count": word_count,
            })

        if self._last_turn_ended_at is not None:
            await self._emit({
                "type": "turn_gap",
                "turn": turn,
                "seconds": round(now - self._last_turn_ended_at, 2),
                "note": "may include interviewer speaking time - see module docstring",
            })
        self._last_turn_ended_at = now

    @staticmethod
    def _find_repeated_words(words: List[str]) -> List[str]:
        repeats = []
        for i in range(1, len(words)):
            if words[i].lower() == words[i - 1].lower():
                repeats.append(words[i].lower())
        return repeats

    async def _emit(self, signal: dict) -> None:
        try:
            result = self._on_signal(signal)
            if inspect.isawaitable(result):
                await result
        except Exception:
            logger.exception(
                f"[behavioral] on_signal callback failed for signal type "
                f"{signal.get('type')!r}"
            )


class ProsodicSignalTagger(FrameProcessor):

    @property
    def task_manager(self):
        if not self._task_manager:
            try:
                from pipecat.utils.asyncio.task_manager import TaskManager
                self._task_manager = TaskManager()
            except Exception:
                pass
        return super().task_manager

    """Passive tap: extracts acoustic & prosodic features (pitch variance,
    speaking rate, pause duration) from candidate audio/transcription frames
    without blocking the real-time LLM loop."""

    def __init__(
        self,
        on_signal: Callable[[dict], Any],
        sample_rate: int = 16000,
        min_speech_duration_secs: float = 0.5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._on_signal = on_signal
        self._sample_rate = sample_rate
        self._min_speech_duration_secs = min_speech_duration_secs

        self._turn_index = 0
        self._turn_audio_chunks: List[bytes] = []
        self._turn_start_time: Optional[float] = None
        self._turn_last_speech_time: Optional[float] = None
        self._intra_turn_pauses: List[float] = []

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        try:
            await super().process_frame(frame, direction)
        except Exception:
            pass

        if isinstance(frame, StartFrame):
            self._turn_index = 0
            self._turn_audio_chunks = []
            self._turn_start_time = None
            self._turn_last_speech_time = None
            self._intra_turn_pauses = []

        elif isinstance(frame, TranscriptionFrame):
            try:
                await self._handle_transcription_prosody(frame)
            except Exception:
                logger.exception("[prosody] tagger failed on transcription turn, skipping")

        else:
            # Check for audio frames defensively (e.g. InputAudioRawFrame / AudioRawFrame)
            frame_type_name = frame.__class__.__name__
            if "Audio" in frame_type_name and hasattr(frame, "audio"):
                try:
                    self._buffer_audio(frame.audio)
                except Exception:
                    pass

        # Tap contract: forward every frame downstream unchanged
        await self.push_frame(frame, direction)

    def _buffer_audio(self, raw_bytes: bytes) -> None:
        if not raw_bytes:
            return
        now = time.monotonic()
        if self._turn_start_time is None:
            self._turn_start_time = now

        if self._turn_last_speech_time is not None:
            gap = now - self._turn_last_speech_time
            # Intra-turn pause between distinct speech chunks (> 300ms)
            if gap >= 0.3:
                self._intra_turn_pauses.append(round(gap, 2))

        self._turn_last_speech_time = now
        self._turn_audio_chunks.append(raw_bytes)

    async def _handle_transcription_prosody(self, frame: TranscriptionFrame) -> None:
        text = (frame.text or "").strip()
        if not text:
            return

        now = time.monotonic()
        self._turn_index += 1
        turn = self._turn_index

        words = re.findall(r"[\w']+", text)
        word_count = len(words)

        # 1. Speaking Rate (Words Per Minute)
        duration_secs = 0.0
        if self._turn_start_time is not None:
            duration_secs = max(now - self._turn_start_time, self._min_speech_duration_secs)
        elif hasattr(frame, "timestamp") and hasattr(frame, "duration") and frame.duration:
            duration_secs = frame.duration
        else:
            # Fallback estimation: average speaking rate ~140 wpm
            duration_secs = max((word_count / 140.0) * 60.0, self._min_speech_duration_secs)

        wpm = round((word_count / duration_secs) * 60.0, 1) if duration_secs > 0 else 0.0

        await self._emit({
            "type": "prosody_speaking_rate",
            "turn": turn,
            "word_count": word_count,
            "duration_seconds": round(duration_secs, 2),
            "words_per_minute": wpm,
        })

        # 2. Pause Duration
        total_pause_secs = sum(self._intra_turn_pauses)
        pause_count = len(self._intra_turn_pauses)
        avg_pause_secs = round(total_pause_secs / pause_count, 2) if pause_count > 0 else 0.0

        await self._emit({
            "type": "prosody_pause_duration",
            "turn": turn,
            "pause_count": pause_count,
            "total_pause_seconds": round(total_pause_secs, 2),
            "average_pause_seconds": avg_pause_secs,
            "pauses": list(self._intra_turn_pauses),
        })

        # 3. Pitch Variance (F0 Estimation over raw PCM audio buffer)
        if self._turn_audio_chunks:
            pitch_stats = self._calculate_pitch_variance(
                b"".join(self._turn_audio_chunks),
                self._sample_rate,
            )
            if pitch_stats:
                await self._emit({
                    "type": "prosody_pitch_variance",
                    "turn": turn,
                    **pitch_stats,
                })

        # Reset turn buffer
        self._turn_audio_chunks = []
        self._turn_start_time = None
        self._turn_last_speech_time = None
        self._intra_turn_pauses = []

    @staticmethod
    def _calculate_pitch_variance(raw_pcm: bytes, sample_rate: int = 16000) -> Optional[dict]:
        """Estimate fundamental frequency (F0) distribution and variance using
        autocorrelation over 30ms windowed frames."""
        try:
            import numpy as np

            # 16-bit mono PCM to float32
            audio = np.frombuffer(raw_pcm, dtype=np.int16).astype(np.float32)
            if len(audio) < sample_rate * 0.2:  # less than 200ms of audio
                return None

            frame_size = int(sample_rate * 0.03)  # 30ms frames
            hop_size = int(sample_rate * 0.015)   # 15ms hop
            min_lag = int(sample_rate / 400)      # 400 Hz max pitch
            max_lag = int(sample_rate / 60)       # 60 Hz min pitch

            pitches = []
            for start in range(0, len(audio) - frame_size, hop_size):
                segment = audio[start : start + frame_size]
                # Energy threshold for voicing
                rms = np.sqrt(np.mean(segment**2))
                if rms < 200.0:
                    continue

                # Normalized autocorrelation
                corr = np.correlate(segment, segment, mode="full")
                corr = corr[len(corr) // 2 :]
                if max_lag >= len(corr):
                    continue

                lag = min_lag + np.argmax(corr[min_lag:max_lag])
                if corr[lag] > 0.3 * corr[0]:
                    f0 = sample_rate / lag
                    pitches.append(f0)

            if len(pitches) < 3:
                return None

            f0_arr = np.array(pitches)
            return {
                "f0_mean_hz": round(float(np.mean(f0_arr)), 1),
                "f0_std_hz": round(float(np.std(f0_arr)), 1),
                "f0_min_hz": round(float(np.min(f0_arr)), 1),
                "f0_max_hz": round(float(np.max(f0_arr)), 1),
                "f0_variance": round(float(np.var(f0_arr)), 2),
            }
        except Exception:
            return None

    async def _emit(self, signal: dict) -> None:
        try:
            result = self._on_signal(signal)
            if inspect.isawaitable(result):
                await result
        except Exception:
            logger.exception(
                f"[prosody] on_signal callback failed for signal type "
                f"{signal.get('type')!r}"
            )



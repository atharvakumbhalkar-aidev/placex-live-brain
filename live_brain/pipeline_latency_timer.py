"""
Live Brain — Pipeline Latency Observer
Tracks turn latency across 5 distinct pipeline stages using Pipecat 1.7.0 BaseObserver.
"""

import json
import time
from collections import deque
from pathlib import Path
from typing import Optional, Callable, Dict, Any, List, Set

from loguru import logger

from pipecat.frames.frames import (
    Frame,
    EndFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
    TranscriptionFrame,
    LLMTextFrame,
    TTSAudioRawFrame,
    OutputImageRawFrame,
    TTSStoppedFrame,
    InterruptionFrame,
)
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class PipelineLatencyTimer(BaseObserver):
    """
    PipelineTask Observer measuring turn latency across 5 stages without blocking the audio loop.
    """

    def __init__(
        self,
        output_path: Optional[str | Path] = None,
        clock: Optional[Callable[[], float]] = None,
        output_processor: Optional[FrameProcessor] = None,
        get_avatar_state: Optional[Callable[[], bool]] = None,
        max_frame_history: int = 500,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._output_path = Path(output_path) if output_path else Path(".tmp/latency.jsonl")
        self._clock = clock or time.monotonic
        self._output_processor = output_processor
        self._get_avatar_state = get_avatar_state
        self._turn_index = 0
        self._turn_active = False

        # Hop-level deduplication (frame.id, destination_id) to allow observing frame at every hop
        self._processed_frame_hops: Set[tuple] = set()
        self._hop_history: deque = deque(maxlen=max_frame_history)

        # Global frame-id deduplication for turn-level trigger events (speech start/stop)
        self._processed_event_frame_ids: Set[str] = set()
        self._event_history: deque = deque(maxlen=max_frame_history)

        # Active turn context and timestamps
        self._active_tts_context_id: Optional[str] = None
        self._t_speech_end: Optional[float] = None
        self._last_stt_final_time: Optional[float] = None
        self._t_first_llm_token: Optional[float] = None
        self._t_first_tts_audio: Optional[float] = None
        self._t_first_simli_frame: Optional[float] = None
        self._t_tts_stopped: Optional[float] = None

    def _is_output_hop(self, data: FramePushed) -> bool:
        """Determines if this hop delivers frame directly into transport_output."""
        if self._output_processor is not None:
            return data.destination is self._output_processor
        try:
            from pipecat.transports.base_output import BaseTransportOutput
            if isinstance(data.destination, BaseTransportOutput):
                return True
        except ImportError:
            pass
        return False

    async def on_push_frame(self, data: FramePushed):
        """
        Observes frame flow between pipeline processors.
        Uses FramePushed attributes: data.source, data.destination, data.frame, data.direction, data.timestamp.
        """
        if data.direction != FrameDirection.DOWNSTREAM:
            return

        frame = data.frame
        frame_id = getattr(frame, "id", None)
        if frame_id:
            hop_key = (frame_id, id(data.destination))
            if hop_key in self._processed_frame_hops:
                return
            self._processed_frame_hops.add(hop_key)
            self._hop_history.append(hop_key)
            if len(self._processed_frame_hops) > len(self._hop_history):
                self._processed_frame_hops = set(self._hop_history)

        now = self._clock()

        # Ignore frames explicitly marked as filler
        if getattr(frame, "is_filler", False) or getattr(frame, "filler", False):
            return

        # 1. User Started Speaking (Activate turn ONCE per speech frame)
        if isinstance(frame, (VADUserStartedSpeakingFrame, UserStartedSpeakingFrame)):
            if frame_id and frame_id in self._processed_event_frame_ids:
                return
            if frame_id:
                self._processed_event_frame_ids.add(frame_id)
                self._event_history.append(frame_id)

            if self._turn_active and self._t_speech_end is not None:
                await self._finalize_turn()

            self._reset_turn()
            self._turn_active = True
            self._turn_index += 1

        # 2. User Stopped Speaking (VAD Definitive Determination)
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            if frame_id and frame_id in self._processed_event_frame_ids:
                return
            if frame_id:
                self._processed_event_frame_ids.add(frame_id)
                self._event_history.append(frame_id)

            if not self._turn_active:
                self._turn_active = True
                self._turn_index += 1

            stop_secs = getattr(frame, "stop_secs", 0.0)
            self._t_speech_end = now - stop_secs

        elif isinstance(frame, UserStoppedSpeakingFrame):
            if frame_id and frame_id in self._processed_event_frame_ids:
                return
            if frame_id:
                self._processed_event_frame_ids.add(frame_id)
                self._event_history.append(frame_id)

            if not self._turn_active:
                self._turn_active = True
                self._turn_index += 1

            if self._t_speech_end is None:
                self._t_speech_end = now

        # 3. TranscriptionFrame: Keep the LAST finalized transcript before first LLMTextFrame
        elif isinstance(frame, TranscriptionFrame):
            text = (frame.text or "").strip()
            if self._turn_active and text and self._t_first_llm_token is None:
                self._last_stt_final_time = now

        # 4. LLMTextFrame: First substantive LLM token arrival
        elif isinstance(frame, LLMTextFrame):
            if self._turn_active and self._t_first_llm_token is None:
                self._t_first_llm_token = now

        # 5. TTSAudioRawFrame: Recorded ONLY at the hop where data.destination is transport_output
        elif isinstance(frame, TTSAudioRawFrame):
            if self._turn_active and self._is_output_hop(data):
                # Filter out audio from pre-LLM filler contexts if an LLM token hasn't arrived
                if self._t_first_llm_token is not None and self._t_first_tts_audio is None:
                    self._t_first_tts_audio = now
                    self._active_tts_context_id = getattr(frame, "context_id", None)

        # 6. OutputImageRawFrame: Simli avatar video frame at output hop
        elif isinstance(frame, OutputImageRawFrame):
            if self._turn_active and self._is_output_hop(data):
                if self._t_first_tts_audio is not None and self._t_first_simli_frame is None:
                    self._t_first_simli_frame = now

        # 7. TTSStoppedFrame: End of TTS playback
        elif isinstance(frame, TTSStoppedFrame):
            if self._turn_active and self._t_first_tts_audio is not None:
                # If context_id is present, ensure it matches the active substantive TTS context
                frame_ctx = getattr(frame, "context_id", None)
                if (
                    self._active_tts_context_id is None
                    or frame_ctx is None
                    or frame_ctx == self._active_tts_context_id
                ):
                    self._t_tts_stopped = now
                    await self._finalize_turn()

        # 8. Interruption
        elif isinstance(frame, InterruptionFrame):
            self._reset_turn()

        # 9. Session End
        elif isinstance(frame, EndFrame):
            if self._turn_active and self._t_speech_end is not None:
                await self._finalize_turn()

    def _reset_turn(self):
        self._turn_active = False
        self._active_tts_context_id = None
        self._t_speech_end = None
        self._last_stt_final_time = None
        self._t_first_llm_token = None
        self._t_first_tts_audio = None
        self._t_first_simli_frame = None
        self._t_tts_stopped = None

    async def _finalize_turn(self):
        t_speech_end = self._t_speech_end
        t_stt = self._last_stt_final_time
        t_llm = self._t_first_llm_token
        t_tts = self._t_first_tts_audio
        t_simli = self._t_first_simli_frame

        s1 = round((t_stt - t_speech_end) * 1000, 2) if (t_stt is not None and t_speech_end is not None) else None
        s2 = round((t_llm - t_stt) * 1000, 2) if (t_llm is not None and t_stt is not None) else None
        s3 = round((t_tts - t_llm) * 1000, 2) if (t_tts is not None and t_llm is not None) else None

        is_avatar_active = bool(self._get_avatar_state()) if self._get_avatar_state else False
        s4 = round((t_simli - t_tts) * 1000, 2) if (is_avatar_active and t_simli is not None and t_tts is not None) else None
        s5 = round((t_tts - t_speech_end) * 1000, 2) if (t_tts is not None and t_speech_end is not None) else None

        for name, val in [("stage_1_speech_end_to_stt_ms", s1),
                          ("stage_2_transcript_to_first_llm_token_ms", s2),
                          ("stage_3_first_llm_token_to_first_tts_audio_ms", s3),
                          ("stage_4_first_tts_audio_to_first_simli_frame_ms", s4),
                          ("stage_5_speech_end_to_first_substantive_audio_ms", s5)]:
            if val is not None and val < 0:
                logger.warning(f"[PipelineLatencyTimer] {name} is negative: {val}ms")

        is_complete = all(x is not None for x in (s1, s2, s3, s4, s5)) if is_avatar_active else all(x is not None for x in (s1, s2, s3, s5))

        record = {
            "turn": self._turn_index,
            "timestamp": time.time(),
            "stage_1_speech_end_to_stt_ms": s1,
            "stage_2_transcript_to_first_llm_token_ms": s2,
            "stage_3_first_llm_token_to_first_tts_audio_ms": s3,
            "stage_4_first_tts_audio_to_first_simli_frame_ms": s4,
            "stage_5_speech_end_to_first_substantive_audio_ms": s5,
            "actual_avatar_active": is_avatar_active,
            "complete": is_complete,
        }

        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        self._reset_turn()

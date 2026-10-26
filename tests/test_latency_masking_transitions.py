"""
Live Brain - Verification Suite for Batches 19-21
State Transitions & Latency-Masking Verification

Verifies without running full live bot session:
1. listening / thinking state fires on VAD-start / VAD-stop
2. speaking state fires on FIRST audio chunk (TTSAudioRawFrame), BEFORE full TTS completion (TTSStoppedFrame)
3. filler cue fires only when 4000ms ceiling is actually exceeded, and never fires when audio arrives earlier
"""

import asyncio
import os
import sys
import time
from pathlib import Path

from live_brain.listening_loop_ux import (
    AvatarMicroReactionController,
    AvatarState,
    MicroReactionType,
    FILLER_CUE_TIMEOUT_SECONDS,
)
from pipecat.frames.frames import (
    Frame,
    StartFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    TranscriptionFrame,
    LLMTextFrame,
    TTSStartedFrame,
    TTSAudioRawFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection


class MockSink:
    def __init__(self):
        self.frames = []

    async def push_frame(self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM):
        self.frames.append((frame, time.monotonic()))


async def test_behavior_1_vad_transitions():
    """1. Test that listening state transitions on VAD-start and thinking/listening-hold on VAD-stop."""
    print("\n--- [Check 1: VAD Start & VAD Stop State Transitions] ---")
    state_log = []
    
    controller = AvatarMicroReactionController(
        on_state_change=lambda s: state_log.append((s, time.monotonic())),
    )
    sink = MockSink()
    controller.push_frame = sink.push_frame

    # StartFrame -> AVATAR_IDLE_LISTENING
    await controller.process_frame(StartFrame(), FrameDirection.DOWNSTREAM)
    assert controller.current_state == AvatarState.AVATAR_IDLE_LISTENING, f"Initial state not IDLE: {controller.current_state}"
    print("  [+] StartFrame -> AVATAR_IDLE_LISTENING verified")

    # User starts speaking -> AVATAR_IDLE_LISTENING
    await controller.process_frame(UserStartedSpeakingFrame(), FrameDirection.DOWNSTREAM)
    assert controller._is_candidate_speaking is True, "Candidate speaking flag not set"
    assert controller.current_state == AvatarState.AVATAR_IDLE_LISTENING, "State not IDLE_LISTENING on user start"
    print("  [+] UserStartedSpeakingFrame -> candidate_speaking=True & state=AVATAR_IDLE_LISTENING verified")

    # User stops speaking (VAD Stop) -> AVATAR_THINKING (holding state)
    await controller.process_frame(UserStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)
    assert controller._is_candidate_speaking is False, "Candidate speaking flag not cleared on VAD stop"
    assert controller.current_state == AvatarState.AVATAR_THINKING, f"State not THINKING on VAD stop: {controller.current_state}"
    assert controller._ceiling_timer_task is not None and not controller._ceiling_timer_task.done(), "4000ms ceiling timer not armed on VAD stop"
    print("  [+] UserStoppedSpeakingFrame -> candidate_speaking=False, state=AVATAR_THINKING, and 4000ms timer armed verified")

    # Cleanup
    controller._cancel_ceiling_task()
    return True


async def test_behavior_2_cut_to_speaking_on_first_chunk():
    """2. Test speaking state fires on FIRST audio chunk, before full TTS completion."""
    print("\n--- [Check 2: Speaking State Fires on FIRST Audio Chunk, not TTS Start or End] ---")
    state_log = []
    
    controller = AvatarMicroReactionController(
        on_state_change=lambda s: state_log.append((s, time.monotonic())),
    )
    sink = MockSink()
    controller.push_frame = sink.push_frame

    await controller.process_frame(StartFrame(), FrameDirection.DOWNSTREAM)
    await controller.process_frame(UserStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)
    assert controller.current_state == AvatarState.AVATAR_THINKING

    # Step A: LLM tokens arrive -> avatar remains in THINKING
    await controller.process_frame(LLMTextFrame(text="Hello"), FrameDirection.DOWNSTREAM)
    assert controller.current_state == AvatarState.AVATAR_THINKING, f"Avatar unexpectedly changed state on LLM token: {controller.current_state}"
    print("  [+] LLMTextFrame -> state remains AVATAR_THINKING")

    # Step B: TTS synthesis request begins (TTSStartedFrame) -> avatar remains in THINKING
    await controller.process_frame(TTSStartedFrame(), FrameDirection.DOWNSTREAM)
    assert controller.current_state == AvatarState.AVATAR_THINKING, f"Avatar unexpectedly changed state on TTSStartedFrame: {controller.current_state}"
    print("  [+] TTSStartedFrame -> state remains AVATAR_THINKING (request started, but no audio yet)")

    # Step C: FIRST audio chunk arrives (TTSAudioRawFrame) -> IMMEDIATE transition to AVATAR_SPEAKING
    t_audio_start = time.monotonic()
    first_chunk = TTSAudioRawFrame(audio=b"\x00\x00" * 160, sample_rate=16000, num_channels=1)
    await controller.process_frame(first_chunk, FrameDirection.DOWNSTREAM)
    assert controller.current_state == AvatarState.AVATAR_SPEAKING, f"Avatar failed to transition to AVATAR_SPEAKING on first audio chunk: {controller.current_state}"
    assert controller._ceiling_timer_task is None, "Ceiling timer was not cancelled on first audio chunk"
    print("  [+] TTSAudioRawFrame (Chunk 1) -> immediate cut to AVATAR_SPEAKING & ceiling timer cancelled verified")

    # Step D: Subsequent audio chunks arrive (Chunks 2..5) -> state remains AVATAR_SPEAKING without duplicate events
    speaking_events_before = len([s for s, _ in state_log if s == AvatarState.AVATAR_SPEAKING])
    for i in range(2, 6):
        chunk = TTSAudioRawFrame(audio=b"\x00\x00" * 160, sample_rate=16000, num_channels=1)
        await controller.process_frame(chunk, FrameDirection.DOWNSTREAM)
        assert controller.current_state == AvatarState.AVATAR_SPEAKING
    speaking_events_after = len([s for s, _ in state_log if s == AvatarState.AVATAR_SPEAKING])
    assert speaking_events_before == speaking_events_after == 1, "Duplicate AVATAR_SPEAKING state events fired on subsequent chunks"
    print("  [+] Subsequent TTSAudioRawFrame chunks -> state stable in AVATAR_SPEAKING without duplicate state events")

    # Step E: Full TTS utterance completes (TTSStoppedFrame) -> transition back to AVATAR_IDLE_LISTENING
    await controller.process_frame(TTSStoppedFrame(), FrameDirection.DOWNSTREAM)
    assert controller.current_state == AvatarState.AVATAR_IDLE_LISTENING, f"Avatar failed to return to IDLE on TTSStoppedFrame: {controller.current_state}"
    print("  [+] TTSStoppedFrame (Utterance End) -> returns to AVATAR_IDLE_LISTENING verified")

    return True


async def test_behavior_3_filler_cue_timing():
    """3. Test filler cue fires ONLY when 4000ms ceiling is exceeded, and NEVER when audio arrives earlier."""
    print("\n--- [Check 3: Filler Cue Ceiling Timing & Early Arrival Suppression] ---")
    
    # Sub-case 3A: Audio arrives EARLIER than 4000ms (e.g. fast LLM+TTS at 300ms) -> NO filler cue
    reactions_fast = []
    state_log_fast = []
    fast_controller = AvatarMicroReactionController(
        on_reaction=lambda r: reactions_fast.append((r, time.monotonic())),
        on_state_change=lambda s: state_log_fast.append((s, time.monotonic())),
        ttfb_ceiling_seconds=0.4, # fast test duration: 400ms
    )
    fast_controller.push_frame = MockSink().push_frame

    await fast_controller.process_frame(StartFrame(), FrameDirection.DOWNSTREAM)
    await fast_controller.process_frame(UserStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)
    assert fast_controller.current_state == AvatarState.AVATAR_THINKING

    # Audio arrives at 100ms (well before 400ms ceiling)
    await asyncio.sleep(0.1)
    await fast_controller.process_frame(
        TTSAudioRawFrame(audio=b"\x00\x00" * 160, sample_rate=16000, num_channels=1),
        FrameDirection.DOWNSTREAM,
    )
    # Wait past the original 400ms mark to confirm no filler cue fires late
    await asyncio.sleep(0.4)
    filler_reactions_fast = [r for r, _ in reactions_fast if r.get("type") == "avatar_latency_filler"]
    assert len(filler_reactions_fast) == 0, f"Filler cue incorrectly fired when audio arrived early: {filler_reactions_fast}"
    assert fast_controller.current_state == AvatarState.AVATAR_SPEAKING
    print("  [+] Sub-case 3A: Audio arrived early (100ms < 400ms) -> 0 filler cues fired, cancelled cleanly")

    # Sub-case 3B: Audio is delayed past ceiling (e.g. timeout exceeds 400ms) -> FILLER CUE FIRES
    reactions_slow = []
    state_log_slow = []
    slow_controller = AvatarMicroReactionController(
        on_reaction=lambda r: reactions_slow.append((r, time.monotonic())),
        on_state_change=lambda s: state_log_slow.append((s, time.monotonic())),
        ttfb_ceiling_seconds=0.3, # fast test duration: 300ms
    )
    slow_controller.push_frame = MockSink().push_frame

    await slow_controller.process_frame(StartFrame(), FrameDirection.DOWNSTREAM)
    t_vad_stop = time.monotonic()
    await slow_controller.process_frame(UserStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)
    assert slow_controller.current_state == AvatarState.AVATAR_THINKING

    # LLM text arrived at 100ms, but NO TTS audio arrived yet
    await asyncio.sleep(0.1)
    await slow_controller.process_frame(LLMTextFrame(text="Processing"), FrameDirection.DOWNSTREAM)
    assert slow_controller.current_state == AvatarState.AVATAR_THINKING

    # Sleep until ceiling expires (total 350ms > 300ms)
    await asyncio.sleep(0.25)
    t_filler_fired = time.monotonic()

    filler_reactions_slow = [r for r, _ in reactions_slow if r.get("type") == "avatar_latency_filler"]
    assert len(filler_reactions_slow) == 1, f"Expected 1 filler cue, got {len(filler_reactions_slow)}"
    assert filler_reactions_slow[0]["reaction"] == MicroReactionType.STILL_PROCESSING.value
    assert slow_controller.current_state == AvatarState.AVATAR_FILLER
    delay_ms = (t_filler_fired - t_vad_stop) * 1000
    print(f"  [+] Sub-case 3B: Audio delayed past ceiling ({delay_ms:.1f}ms > 300ms) -> [FILLER_CUE_FIRED] accurately triggered")

    # Now when TTS audio eventually arrives after filler cue -> cleanly transitions to AVATAR_SPEAKING
    await slow_controller.process_frame(
        TTSAudioRawFrame(audio=b"\x00\x00" * 160, sample_rate=16000, num_channels=1),
        FrameDirection.DOWNSTREAM,
    )
    assert slow_controller.current_state == AvatarState.AVATAR_SPEAKING
    print("  [+] Sub-case 3B (cont.): Delayed audio arrives after filler -> cuts seamlessly to AVATAR_SPEAKING")

    return True


async def main():
    print("=" * 65)
    print(" Live Brain -- State Transitions & Timing Verification")
    print(f" Default FILLER_CUE_TIMEOUT_SECONDS: {FILLER_CUE_TIMEOUT_SECONDS}s (4000ms)")
    print("=" * 65)

    try:
        await test_behavior_1_vad_transitions()
        await test_behavior_2_cut_to_speaking_on_first_chunk()
        await test_behavior_3_filler_cue_timing()
        print("\n" + "=" * 65)
        print(" ALL 3 LATENCY-MASKING STATE TRANSITION CHECKS PASSED [3/3]")
        print("=" * 65)
    except AssertionError as e:
        print("\n" + "!" * 65)
        print(f" VERIFICATION FAILED: {e}")
        print("!" * 65)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

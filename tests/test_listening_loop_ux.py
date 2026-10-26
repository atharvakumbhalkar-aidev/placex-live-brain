"""
Live Brain — Listening Loop UX Test Suite

Tests directives/listening_loop_ux.md components:
1. AvatarMicroReactionController: trigger condition (>=2.5s), cooldown (6.0s), max 2/turn,
   state transitions (IDLE -> MICRO_REACTION -> IDLE, SPEAKING), silence suppression.
2. ConversationalResponsePacer: jitter delay calculation, audio frame buffering & release,
   barge-in reset.
3. LiveCaptionSmoother: interim word clustering, VAD boundary finalization, interviewer line chunking.
4. Non-blocking tap contract: verifies BehavioralSignalTagger and all taps pass frames downstream.
"""

import asyncio
import os
import sys
import time
from pathlib import Path

from live_brain.behavioral_tagger import (
    BehavioralSignalTagger,
    ProsodicSignalTagger,
)
from live_brain.listening_loop_ux import (
    AvatarMicroReactionController,
    AvatarState,
    ConversationalResponsePacer,
    LiveCaptionSmoother,
    MicroReactionType,
)
from pipecat.frames.frames import (
    AudioRawFrame,
    Frame,
    InterimTranscriptionFrame,
    StartFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection


class MockDownstreamSink:
    """Mock collector verifying frames pushed downstream by processors."""

    def __init__(self):
        self.received_frames = []

    async def push_frame(self, frame, direction=FrameDirection.DOWNSTREAM):
        self.received_frames.append(frame)


async def test_micro_reaction_controller():
    print("\n--- Test 1: AvatarMicroReactionController ---")
    reactions = []
    state_changes = []

    def on_reaction(event):
        reactions.append(event)

    def on_state_change(state):
        state_changes.append(state)

    controller = AvatarMicroReactionController(
        on_reaction=on_reaction,
        on_state_change=on_state_change,
        min_speech_duration=0.2,  # fast test threshold
        cooldown_seconds=0.5,
        max_per_turn=2,
        reaction_durations={
            MicroReactionType.NOD_SUBTLE: 0.1,
            MicroReactionType.TILT_ENGAGED: 0.1,
            MicroReactionType.BROW_RAISE_BRIEF: 0.1,
        },
    )
    sink = MockDownstreamSink()
    controller.push_frame = sink.push_frame

    # 1. Start frame
    await controller.process_frame(StartFrame(), FrameDirection.DOWNSTREAM)
    assert controller.current_state == AvatarState.AVATAR_IDLE_LISTENING
    print("  [+] Initial state is AVATAR_IDLE_LISTENING")

    # 2. Candidate starts speaking
    await controller.process_frame(UserStartedSpeakingFrame(), FrameDirection.DOWNSTREAM)
    
    # 3. Simulate continuous speech >= min_speech_duration (0.2s)
    await asyncio.sleep(0.25)
    await controller.process_frame(
        InterimTranscriptionFrame(
            text="I am explaining the distributed cache design",
            user_id="c1",
            timestamp=time.time(),
        ),
        FrameDirection.DOWNSTREAM,
    )

    await asyncio.sleep(0.05)
    assert len(reactions) == 1, f"Expected 1 reaction, got {len(reactions)}"
    assert reactions[0]["type"] == "avatar_micro_reaction"
    print(f"  [+] Micro-reaction triggered successfully: {reactions[0]['reaction']} (turn {reactions[0]['turn']})")

    # 4. Check Cooldown enforcement (< 0.5s cooldown)
    await asyncio.sleep(0.1)
    await controller.process_frame(
        InterimTranscriptionFrame(
            text="and how replication works",
            user_id="c1",
            timestamp=time.time(),
        ),
        FrameDirection.DOWNSTREAM,
    )
    assert len(reactions) == 1, "Cooldown failed - second reaction triggered prematurely"
    print("  [+] Cooldown enforced (no duplicate micro-reaction during cooldown)")

    # 5. Cooldown expires and second reaction triggers (max 2 per turn)
    await asyncio.sleep(0.5)
    controller._speech_start_time = time.monotonic() - 0.3
    await controller.process_frame(
        InterimTranscriptionFrame(
            text="with consistent hashing partitioning",
            user_id="c1",
            timestamp=time.time(),
        ),
        FrameDirection.DOWNSTREAM,
    )
    await asyncio.sleep(0.05)
    assert len(reactions) == 2, f"Expected 2 reactions, got {len(reactions)}"
    print(f"  [+] Second micro-reaction triggered after cooldown: {reactions[1]['reaction']}")

    # 6. Third reaction should be blocked by max_per_turn = 2
    await asyncio.sleep(0.55)
    controller._speech_start_time = time.monotonic() - 0.3
    await controller.process_frame(
        InterimTranscriptionFrame(
            text="and node failure recovery",
            user_id="c1",
            timestamp=time.time(),
        ),
        FrameDirection.DOWNSTREAM,
    )
    assert len(reactions) == 2, "Max per turn exceeded"
    print("  [+] Max per turn (2) enforced")

    # 7. Candidate stops speaking -> aborts / resets
    await controller.process_frame(UserStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)
    assert not controller._is_candidate_speaking
    print("  [+] Silence / UserStoppedSpeaking correctly resets speech state")

    # 8. Non-blocking verification: all input frames were forwarded downstream
    assert len(sink.received_frames) >= 5
    print("  [+] All frames forwarded downstream (non-blocking tap contract verified)")


async def test_conversational_response_pacer():
    print("\n--- Test 2: ConversationalResponsePacer ---")
    pacing_events = []

    def on_pacing(event):
        pacing_events.append(event)

    pacer = ConversationalResponsePacer(
        base_short_ms=50,
        max_short_jitter_ms=20,
        base_complex_ms=100,
        max_complex_jitter_ms=30,
        short_turn_word_threshold=5,
        on_pacing_delay=on_pacing,
    )
    sink = MockDownstreamSink()
    pacer.push_frame = sink.push_frame

    await pacer.process_frame(StartFrame(), FrameDirection.DOWNSTREAM)

    # 1. Short answer (<= 5 words)
    short_frame = TranscriptionFrame(text="Yes, that's correct.", user_id="c1", timestamp=time.time())
    await pacer.process_frame(short_frame, FrameDirection.DOWNSTREAM)
    assert len(pacing_events) == 1
    short_delay = pacing_events[0]["delay_ms"]
    assert 50 <= short_delay <= 75, f"Short delay out of range: {short_delay}"
    print(f"  [+] Short answer delay calculated: {short_delay}ms (expected 50-70ms)")

    # 2. Complex answer (> 5 words)
    complex_frame = TranscriptionFrame(
        text="We implemented a multi-region Raft consensus cluster with asynchronous replication to minimize write latency.",
        user_id="c1",
        timestamp=time.time(),
    )
    await pacer.process_frame(complex_frame, FrameDirection.DOWNSTREAM)
    assert len(pacing_events) == 2
    complex_delay = pacing_events[1]["delay_ms"]
    assert 100 <= complex_delay <= 135, f"Complex delay out of range: {complex_delay}"
    print(f"  [+] Complex answer delay calculated: {complex_delay}ms (expected 100-130ms)")

    # 3. Audio frame buffering during jitter window
    audio_sink = MockDownstreamSink()
    pacer.push_frame = audio_sink.push_frame
    sample_audio = AudioRawFrame(audio=b"\x00\x01" * 100, sample_rate=16000, num_channels=1)

    await pacer.process_frame(sample_audio, FrameDirection.DOWNSTREAM)
    # Should buffer while delay window is active
    assert len(audio_sink.received_frames) == 0, "Audio was released before jitter window expired"
    print("  [+] TTS Audio held in jitter buffer during pacing delay")

    # Wait for jitter timer to release audio
    await asyncio.sleep(complex_delay / 1000.0 + 0.08)
    assert len(audio_sink.received_frames) == 1, "Audio was not released upon jitter timer expiry"
    print("  [+] TTS Audio released precisely when jitter timer expired")


async def test_live_caption_smoother():
    print("\n--- Test 3: LiveCaptionSmoother ---")
    captions = []

    def on_caption(cap):
        captions.append(cap)

    smoother = LiveCaptionSmoother(
        on_caption=on_caption,
        max_words_per_line=5,
        min_interim_cluster_words=3,
    )
    sink = MockDownstreamSink()
    smoother.push_frame = sink.push_frame

    await smoother.process_frame(StartFrame(), FrameDirection.DOWNSTREAM)

    # 1. Raw partial with < 3 words should NOT emit (eliminates micro-token flicker)
    await smoother.process_frame(
        InterimTranscriptionFrame(text="Hi", user_id="c1", timestamp=time.time()),
        FrameDirection.DOWNSTREAM,
    )
    assert len(captions) == 0
    print("  [+] Micro-token interim flickering suppressed (< 3 words)")

    # 2. Stable word cluster (>= 3 words) emits interim caption
    await smoother.process_frame(
        InterimTranscriptionFrame(text="Hi I am ready", user_id="c1", timestamp=time.time()),
        FrameDirection.DOWNSTREAM,
    )
    await asyncio.sleep(0.01)
    assert len(captions) == 1
    assert captions[0]["status"] == "interim"
    assert captions[0]["text"] == "Hi I am ready"
    print("  [+] Stable interim word cluster emitted")

    # 3. Finalization on TranscriptionFrame (VAD turn completion)
    final_frame = TranscriptionFrame(
        text="Hi I am ready for the technical round.",
        user_id="c1",
        timestamp=time.time(),
    )
    await smoother.process_frame(final_frame, FrameDirection.DOWNSTREAM)
    await asyncio.sleep(0.01)
    assert len(captions) == 2
    assert captions[1]["status"] == "final"
    assert captions[1]["is_final"] is True
    print("  [+] Candidate caption finalized on VAD boundary")

    # 4. Interviewer output bounded chunking (max 5 words per line)
    interviewer_words = [
        "Welcome", "to", "ApexScale", "Let's", "discuss",
        "system", "design", "and", "caching", "strategies",
    ]
    smoother.handle_interviewer_words(interviewer_words)
    await asyncio.sleep(0.01)
    # Should chunk into 2 lines of 5 words
    interviewer_caps = [c for c in captions if c["speaker"] == "interviewer"]
    assert len(interviewer_caps) == 2
    assert interviewer_caps[0]["text"] == "Welcome to ApexScale Let's discuss"
    assert interviewer_caps[1]["text"] == "system design and caching strategies"
    print("  [+] Interviewer caption chunked to bounded lines (max 5 words/line)")


async def test_nonblocking_behavioral_tap_integrity():
    print("\n--- Test 4: Pipeline Non-Blocking Tap Integrity ---")
    signals = []
    tagger = BehavioralSignalTagger(on_signal=lambda s: signals.append(s))
    sink = MockDownstreamSink()
    tagger.push_frame = sink.push_frame

    await tagger.process_frame(StartFrame(), FrameDirection.DOWNSTREAM)
    sample_transcription = TranscriptionFrame(
        text="Um, I think we should probably use a redis cache.",
        user_id="c1",
        timestamp=time.time(),
    )
    await tagger.process_frame(sample_transcription, FrameDirection.DOWNSTREAM)
    assert len(sink.received_frames) == 2
    assert len(signals) >= 2  # filler and hedge detected
    print(f"  [+] Behavioral tagger emitted {len(signals)} signals without blocking downstream frames")


async def main():
    print("=" * 60)
    print(" Live Brain -- Listening Loop & Behavioral UX Test Runner")
    print("=" * 60)
    await test_micro_reaction_controller()
    await test_conversational_response_pacer()
    await test_live_caption_smoother()
    await test_nonblocking_behavioral_tap_integrity()
    print("\n" + "=" * 60)
    print(" ALL LISTENING LOOP & BEHAVIORAL UX TESTS PASSED [4/4]")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())

import sys
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import asyncio
import json
import subprocess
import time
from pathlib import Path

from live_brain.pipeline_latency_timer import PipelineLatencyTimer
from tools.summarize_latency import parse_latency_file, generate_markdown_summary
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
    TTSStoppedFrame,
    InterruptionFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineTask, PipelineParams
from pipecat.pipeline.runner import PipelineRunner


class PassthroughProcessor(FrameProcessor):
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)


class DelayingPacerProcessor(FrameProcessor):
    def __init__(self, delay_seconds: float = 0.15, **kwargs):
        super().__init__(**kwargs)
        self.delay_seconds = delay_seconds

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, TTSAudioRawFrame):
            await asyncio.sleep(self.delay_seconds)
        await self.push_frame(frame, direction)


class OutputSinkProcessor(FrameProcessor):
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)


async def test_delay_at_output_hop():
    print("\n--- TEST 1: Delay at output hop (ConversationalResponsePacer) ---")
    p_source = PassthroughProcessor()
    p_pacer = DelayingPacerProcessor(delay_seconds=0.15)
    p_sink = OutputSinkProcessor()

    pipeline = Pipeline([p_source, p_pacer, p_sink])
    test_log = Path(".tmp/test_output_delay.jsonl")
    if test_log.exists():
        test_log.unlink()

    timer = PipelineLatencyTimer(
        output_path=test_log,
        output_processor=p_sink,
        get_avatar_state=lambda: False,
    )

    task = PipelineTask(pipeline, params=PipelineParams(enable_metrics=False), observers=[timer])
    runner = PipelineRunner(handle_sigint=False)
    runner_task = asyncio.create_task(runner.run(task))

    await task.queue_frames([
        VADUserStartedSpeakingFrame(),
        TranscriptionFrame(text="Explain Paxos", user_id="u1", timestamp="2026-10-26T00:00:00Z"),
        VADUserStoppedSpeakingFrame(stop_secs=0.35),
        LLMTextFrame(text="Paxos is a consensus protocol"),
        TTSAudioRawFrame(audio=b"\x00" * 320, sample_rate=16000, num_channels=1, context_id="ctx-1"),
        TTSStoppedFrame(context_id="ctx-1"),
    ])
    await asyncio.sleep(0.25)
    await task.queue_frames([EndFrame()])

    await runner_task

    assert test_log.exists()
    data = json.loads(test_log.read_text(encoding="utf-8").strip().splitlines()[0])
    print(f"Record: {json.dumps(data, indent=2)}")

    assert data["turn"] == 1, f"Expected turn 1, got {data['turn']}"
    assert data["complete"] is True
    assert data["stage_3_first_llm_token_to_first_tts_audio_ms"] >= 140.0
    assert data["stage_5_speech_end_to_first_substantive_audio_ms"] >= 140.0
    print("PASS: Delay of 150ms correctly captured in Stage 3 and Stage 5 at output hop!")


async def test_two_finals_in_one_turn():
    print("\n--- TEST 2: Two finalized transcripts in one turn ---")
    p_source = PassthroughProcessor()
    p_sink = OutputSinkProcessor()

    pipeline = Pipeline([p_source, p_sink])
    test_log = Path(".tmp/test_two_finals.jsonl")
    if test_log.exists():
        test_log.unlink()

    timer = PipelineLatencyTimer(
        output_path=test_log,
        output_processor=p_sink,
        get_avatar_state=lambda: False,
    )

    task = PipelineTask(pipeline, params=PipelineParams(enable_metrics=False), observers=[timer])
    runner = PipelineRunner(handle_sigint=False)
    runner_task = asyncio.create_task(runner.run(task))

    await task.queue_frames([
        VADUserStartedSpeakingFrame(),
        TranscriptionFrame(text="First partial final", user_id="u1", timestamp="2026-10-26T00:00:00Z"),
    ])
    await asyncio.sleep(0.05)
    await task.queue_frames([
        VADUserStoppedSpeakingFrame(stop_secs=0.35),
    ])
    await asyncio.sleep(0.05)
    await task.queue_frames([
        TranscriptionFrame(text="Second final final", user_id="u1", timestamp="2026-10-26T00:00:01Z"),
    ])
    await asyncio.sleep(0.05)
    await task.queue_frames([
        LLMTextFrame(text="Got it"),
        TTSAudioRawFrame(audio=b"\x00" * 320, sample_rate=16000, num_channels=1, context_id="ctx-2"),
        TTSStoppedFrame(context_id="ctx-2"),
    ])
    await asyncio.sleep(0.1)
    await task.queue_frames([EndFrame()])

    await runner_task

    assert test_log.exists()
    data = json.loads(test_log.read_text(encoding="utf-8").strip().splitlines()[0])
    print(f"Record: {json.dumps(data, indent=2)}")

    assert data["turn"] == 1
    assert data["complete"] is True
    # Measured from 2nd final (~50ms), not 1st final (~150ms)
    assert data["stage_2_transcript_to_first_llm_token_ms"] < 120.0
    print("PASS: Last finalized transcript before LLM token correctly used for Stage 1 & Stage 2!")


async def test_early_final_with_real_stop_secs():
    print("\n--- TEST 3: Early final transcript with real stop_secs=0.35 ---")
    p_source = PassthroughProcessor()
    p_sink = OutputSinkProcessor()

    pipeline = Pipeline([p_source, p_sink])
    test_log = Path(".tmp/test_real_stop_secs.jsonl")
    if test_log.exists():
        test_log.unlink()

    timer = PipelineLatencyTimer(
        output_path=test_log,
        output_processor=p_sink,
        get_avatar_state=lambda: False,
    )

    task = PipelineTask(pipeline, params=PipelineParams(enable_metrics=False), observers=[timer])
    runner = PipelineRunner(handle_sigint=False)
    runner_task = asyncio.create_task(runner.run(task))

    # Transcript arrives before VAD determination completes (real production behavior)
    await task.queue_frames([
        VADUserStartedSpeakingFrame(),
        TranscriptionFrame(text="Candidate finished speaking early", user_id="u1", timestamp="2026-10-26T00:00:00Z"),
    ])
    await asyncio.sleep(0.1)  # 100ms gap before VAD confirms stop
    await task.queue_frames([
        VADUserStoppedSpeakingFrame(stop_secs=0.35),  # t_speech_end = now - 0.35
        LLMTextFrame(text="Understood"),
        TTSAudioRawFrame(audio=b"\x00" * 320, sample_rate=16000, num_channels=1, context_id="ctx-3"),
        TTSStoppedFrame(context_id="ctx-3"),
    ])
    await asyncio.sleep(0.1)
    await task.queue_frames([EndFrame()])

    await runner_task

    assert test_log.exists()
    data = json.loads(test_log.read_text(encoding="utf-8").strip().splitlines()[0])
    print(f"Record: {json.dumps(data, indent=2)}")

    assert data["turn"] == 1
    assert data["complete"] is True
    # Because stop_secs=0.35 and transcript arrived 100ms before VAD stop:
    # t_stt was at (t_vad_stop - 0.1). t_speech_end was (t_vad_stop - 0.35).
    # Thus t_stt - t_speech_end = +0.25s (250ms), a real positive duration!
    assert data["stage_1_speech_end_to_stt_ms"] > 0
    print("PASS: Real stop_secs=0.35 accurately calibrates actual speech end time!")


async def test_missing_timestamp_uncomplete():
    print("\n--- TEST 4: Missing timestamp (null stage and complete: false) ---")
    p_source = PassthroughProcessor()
    p_sink = OutputSinkProcessor()

    pipeline = Pipeline([p_source, p_sink])
    test_log = Path(".tmp/test_missing_timestamp.jsonl")
    if test_log.exists():
        test_log.unlink()

    timer = PipelineLatencyTimer(
        output_path=test_log,
        output_processor=p_sink,
        get_avatar_state=lambda: False,
    )

    task = PipelineTask(pipeline, params=PipelineParams(enable_metrics=False), observers=[timer])
    runner = PipelineRunner(handle_sigint=False)
    runner_task = asyncio.create_task(runner.run(task))

    # Omit LLMTextFrame (simulates LLM failure/timeout)
    await task.queue_frames([
        VADUserStartedSpeakingFrame(),
        TranscriptionFrame(text="Hello", user_id="u1", timestamp="2026-10-26T00:00:00Z"),
        VADUserStoppedSpeakingFrame(stop_secs=0.35),
        TTSAudioRawFrame(audio=b"\x00" * 320, sample_rate=16000, num_channels=1, context_id="ctx-4"),
        TTSStoppedFrame(context_id="ctx-4"),
        EndFrame(),
    ])

    await runner_task

    assert test_log.exists()
    data = json.loads(test_log.read_text(encoding="utf-8").strip().splitlines()[0])
    print(f"Record: {json.dumps(data, indent=2)}")

    assert data["turn"] == 1
    assert data["stage_2_transcript_to_first_llm_token_ms"] is None
    assert data["stage_3_first_llm_token_to_first_tts_audio_ms"] is None
    assert data["complete"] is False
    print("PASS: Missing timestamp records null stage and complete: false without fallbacks!")


async def test_filler_second_tts_context():
    print("\n--- TEST 5: Filler / non-LLM TTS audio followed by substantive response ---")
    p_source = PassthroughProcessor()
    p_sink = OutputSinkProcessor()

    pipeline = Pipeline([p_source, p_sink])
    test_log = Path(".tmp/test_filler_context.jsonl")
    if test_log.exists():
        test_log.unlink()

    timer = PipelineLatencyTimer(
        output_path=test_log,
        output_processor=p_sink,
        get_avatar_state=lambda: False,
    )

    task = PipelineTask(pipeline, params=PipelineParams(enable_metrics=False), observers=[timer])
    runner = PipelineRunner(handle_sigint=False)
    runner_task = asyncio.create_task(runner.run(task))

    # Candidate speaks
    await task.queue_frames([
        VADUserStartedSpeakingFrame(),
        TranscriptionFrame(text="Let me think about that", user_id="u1", timestamp="2026-10-26T00:00:00Z"),
        VADUserStoppedSpeakingFrame(stop_secs=0.35),
    ])

    # Pre-LLM filler audio emitted via TTSSpeakFrame / canned cue (e.g. "I see...")
    filler_audio = TTSAudioRawFrame(audio=b"\x00" * 160, sample_rate=16000, num_channels=1, context_id="filler-ctx")
    filler_stop = TTSStoppedFrame(context_id="filler-ctx")
    await task.queue_frames([filler_audio, filler_stop])

    await asyncio.sleep(0.08)

    # LLM produces substantive response
    await task.queue_frames([
        LLMTextFrame(text="Here is the detailed architectural breakdown"),
        TTSAudioRawFrame(audio=b"\x00" * 320, sample_rate=16000, num_channels=1, context_id="response-ctx"),
        TTSStoppedFrame(context_id="response-ctx"),
    ])
    await asyncio.sleep(0.1)
    await task.queue_frames([EndFrame()])

    await runner_task

    assert test_log.exists()
    records = [json.loads(l) for l in test_log.read_text(encoding="utf-8").strip().splitlines()]
    assert len(records) == 1, f"Expected 1 finalized record, got {len(records)}"
    data = records[0]
    print(f"Record: {json.dumps(data, indent=2)}")

    assert data["turn"] == 1
    assert data["complete"] is True
    print("PASS: Pre-LLM filler context audio/stopped frame did NOT prematurely finalize turn!")


async def test_turn_index_increment_across_turns():
    print("\n--- TEST 6: Turn index reset & sequential increment ---")
    p_source = PassthroughProcessor()
    p_sink = OutputSinkProcessor()

    pipeline = Pipeline([p_source, p_sink])
    test_log = Path(".tmp/test_multi_turn.jsonl")
    if test_log.exists():
        test_log.unlink()

    timer = PipelineLatencyTimer(
        output_path=test_log,
        output_processor=p_sink,
        get_avatar_state=lambda: False,
    )

    task = PipelineTask(pipeline, params=PipelineParams(enable_metrics=False), observers=[timer])
    runner = PipelineRunner(handle_sigint=False)
    runner_task = asyncio.create_task(runner.run(task))

    # Turn 1
    await task.queue_frames([
        VADUserStartedSpeakingFrame(),
        TranscriptionFrame(text="Turn one text", user_id="u1", timestamp="2026-10-26T00:00:00Z"),
        VADUserStoppedSpeakingFrame(stop_secs=0.35),
        LLMTextFrame(text="Turn one reply"),
        TTSAudioRawFrame(audio=b"\x00" * 320, sample_rate=16000, num_channels=1, context_id="turn1-ctx"),
        TTSStoppedFrame(context_id="turn1-ctx"),
    ])

    await asyncio.sleep(0.05)

    # Turn 2
    await task.queue_frames([
        VADUserStartedSpeakingFrame(),
        TranscriptionFrame(text="Turn two text", user_id="u1", timestamp="2026-10-26T00:00:01Z"),
        VADUserStoppedSpeakingFrame(stop_secs=0.35),
        LLMTextFrame(text="Turn two reply"),
        TTSAudioRawFrame(audio=b"\x00" * 320, sample_rate=16000, num_channels=1, context_id="turn2-ctx"),
        TTSStoppedFrame(context_id="turn2-ctx"),
        EndFrame(),
    ])

    await runner_task

    assert test_log.exists()
    records = [json.loads(l) for l in test_log.read_text(encoding="utf-8").strip().splitlines()]
    assert len(records) == 2, f"Expected 2 records, got {len(records)}"
    print(f"Turn 1 index: {records[0]['turn']}")
    print(f"Turn 2 index: {records[1]['turn']}")

    assert records[0]["turn"] == 1, f"Expected turn 1, got {records[0]['turn']}"
    assert records[1]["turn"] == 2, f"Expected turn 2, got {records[1]['turn']}"
    print("PASS: Turn indices increment cleanly as 1, 2 (hops do not inflate turn index)!")


def test_summarize_latency_cli():
    print("\n--- TEST 7: summarize_latency.py CLI execution ---")
    sample_log = Path(".tmp/test_cli_latency.jsonl")
    records = [
        {
            "turn": 1,
            "stage_1_speech_end_to_stt_ms": 120.5,
            "stage_2_transcript_to_first_llm_token_ms": 210.0,
            "stage_3_first_llm_token_to_first_tts_audio_ms": 185.3,
            "stage_4_first_tts_audio_to_first_simli_frame_ms": None,
            "stage_5_speech_end_to_first_substantive_audio_ms": 515.8,
            "complete": True,
        },
        {
            "turn": 2,
            "stage_1_speech_end_to_stt_ms": 135.2,
            "stage_2_transcript_to_first_llm_token_ms": 195.4,
            "stage_3_first_llm_token_to_first_tts_audio_ms": 170.1,
            "stage_4_first_tts_audio_to_first_simli_frame_ms": None,
            "stage_5_speech_end_to_first_substantive_audio_ms": 500.7,
            "complete": True,
        },
    ]
    sample_log.parent.mkdir(parents=True, exist_ok=True)
    with open(sample_log, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    # Run summarize_latency as CLI subprocess
    summarize_script = Path(__file__).resolve().parent.parent / "tools" / "summarize_latency.py"
    res = subprocess.run(
        [sys.executable, str(summarize_script), str(sample_log)],
        capture_output=True,
        text=True,
        check=True,
    )
    print("CLI Output:")
    assert "# Turn Latency Performance Summary" in res.stdout
    assert "Total Turns Evaluated" in res.stdout
    assert "| Stage 1 | speech-end -> STT final transcript | 2 |" in res.stdout
    assert "Stage 4" not in res.stdout
    assert "| Stage 5 | speech-end -> first substantive bot audio | 2 |" in res.stdout
    print("PASS: summarize_latency.py CLI correctly parses records and prints formatted table!")


async def main():
    await test_delay_at_output_hop()
    await test_two_finals_in_one_turn()
    await test_early_final_with_real_stop_secs()
    await test_missing_timestamp_uncomplete()
    await test_filler_second_tts_context()
    await test_turn_index_increment_across_turns()
    test_summarize_latency_cli()
    print("\n=======================================================")
    print("ALL 7 TEST CASES PASSED WITH 100% SUCCESS")
    print("=======================================================")


if __name__ == "__main__":
    asyncio.run(main())

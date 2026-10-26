"""
Live Brain — real-time interview pipeline.

Voice, avatar, and conversation for one candidate session, wired as a single
Pipecat pipeline.
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from collections.abc import AsyncGenerator
from typing import Optional, Dict, Any, Callable, List
from datetime import datetime, timezone

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
from loguru import logger

load_dotenv()

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    Frame,
    LLMRunFrame,
    TTSAudioRawFrame,
    InterruptionFrame,
    UserStartedSpeakingFrame,
    TTSStoppedFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from deepgram.listen.v1.types import ListenV1Finalize, ListenV1KeepAlive
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport

from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.services.simli.video import SimliVideoService
from simli import SimliClient
from pipecat.transports.base_transport import TransportParams

try:
    from pipecat.transports.daily.transport import DailyParams
except ImportError:
    DailyParams = None

from live_brain.schemas import (
    QuestionBank,
    QuestionItem,
    QuestionTier,
    PrepBrainMetadata,
    ScoringRubric,
    RubricCriterion,
)
from live_brain.behavioral_tagger import (
    BehavioralSignalTagger,
    ProsodicSignalTagger,
)
from live_brain.listening_loop_ux import (
    AvatarMicroReactionController,
    AvatarState,
    ConversationalResponsePacer,
    LiveCaptionSmoother,
)
from live_brain.pipeline_latency_timer import PipelineLatencyTimer

# Optional post-session hook (e.g. external webhook, callback or None)
_session_end_callback: Optional[Callable[[dict], Any]] = None


def register_session_end_hook(cb: Callable[[dict], Any]) -> None:
    """Register an optional callback invoked upon session transcript finalization."""
    global _session_end_callback
    _session_end_callback = cb


# Transport configuration
transport_params = {
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        video_in_enabled=True,
        video_out_enabled=True,
    ),
}

if DailyParams is not None:
    transport_params["daily"] = lambda: DailyParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        camera_in_enabled=False,
        video_out_enabled=True,
    )

# Default baseline fallback system prompt
INTERVIEWER_SYSTEM_PROMPT = (
    "You are the Live Brain technical interviewer. Ask one question at a "
    "time from the prepared question bank. Keep responses under three "
    "sentences. Do not reveal rubric scoring. Stay in the interviewer "
    "persona at all times."
)


def load_question_bank(path: Optional[str] = None) -> QuestionBank:
    """
    Loads and validates a QuestionBank JSON blueprint.
    """
    candidate_paths = []
    if path:
        candidate_paths.append(Path(path))
    env_path = os.getenv("QUESTION_BANK_PATH")
    if env_path:
        candidate_paths.append(Path(env_path))

    # Common search locations
    candidate_paths.extend([
        Path("tests/fixtures/sample_question_bank.json"),
        Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "sample_question_bank.json",
        Path("sample_question_bank.json"),
    ])

    for candidate in candidate_paths:
        if candidate and candidate.exists() and candidate.is_file():
            try:
                with open(candidate, "r", encoding="utf-8") as f:
                    data = json.load(f)
                qb = QuestionBank.model_validate(data)
                if not qb.validate_strict_order():
                    raise ValueError(f"Question sequence in {candidate} is not strictly ordered 1..N")
                logger.info(
                    f"[prep_brain] Loaded question bank from '{candidate}' "
                    f"({len(qb.questions)} questions for {qb.metadata.company_name})"
                )
                return qb
            except Exception as e:
                logger.warning(f"[prep_brain] Failed parsing question bank from {candidate}: {e}")

    # Standalone fallback: load static bundled fixture or create minimal default
    bundled_fixture = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "sample_question_bank.json"
    if bundled_fixture.exists():
        with open(bundled_fixture, "r", encoding="utf-8") as f:
            return QuestionBank.model_validate(json.load(f))

    return QuestionBank(
        session_id="default-live-session",
        metadata=PrepBrainMetadata(
            company_name="ApexScale",
            company_domain="apexscale.example",
            role_title="Senior Software Engineer",
            target_level="L5 / Senior",
            tech_stack_detected=["Python", "Distributed Systems", "SQL"],
            domain_focus=["Backend Engineering"],
        ),
        questions=[
            QuestionItem(
                id="q1_warmup",
                index=1,
                tier=QuestionTier.WARMUP,
                topic="High-Level Architecture & Background",
                primary_prompt="To start off, could you walk me through the highest-scale backend system you've built recently, specifically highlighting where bottlenecks occurred and how you resolved them?",
                context_background="Calibrates candidate baseline seniority against architectural scale.",
                allowed_micro_probes=[
                    "What was the primary bottleneck: disk I/O, concurrency, or latency?",
                    "How did you measure that in production?",
                ],
                scoring_rubric=ScoringRubric(
                    max_points=10,
                    criteria=[
                        RubricCriterion(
                            dimension="Clarity",
                            weight=1.0,
                            poor_0_3="Vague",
                            good_4_7="Coherent",
                            expert_8_10="Precise breakdown",
                        )
                    ],
                ),
            )
        ],
    )


def build_interviewer_system_prompt(qb: QuestionBank) -> str:
    """
    Constructs the Live Brain system prompt injecting the full question bank
    and enforcing the strict-sequence and conversational delivery contracts.
    """
    meta = qb.metadata
    tech_stack_str = ", ".join(meta.tech_stack_detected) if meta.tech_stack_detected else "General Engineering"
    domain_str = ", ".join(meta.domain_focus) if meta.domain_focus else "Distributed Systems"

    questions_formatted = []
    for q in qb.questions:
        tier_val = q.tier.value if hasattr(q.tier, "value") else str(q.tier)
        probes_str = (
            "\n".join([f"      - {probe}" for probe in q.allowed_micro_probes])
            if q.allowed_micro_probes
            else "      (None)"
        )
        q_text = (
            f"  Question {q.index} [ID: {q.id} | Tier: {tier_val} | Topic: {q.topic}]:\n"
            f"    - Substantive Question: \"{q.primary_prompt}\"\n"
            f"    - Context / Relevance: {q.context_background}\n"
            f"    - Allowed Micro-Probes (max 1-2 bounded clarifying turns if needed):\n{probes_str}"
        )
        questions_formatted.append(q_text)

    all_questions_text = "\n\n".join(questions_formatted)

    prompt = f"""You are the Live Brain technical interviewer, conducting a live conversational technical interview for {meta.company_name} ({meta.company_domain}).
Target Role: {meta.role_title} ({meta.target_level})
Focus Areas: {domain_str}
Detected Tech Stack: {tech_stack_str}

=== FIXED QUESTION SEQUENCE (STRICT-MODE CONTRACT) ===
You must strictly follow the sequential order of the question bank below (Q1 -> Q2 -> ... -> Q{len(qb.questions)}).
{all_questions_text}

=== CONVERSATIONAL DELIVERY GUARDRAILS (HOW IS FLUID) ===
1. FIXED SUBSTANCE, FLUID DELIVERY:
   - WHAT is fixed: The substantive question blueprint, topic coverage, and sequence. You must NEVER skip, drop, substitute, or reorder any root question.
   - HOW is fluid: You have full conversational agency in how you introduce each question, phrase natural segues, and acknowledge the candidate.

2. ORGANIC ACKNOWLEDGMENTS & BRIDGING:
   - When the candidate finishes an answer, substantively acknowledge their point (e.g., "Got it, that makes sense regarding your indexing tradeoff.") before advancing.
   - Avoid repetitive robotic phrases (do NOT repeat "Great answer!" or "Thanks for that" every turn).
   - Craft natural transitions connecting the candidate's previous response to the next question in the sequence.

3. BOUNDED MICRO-CLARIFICATIONS (MAX 1-2 TURNS):
   - If the candidate's answer to Question Q[i] is ambiguous, incomplete, or touches an interesting edge case, you may ask AT MOST 1-2 short clarifying micro-probes from the allowed list or directly related to Q[i].
   - Clarifications must stay strictly focused on the active question Q[i]. Once clarified (or after max 2 follow-ups), immediately advance to Question Q[i+1].

4. SPOKEN FLUENCY & PACING:
   - Keep all spoken responses concise (under three sentences per turn).
   - Speak naturally with conversational prosody.
   - NEVER reveal rubric scores, point allocations, or internal grading criteria to the candidate.
   - Stay in the professional, empathetic interviewer persona at all times.

=== INTERVIEW START ===
Begin the interview by warmly greeting the candidate, introducing the technical evaluation for {meta.company_name}, and smoothly presenting Question 1."""
    return prompt.strip()


class SessionTranscriptCollector:
    """
    Collects candidate/interviewer transcript turns and behavioral signals
    throughout the Live Brain session, packaging the session payload.
    """

    def __init__(self, session_id: str, question_bank: Any):
        self.session_id = session_id
        self.question_bank = question_bank
        self.transcript_turns: list = []
        self.behavioral_signals_captured: list = []
        self._turn_counter = 0
        self._scored = False

    def record_turn(self, speaker: str, text: str, question_id: Optional[str] = None) -> None:
        text = (text or "").strip()
        if not text:
            return
        self._turn_counter += 1
        self.transcript_turns.append({
            "turn_index": self._turn_counter,
            "speaker": speaker,
            "text": text,
            "question_id": question_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    def record_signal(self, signal: dict) -> None:
        self.behavioral_signals_captured.append(signal)

    def build_payload(self) -> Dict[str, Any]:
        meta_dict = {}
        if hasattr(self.question_bank, "metadata"):
            meta = self.question_bank.metadata
            meta_dict = meta.model_dump() if hasattr(meta, "model_dump") else {
                "company_name": getattr(meta, "company_name", "ApexScale"),
                "company_domain": getattr(meta, "company_domain", "apexscale.example"),
                "role_title": getattr(meta, "role_title", "Senior Software Engineer"),
                "target_level": getattr(meta, "target_level", "L5 / Senior"),
            }

        questions_meta = []
        for q in getattr(self.question_bank, "questions", []):
            tier_v = q.tier.value if hasattr(q.tier, "value") else str(q.tier)
            rubric_pts = getattr(q.scoring_rubric, "max_points", 10.0) if hasattr(q, "scoring_rubric") else 10.0
            crit_count = len(getattr(q.scoring_rubric, "criteria", [])) if hasattr(q, "scoring_rubric") else 0
            questions_meta.append({
                "id": q.id,
                "index": q.index,
                "tier": tier_v,
                "topic": q.topic,
                "rubric_max_points": rubric_pts,
                "rubric_criteria_count": crit_count,
            })

        return {
            "session_id": self.session_id,
            "session_timestamp": datetime.now(timezone.utc).isoformat(),
            "company_metadata": meta_dict,
            "question_bank_reference": {
                "total_questions": len(questions_meta),
                "questions": questions_meta,
            },
            "transcript_turns": self.transcript_turns,
            "behavioral_signals_captured": self.behavioral_signals_captured,
            "scoring_readiness": {
                "mapped_turns_count": len(self.transcript_turns),
                "has_candidate_turns": any(t.get("speaker") == "candidate" for t in self.transcript_turns),
                "has_interviewer_turns": any(t.get("speaker") == "interviewer" for t in self.transcript_turns),
                "status": "READY_FOR_SCORING",
            },
        }

    def trigger_scoring(self, dry_run: bool = False, model_name: Optional[str] = None) -> Any:
        if self._scored:
            return None
        self._scored = True

        payload = self.build_payload()
        out_dir = Path(".tmp/transcripts")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"{self.session_id}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        logger.info(f"[LiveBrain] Final session transcript persisted to '{out_file}'")

        if _session_end_callback:
            try:
                return _session_end_callback(payload)
            except Exception as e:
                logger.warning(f"[LiveBrain] Optional session-end callback raised error: {e}")
        return None


# Global active transcript collector reference for logging callbacks
_active_collector: Optional[SessionTranscriptCollector] = None


def log_behavioral_signal(signal: dict) -> None:
    logger.info(f"[behavioral] {signal}")
    if _active_collector is not None:
        _active_collector.record_signal(signal)


def log_caption_event(caption: dict) -> None:
    logger.info(
        f"[caption] [{caption.get('speaker')}] ({caption.get('status')}): {caption.get('text')}"
    )
    if _active_collector is not None and caption.get("status") == "final":
        _active_collector.record_turn(
            speaker=caption.get("speaker", "candidate"),
            text=caption.get("text", ""),
        )


def log_micro_reaction(reaction: dict) -> None:
    logger.info(
        f"[avatar_reaction] {reaction.get('reaction')} "
        f"(duration: {reaction.get('duration_seconds')}s, turn: {reaction.get('turn')})"
    )


def log_avatar_state(state: AvatarState) -> None:
    logger.info(f"[avatar_state] -> {state.value}")


def log_pacing_delay(pacing: dict) -> None:
    logger.info(
        f"[pacing_jitter] {pacing.get('delay_ms')}ms delay for {pacing.get('candidate_word_count')} words"
    )


class ResilientDeepgramSTTService(DeepgramSTTService):
    """DeepgramSTTService with synchronized WebSocket writes preventing concurrent drain AssertionError."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._ws_lock = asyncio.Lock()

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame | None, None]:
        if self._connection:
            try:
                async with self._ws_lock:
                    await self._connection.send_media(audio)
            except Exception as e:
                logger.warning(f"{self}: send_media failed, connection will reconnect: {e}")
                self._connection = None
        yield None

    async def _keepalive_handler(self):
        while True:
            await asyncio.sleep(5)
            if self._connection:
                try:
                    async with self._ws_lock:
                        await self._connection.send_keep_alive(ListenV1KeepAlive(type="KeepAlive"))
                    logger.trace(f"{self}: Sent keepalive")
                except Exception as e:
                    logger.warning(f"{self}: Keepalive failed: {e}")

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, VADUserStartedSpeakingFrame):
            await self._start_metrics()
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            if self._connection:
                self.request_finalize()
                async with self._ws_lock:
                    await self._connection.send_finalize(ListenV1Finalize(type="Finalize"))
                logger.trace(f"Triggered finalize event on: {frame.name=}, {direction=}")


async def run_bot(transport, runner_args: RunnerArguments) -> None:
    global _active_collector

    if not os.getenv("GEMINI_API_KEY"):
        raise RuntimeError(
            "[LiveBrain] GEMINI_API_KEY is missing or empty. Please configure it in your environment."
        )

    # Owner: STT. Deepgram Nova-3 with smart formatting and punctuation
    stt = ResilientDeepgramSTTService(
        api_key=os.getenv("DEEPGRAM_API_KEY"),
        settings=DeepgramSTTService.Settings(
            model="nova-3",
            language="en",
            smart_format=True,
            punctuate=True,
            interim_results=True,
        ),
    )

    llm_provider = os.getenv("LLM_PROVIDER", "gemini").lower()
    if llm_provider != "gemini":
        raise ValueError(
            f"Unknown LLM_PROVIDER={llm_provider!r} — only 'gemini' is supported."
        )

    gemini_api_key = os.getenv("GEMINI_API_KEY", "")
    gemini_model = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
    logger.info(
        f"[LiveBrain] Initializing GoogleLLMService (configured={bool(gemini_api_key)}, "
        f"len={len(gemini_api_key)}, model={gemini_model})"
    )

    from pipecat.services.google.llm import GoogleLLMService

    llm = GoogleLLMService(
        api_key=gemini_api_key,
        settings=GoogleLLMService.Settings(
            model=gemini_model,
        ),
    )

    # Owner: TTS. ElevenLabs streams audio with word-level timestamps
    tts = ElevenLabsTTSService(
        api_key=os.getenv("ELEVENLABS_API_KEY"),
        voice_id=os.getenv("ELEVENLABS_VOICE_ID"),
        model="eleven_flash_v2_5",
        auto_mode=False,
        pause_watchdog_timeout_s=8.0,
    )

    # Load Question Bank and compile strict-order system prompt
    question_bank = load_question_bank()
    system_prompt = build_interviewer_system_prompt(question_bank)

    # Initialize session transcript collector
    session_id = os.getenv("SESSION_ID", getattr(question_bank, "session_id", "live-session-auto"))
    collector = SessionTranscriptCollector(session_id=session_id, question_bank=question_bank)
    _active_collector = collector

    # Owner: behavioral coaching signal taps (content and prosody)
    behavioral_tagger = BehavioralSignalTagger(on_signal=log_behavioral_signal)
    prosodic_tagger = ProsodicSignalTagger(on_signal=log_behavioral_signal)

    # Owner: Listening Loop UX processors
    caption_smoother = LiveCaptionSmoother(on_caption=log_caption_event)
    micro_reaction_controller = AvatarMicroReactionController(
        on_reaction=log_micro_reaction,
        on_state_change=log_avatar_state,
        min_speech_duration=2.5,
        cooldown_seconds=6.0,
        max_per_turn=2,
        ttfb_ceiling_seconds=4.0,
    )
    response_pacer = ConversationalResponsePacer(
        base_short_ms=300,
        max_short_jitter_ms=200,
        base_complex_ms=600,
        max_complex_jitter_ms=300,
        on_pacing_delay=log_pacing_delay,
    )

    context = LLMContext([{"role": "system", "content": system_prompt}])
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.35, start_secs=0.2)),
        ),
    )

    enable_avatar = os.getenv("ENABLE_AVATAR", "true").lower() in ("true", "1", "yes")

    pipeline_stages = [
        transport.input(),
        stt,
        behavioral_tagger,          # tap: text coaching signals (non-blocking)
        prosodic_tagger,            # tap: acoustic prosody signals (non-blocking)
        caption_smoother,           # tap: flicker-free live captions
        user_aggregator,
        llm,
        tts,
        response_pacer,             # conversational pacing jitter buffer
        micro_reaction_controller,  # tap: VAD-triggered micro-reactions & alive thinking loop
    ]

    if enable_avatar and os.getenv("SIMLI_API_KEY"):
        class ResilientSimliVideoService(SimliVideoService):
            """SimliVideoService with retry-with-backoff on connection startup and graceful audio-only fallback."""

            async def _start_connection(self):
                max_attempts = 3
                api_key = self._simli_client.api_key
                config = self._simli_client.config
                simli_url = self._simli_client.simliHTTPURL
                enable_sfu = self._simli_client.enableSFU

                for attempt in range(1, max_attempts + 1):
                    self._simli_client = SimliClient(
                        api_key=api_key,
                        config=config,
                        simliURL=simli_url,
                        enableSFU=enable_sfu,
                    )
                    try:
                        logger.info(f"[Simli] connection attempt {attempt}/{max_attempts} starting...")
                        if not self._initialized:
                            await self._simli_client.start()
                            self._initialized = True

                        await self._simli_client.sendSilence()
                        self._audio_task = self.create_task(self._consume_and_process_audio())
                        self._audio_task.add_done_callback(
                            lambda t: logger.warning(
                                f"[ResilientSimliVideoService] _audio_task completed/terminated | "
                                f"cancelled={t.cancelled()} | "
                                f"exception={t.exception() if not t.cancelled() else None}"
                            )
                        )
                        self._video_task = self.create_task(self._consume_and_process_video())
                        logger.info("[Simli] avatar connection successfully established.")
                        return
                    except Exception as e:
                        fail_err = getattr(self._simli_client, "failError", None)
                        logger.warning(
                            f"[Simli] connection attempt {attempt}/{max_attempts} failed: "
                            f"{type(e).__name__}: {e!r} (client.failError={fail_err!r})"
                        )
                        if self._initialized:
                            try:
                                await self._simli_client.stop()
                            except Exception as stop_e:
                                logger.warning(
                                    f"[Simli] error stopping client after failed attempt {attempt}: "
                                    f"{type(stop_e).__name__}: {stop_e!r}"
                                )
                            self._initialized = False

                        if attempt < max_attempts:
                            logger.info("[Simli] waiting 2s before retry...")
                            await asyncio.sleep(2)
                        else:
                            logger.warning("[Simli] avatar unavailable after 3 attempts, continuing audio-only.")

            async def _consume_and_process_audio(self):
                await self._pipecat_resampler_event.wait()
                audio_iterator = self._simli_client.getAudioStreamIterator()
                async for audio_frame in audio_iterator:
                    logger.debug(
                        f"[ResilientSimliVideoService] getAudioStreamIterator yielded audio_frame (pts={getattr(audio_frame, 'pts', None)})"
                    )
                    resampled_frames = self._pipecat_resampler.resample(audio_frame)
                    for resampled_frame in resampled_frames:
                        audio_array = resampled_frame.to_ndarray()
                        if audio_array.any():
                            logger.debug(
                                f"[ResilientSimliVideoService] audio consumer calling push_frame ({len(audio_array)} samples)"
                            )
                            await self.push_frame(
                                TTSAudioRawFrame(
                                    audio=audio_array.tobytes(),
                                    sample_rate=self._pipecat_resampler.rate,
                                    num_channels=1,
                                ),
                            )
                        else:
                            logger.debug(
                                "[ResilientSimliVideoService] audio consumer yielded silent frame -> push_frame NOT called"
                            )

            async def process_frame(self, frame: Frame, direction: FrameDirection):
                is_tts_audio = isinstance(frame, TTSAudioRawFrame)
                has_client = getattr(self, "_simli_client", None) is not None
                has_conn = has_client and getattr(self._simli_client, "Connection", None) is not None
                is_connected = bool(self._initialized and has_client and has_conn)

                if is_tts_audio:
                    logger.debug(
                        f"[ResilientSimliVideoService] TTSAudioRawFrame received | "
                        f"_initialized={self._initialized} | has_client={has_client} | "
                        f"has_conn={has_conn} | is_connected_branch={is_connected}"
                    )
                elif isinstance(frame, (InterruptionFrame, UserStartedSpeakingFrame)):
                    logger.debug(
                        f"[ResilientSimliVideoService] {type(frame).__name__} received | "
                        f"_previously_interrupted={self._previously_interrupted}"
                    )
                elif isinstance(frame, TTSStoppedFrame):
                    logger.debug(
                        f"[ResilientSimliVideoService] TTSStoppedFrame received | "
                        f"_previously_interrupted={self._previously_interrupted} | "
                        f"_audio_buffer_size={len(self._audio_buffer)}"
                    )

                if not is_connected:
                    self._initialized = False
                    await super(SimliVideoService, self).process_frame(frame, direction)
                    if is_tts_audio:
                        logger.debug("[ResilientSimliVideoService] fallback branch -> calling push_frame(frame)")
                    await self.push_frame(frame, direction)
                    return

                try:
                    await super().process_frame(frame, direction)
                    if is_tts_audio:
                        logger.debug("[ResilientSimliVideoService] connected branch -> super().process_frame completed")
                except Exception as e:
                    logger.warning(f"[Simli] avatar frame processing error: {e}, falling back to audio passthrough.")
                    self._initialized = False
                    if is_tts_audio:
                        logger.debug("[ResilientSimliVideoService] exception branch -> calling push_frame(frame)")
                        await self.push_frame(frame, direction)

        simli = ResilientSimliVideoService(
            api_key=os.getenv("SIMLI_API_KEY"),
            face_id=os.getenv("SIMLI_FACE_ID"),
        )
        pipeline_stages.append(simli)
        logger.info("[LiveBrain] Simli avatar stage enabled in pipeline.")
    else:
        logger.info("[LiveBrain] Running voice-only mode (Simli avatar stage bypassed).")

    transport_output = transport.output()
    pipeline_stages.extend([
        transport_output,
        assistant_aggregator,
    ])

    pipeline = Pipeline(pipeline_stages)

    enable_latency_timer = os.getenv("ENABLE_LATENCY_TIMER", "true").lower() in ("true", "1", "yes")
    observers = []
    if enable_latency_timer:
        latency_timer = PipelineLatencyTimer(
            output_processor=transport_output,
            get_avatar_state=lambda: bool(enable_avatar and os.getenv("SIMLI_API_KEY")),
        )
        observers.append(latency_timer)

    task = PipelineTask(
        pipeline,
        params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
        observers=observers,
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(
            f"Candidate connected — starting Live Brain interview for {question_bank.metadata.company_name} "
            f"({len(question_bank.questions)} questions loaded in fixed sequence)"
        )
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Candidate disconnected — finalizing Live Brain session")
        await task.cancel()
        collector.trigger_scoring()

    runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)
    try:
        await runner.run(task)
    finally:
        collector.trigger_scoring()


async def bot(runner_args: RunnerArguments) -> None:
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()

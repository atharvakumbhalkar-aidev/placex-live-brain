"""
Live Brain - Data Models & Schema Contract.
Defines Pydantic schemas for questions, rubrics, metadata, and QuestionBank.
"""

from __future__ import annotations
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class QuestionTier(str, Enum):
    WARMUP = "warmup"
    CORE_TECHNICAL = "core_technical"
    DEEP_ARCHITECTURE = "deep_architecture"
    BEHAVIORAL_TRADEOFF = "behavioral_tradeoff"


class RubricCriterion(BaseModel):
    dimension: str = Field(..., description="Evaluation dimension (e.g. Technical Depth, Tradeoff Analysis)")
    weight: float = Field(..., ge=0.0, le=1.0, description="Relative weight summing to 1.0 across criteria")
    poor_0_3: str = Field(..., description="Anchor criteria for 0-3 score (insufficient)")
    good_4_7: str = Field(..., description="Anchor criteria for 4-7 score (competent)")
    expert_8_10: str = Field(..., description="Anchor criteria for 8-10 score (distinguished)")


class ScoringRubric(BaseModel):
    max_points: int = Field(default=10, description="Max points allocatable for this question")
    criteria: List[RubricCriterion] = Field(..., min_length=1)
    key_phrases_expected: List[str] = Field(default_factory=list, description="Keywords or architectural concepts expected")
    anti_patterns: List[str] = Field(default_factory=list, description="Common misconceptions or flawed approaches to penalize")


class QuestionItem(BaseModel):
    id: str = Field(..., description="Unique question identifier (e.g. q1_warmup)")
    index: int = Field(..., ge=1, description="Strict 1-indexed sequential execution position")
    tier: QuestionTier = Field(..., description="Difficulty/Focus tier")
    topic: str = Field(..., description="Subject or problem domain")
    primary_prompt: str = Field(..., description="Natural spoken interviewer prompt")
    context_background: str = Field(..., description="Why this question is relevant to the target company & candidate")
    allowed_micro_probes: List[str] = Field(
        default_factory=list,
        description="Optional clarifying probes (max 1-2) the interviewer can use to clarify answers"
    )
    scoring_rubric: ScoringRubric


class PrepBrainMetadata(BaseModel):
    company_name: str
    company_domain: str
    role_title: str
    target_level: str = Field(default="L5 / Senior")
    generated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    tech_stack_detected: List[str] = Field(default_factory=list)
    domain_focus: List[str] = Field(default_factory=list)


class QuestionBank(BaseModel):
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    metadata: PrepBrainMetadata
    questions: List[QuestionItem] = Field(..., min_length=1)

    def validate_strict_order(self) -> bool:
        """Ensure questions are ordered 1..N sequentially without gaps or reordering."""
        for expected_idx, q in enumerate(self.questions, start=1):
            if q.index != expected_idx:
                return False
        return True

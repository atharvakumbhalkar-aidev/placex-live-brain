"""
Live Brain — Latency Summarizer
Parses .tmp/latency.jsonl and renders markdown percentile tables for turn latency.
Note: The avatar stage (Stage 4) is not reported in this summary.
"""

import json
import math
import sys
from pathlib import Path
from typing import List, Dict, Any, Union


def calculate_percentiles(values: List[float]) -> Dict[str, float]:
    """Calculates p50, p95, min, max for a list of floats."""
    if not values:
        return {"p50": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0, "count": 0}
    sorted_vals = sorted(values)
    n = len(sorted_vals)

    def percentile(p: float) -> float:
        if n == 1:
            return sorted_vals[0]
        k = (n - 1) * p
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return sorted_vals[int(k)]
        return sorted_vals[int(f)] * (c - k) + sorted_vals[int(c)] * (k - f)

    return {
        "p50": round(percentile(0.50), 2),
        "p95": round(percentile(0.95), 2),
        "min": round(sorted_vals[0], 2),
        "max": round(sorted_vals[-1], 2),
        "count": n,
    }


def parse_latency_file(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Reads a jsonl latency file and returns a list of record dicts."""
    p = Path(path)
    if not p.exists():
        return []
    records = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if line_str:
                try:
                    records.append(json.loads(line_str))
                except json.JSONDecodeError:
                    continue
    return records


def generate_markdown_summary(records: List[Dict[str, Any]]) -> str:
    """Renders markdown summary table handling null/None gracefully (Stage 4 omitted)."""
    stages = [
        ("stage_1_speech_end_to_stt_ms", "Stage 1", "speech-end -> STT final transcript"),
        ("stage_2_transcript_to_first_llm_token_ms", "Stage 2", "transcript -> first LLM token"),
        ("stage_3_first_llm_token_to_first_tts_audio_ms", "Stage 3", "first LLM token -> first TTS audio byte"),
        ("stage_5_speech_end_to_first_substantive_audio_ms", "Stage 5", "speech-end -> first substantive bot audio"),
    ]

    lines = [
        "# Turn Latency Performance Summary",
        "",
        f"**Total Turns Evaluated:** {len(records)}",
        "",
        "| Stage | Description | Samples | p50 (ms) | p95 (ms) | Min (ms) | Max (ms) |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: |",
    ]

    for key, label, desc in stages:
        vals = [r[key] for r in records if r.get(key) is not None and isinstance(r[key], (int, float))]
        stats = calculate_percentiles(vals)
        if stats["count"] == 0:
            lines.append(f"| {label} | {desc} | 0 | N/A | N/A | N/A | N/A |")
        else:
            lines.append(
                f"| {label} | {desc} | {stats['count']} | {stats['p50']} | {stats['p95']} | {stats['min']} | {stats['max']} |"
            )

    return "\n".join(lines)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else ".tmp/latency.jsonl"
    records = parse_latency_file(path)
    summary = generate_markdown_summary(records)
    print(summary)


if __name__ == "__main__":
    main()

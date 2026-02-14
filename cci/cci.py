from __future__ import annotations

import json
import math
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

from cci.llm_client import LLMClient
from cci.prompts import PROMPT_TMPL, SYSTEM


def _safe_float_01(value: Any) -> float:
    """Parse a score into [0, 1]. Return NaN if parsing fails."""
    try:
        if isinstance(value, str):
            token = value.strip().replace(",", ".")
            numeric = float(token[:-1]) / 100.0 if token.endswith("%") else float(token)
        else:
            numeric = float(value)
        return max(0.0, min(1.0, numeric))
    except Exception:
        return float("nan")


def _nanmean(values: Iterable[float]) -> float:
    """Compute mean while ignoring NaN values."""
    clean = [
        float(v) for v in values if isinstance(v, (int, float)) and not math.isnan(v)
    ]
    return float(sum(clean) / len(clean)) if clean else float("nan")


def _extract_json(text: str) -> Dict[str, Any]:
    """Extract JSON object from model output using progressively loose strategies."""
    fenced = re.findall(
        r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text, flags=re.IGNORECASE
    )
    for block in fenced:
        try:
            return json.loads(block)
        except Exception:
            pass

    scores_idx = text.find('"scores"')
    scope = text[max(0, scores_idx - 2000) :] if scores_idx != -1 else text

    start, depth = None, 0
    for i, ch in enumerate(scope):
        if ch == "{":
            if start is None:
                start = i
            depth += 1
        elif ch == "}" and start is not None:
            depth -= 1
            if depth == 0:
                candidate = scope[start : i + 1]
                if '"scores"' in candidate:
                    try:
                        return json.loads(candidate)
                    except Exception:
                        pass
                start = None

    try:
        return json.loads(text)
    except Exception:
        return {}


class GeneralityScorer:
    """Run multiple LLM calls and aggregate per-culture generality scores."""

    def __init__(
        self, llm: LLMClient, n_samples: int = 3, per_sample_retry: int = 3
    ) -> None:
        self.llm = llm
        self.n_samples = max(1, n_samples)
        self.per_sample_retry = max(0, per_sample_retry)

    @staticmethod
    def _build_prompt(sentence: str, cultures: List[str]) -> str:
        return PROMPT_TMPL.format(
            sentence=sentence,
            cultures_json=json.dumps(cultures, ensure_ascii=False),
        )

    @staticmethod
    def _validate_scores(obj: Dict[str, Any], cultures: List[str]) -> Dict[str, float]:
        scores = obj.get("scores")
        if not isinstance(scores, dict):
            return {culture: float("nan") for culture in cultures}
        return {
            culture: _safe_float_01(scores.get(culture, float("nan")))
            for culture in cultures
        }

    def compute(self, sentence: str, cultures: List[str]) -> Dict[str, float]:
        """Return averaged per-culture scores, e.g. {'Japan': 0.61, ...}."""
        prompt = self._build_prompt(sentence, cultures)
        acc: Dict[str, List[float]] = {culture: [] for culture in cultures}

        for _ in range(self.n_samples):
            parsed: Optional[Dict[str, float]] = None
            for _ in range(self.per_sample_retry + 1):
                raw = self.llm.chat(SYSTEM, prompt)
                scores = self._validate_scores(_extract_json(raw), cultures)
                if any(not math.isnan(v) for v in scores.values()):
                    parsed = scores
                    break

            if parsed is None:
                parsed = {culture: float("nan") for culture in cultures}

            for culture, value in parsed.items():
                acc[culture].append(value)

        return {culture: _nanmean(values) for culture, values in acc.items()}


def _to_valid_float(value: Any) -> Optional[float]:
    """Convert value to float, returning None for invalid/NaN values."""
    try:
        numeric = float(value)
        return None if math.isnan(numeric) else numeric
    except Exception:
        return None


def compute_cci(
    x: str,
    cultures: List[str],
    target_culture: str,
    calc: GeneralityScorer,
) -> Tuple[Dict[str, float], float]:
    """Compute CCI(target) = p_target - mean(other cultures)."""
    p = calc.compute(sentence=x, cultures=list(cultures))
    p_target = _to_valid_float(p.get(target_culture))
    if p_target is None:
        return p, 0.0

    other_values = [
        _to_valid_float(score)
        for culture, score in p.items()
        if culture != target_culture
    ]
    other_values = [v for v in other_values if v is not None]
    if not other_values:
        return p, 0.0

    return p, float(p_target - (sum(other_values) / len(other_values)))

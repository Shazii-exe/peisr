from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal, Optional, Any, Dict

from gemini_client import generate_json
from prompts import CLASSIFIER_SYSTEM

Route = Literal["SOCIAL", "QA", "TASK", "TECH", "CREATIVE"]


@dataclass(frozen=True)
class IntentResult:
    route: Route
    confidence: float
    reason: str


_SOCIAL_PAT = re.compile(
    r"^(\s*(hi|hey|hello|yo|hii|hiii|sup|what's up|whats up|good\s+morning|good\s+afternoon|good\s+evening|how\s+are\s+you)\b.*)$",
    re.IGNORECASE,
)

_TECH_HINTS = re.compile(
    r"\b(traceback|stack\s*trace|exception|error|bug|debug|python|java|javascript|typescript|sql|select\b|join\b|power\s*bi|dax|m\s*code|streamlit|pip|conda|npm|git|docker|api|http\s*\d\d\d|json|yaml)\b",
    re.IGNORECASE,
)

_TASK_VERBS = re.compile(
    r"\b(draft|write|create|make|build|generate|design|plan|summarize|summarise|compare|review|fix|refactor|implement|convert|translate|explain\s+step\s+by\s+step)\b",
    re.IGNORECASE,
)

_CREATIVE_HINTS = re.compile(
    r"\b(story|poem|rap|lyrics|fantasy|character|plot|brainstorm|ideas|creative)\b",
    re.IGNORECASE,
)

_ALLOWED: set[str] = {"SOCIAL", "QA", "TASK", "TECH", "CREATIVE"}


def _clamp01(x: Any, default: float = 0.5) -> float:
    try:
        v = float(x)
    except Exception:
        return default
    if v != v:  # NaN
        return default
    return max(0.0, min(1.0, v))


def _normalize_route(x: Any) -> Route:
    try:
        s = str(x).strip().upper()
    except Exception:
        return "QA"
    return s if s in _ALLOWED else "QA"  # type: ignore[return-value]


def _rule_route(text: str) -> Optional[IntentResult]:
    t = (text or "").strip()
    if not t:
        return IntentResult(route="SOCIAL", confidence=0.6, reason="empty/blank")

    # Very short social openers
    if len(t) <= 12 and _SOCIAL_PAT.match(t):
        return IntentResult(route="SOCIAL", confidence=0.95, reason="short greeting")

    if _SOCIAL_PAT.match(t) and len(t.split()) <= 8:
        return IntentResult(route="SOCIAL", confidence=0.9, reason="greeting/small-talk")

    if _TECH_HINTS.search(t):
        return IntentResult(route="TECH", confidence=0.85, reason="tech keywords")

    if _CREATIVE_HINTS.search(t):
        return IntentResult(route="CREATIVE", confidence=0.75, reason="creative keywords")

    if _TASK_VERBS.search(t):
        return IntentResult(route="TASK", confidence=0.7, reason="task verb")

    # Questions without task verbs usually map to QA
    if "?" in t or re.search(r"\b(what|why|how|when|where|which|who)\b", t, re.I):
        return IntentResult(route="QA", confidence=0.65, reason="question form")

    return None


def classify_intent(text: str, *, allow_llm: bool = True) -> IntentResult:
    """
    Fast rules first. If rules don't decide and allow_llm=True, ask the LLM.
    This function is hardened: it NEVER throws due to bad/empty JSON.
    """

    rule = _rule_route(text)
    if rule is not None:
        return rule

    if not allow_llm:
        return IntentResult(route="QA", confidence=0.4, reason="default QA (no llm)")

    # LLM fallback (hardened)
    try:
        data = generate_json(
            system=CLASSIFIER_SYSTEM,
            user=f"User message:\n{text}\n",
            temperature=0.0,
        )
    except Exception as e:
        return IntentResult(route="QA", confidence=0.3, reason=f"llm classify failed: {type(e).__name__}")

    if not isinstance(data, dict):
        return IntentResult(route="QA", confidence=0.3, reason="llm returned non-dict")

    route: Route = _normalize_route(data.get("route", "QA"))
    conf = _clamp01(data.get("confidence", 0.5), default=0.5)
    reason = str(data.get("reason", "") or "llm classified")

    return IntentResult(route=route, confidence=conf, reason=reason)


def choose_temperature(route: Route) -> float:
    """Temperature policy (rule-based auto selection)."""
    return {
        "SOCIAL": 0.8,
        "CREATIVE": 1.0,
        "QA": 0.2,
        "TECH": 0.1,
        "TASK": 0.35,
    }[route]


def choose_rewrite_threshold(intent: IntentResult) -> int:
    """Auto rewrite threshold policy (robust)."""

    base_by_route = {
        "SOCIAL": 20,   # effectively unused because SOCIAL bypasses rewrite upstream
        "QA": 13,
        "TASK": 15,
        "TECH": 14,
        "CREATIVE": 11,
    }

    min_threshold = 9
    base = base_by_route[intent.route]

    conf = _clamp01(intent.confidence, default=0.5)
    thr = round(base * conf + min_threshold * (1.0 - conf))

    return int(max(4, min(20, thr)))

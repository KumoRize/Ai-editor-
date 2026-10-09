import pytest

from app.agent.llm_planner import build_context, plan_with_llm
from app.agent.plan import EditPlan, validate_plan
from app.agent.providers.base import LLMError
from app.agent.rule_planner import detect_unsupported, plan_rules

ASSETS = {"a_1": {"asset_id": "a_1", "role": "clip",
                  "meta": {"duration": 60.0, "has_video": True, "has_audio": True, "width": 1920, "height": 1080}}}
ANALYSIS = {"a_1": {"scenes": [{"t": 10.0, "score": 0.5}, {"t": 31.5, "score": 0.6}],
                    "audio": {"window_s": 0.5, "energy_db": [-20.0] * 40 + [-60.0] * 20 + [-10.0] * 60,
                              "silent_ranges": [[20.0, 30.0]]}}}


@pytest.mark.parametrize("prompt,target", [("fast energetic reel", 30), ("calm vlog", 20), ("", 15)])
def test_rule_plan_is_valid_and_hits_target(prompt, target):
    plan = plan_rules(prompt, ASSETS, ANALYSIS, target)
    errors, warnings = validate_plan(plan, ASSETS, target, 90)
    assert errors == [] and warnings == []
    assert plan.duration == pytest.approx(target, abs=0.05)
    ins = [s.src_in for s in plan.segments]
    assert ins == sorted(ins)  # chronological
    for s in plan.segments:      # avoids the silent stretch
        assert not (20.0 < s.src_in < 29.0)


def test_fast_prompt_gives_shorter_shots():
    fast = plan_rules("fast cuts", ASSETS, ANALYSIS, 30)
    slow = plan_rules("slow and calm", ASSETS, ANALYSIS, 30)
    assert len(fast.segments) > len(slow.segments)


def test_short_source_uses_whole_clip():
    plan = plan_rules("", ASSETS, ANALYSIS, 120)
    assert len(plan.segments) == 1 and plan.duration == 60.0


def test_unsupported_detection():
    found = detect_unsupported("Add subtitles, beat-synced cuts, zooms and music like the reference")
    joined = " ".join(found)
    for word in ("subtitles", "beat", "zoom", "music", "reference"):
        assert word in joined
    assert detect_unsupported("just cut the boring parts") == []


def test_validate_plan_catches_bad_segments():
    plan = EditPlan(segments=[{"asset_id": "a_1", "src_in": -1, "src_out": 2},
                              {"asset_id": "a_1", "src_in": 50, "src_out": 70},
                              {"asset_id": "zzz", "src_in": 0, "src_out": 1}])
    errors, _ = validate_plan(plan, ASSETS, 30, 90)
    assert len(errors) == 3


class FakeProvider:
    name = "fake"

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def generate_json(self, *, system, user, schema, schema_name):
        self.calls.append(user)
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


GOOD = {"summary": "ok", "unsupported_requests": [],
        "segments": [{"asset_id": "a_1", "src_in": 0, "src_out": 15, "reason": ""},
                     {"asset_id": "a_1", "src_in": 31.5, "src_out": 46.5, "reason": ""}]}
BAD = {"summary": "bad", "unsupported_requests": [],
       "segments": [{"asset_id": "a_1", "src_in": 50, "src_out": 80, "reason": ""}]}


def test_llm_plan_repair_round():
    p = FakeProvider([BAD, GOOD])
    plan, warnings, calls = plan_with_llm(p, "x", ASSETS, ANALYSIS, 30, 90, None, 3)
    assert calls == 2 and plan.duration == 30
    assert "previous plan was invalid" in p.calls[1]


def test_llm_plan_gives_up_honestly():
    p = FakeProvider([BAD, BAD, BAD])
    with pytest.raises(LLMError):
        plan_with_llm(p, "x", ASSETS, ANALYSIS, 30, 90, None, 3)


def test_llm_transient_error_retried_once():
    p = FakeProvider([LLMError("busy", retryable=True), GOOD])
    plan, _, calls = plan_with_llm(p, "x", ASSETS, ANALYSIS, 30, 90, None, 3)
    assert calls == 2
    p = FakeProvider([LLMError("bad key", retryable=False)])
    with pytest.raises(LLMError):
        plan_with_llm(p, "x", ASSETS, ANALYSIS, 30, 90, None, 3)


def test_context_is_compact():
    long = {"a_1": {**ANALYSIS["a_1"], "audio": {**ANALYSIS["a_1"]["audio"], "energy_db": [-20.0] * 4000}}}
    ctx = build_context("p", {"a_1": {**ASSETS["a_1"], "meta": {**ASSETS["a_1"]["meta"], "duration": 2000}}},
                        long, 30, {"segments": [], "summary": "prev"})
    assert len(ctx) < 6000 and "previous_edit" in ctx

"""The CrewAI tools against a scripted Treza: quote, cap, start once, wait, report."""

import json

import pytest

from crewai_treza import TrezaBalance, TrezaGenerateImage, TrezaGenerateVideo, TrezaGetRun, TrezaRunPipeline, treza_tools
from crewai_treza import client as c

from .fakes import FakeTreza

QUOTE = {"pipelineId": "pipeline_1", "quoteId": "q1", "estimatedChargeUsd": 0.28, "sufficient": True, "model": {"id": "google/veo-3.1-lite"}}
DONE = {"status": "success", "outputs": {"video": "https://media.trezalabs.com/v.mp4"}, "durationMs": 56000}


def build(tool, fake, **settings):
    return tool(client=c.TrezaClient("treza_live_test", session=fake), **settings)


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(c.time, "sleep", lambda s: None)


def test_video_is_quoted_started_once_and_reported():
    fake = FakeTreza(make_video=[(200, QUOTE)], run_pipeline=[(200, {"runId": "r1"})], get_run=[(200, DONE)])
    out = json.loads(build(TrezaGenerateVideo, fake).run(prompt="a fox runs", aspect_ratio="9:16", duration_sec=4))
    assert out["url"] == "https://media.trezalabs.com/v.mp4"
    assert (out["status"], out["price_usd"], out["model"]) == ("success", 0.28, "google/veo-3.1-lite")
    assert fake.asked("make_video") == [{"prompt": "a fox runs", "aspectRatio": "9:16", "durationSec": 4.0}]
    assert fake.asked("run_pipeline") == [{"pipelineId": "pipeline_1", "quoteId": "q1"}]
    assert fake.calls[0][2]["User-Agent"].startswith("crewai-treza/")


def test_over_the_cap_goes_back_to_the_agent_and_starts_nothing():
    fake = FakeTreza(make_video=[(200, QUOTE)])
    said = build(TrezaGenerateVideo, fake, max_cost_usd=0.10).run(prompt="a fox runs")
    assert "over the Max cost of $0.10" in said and "Nothing was spent" in said
    assert fake.asked("run_pipeline") == []


def test_a_refusal_can_raise():
    fake = FakeTreza(make_video=[(200, QUOTE)])
    with pytest.raises(c.TrezaError):
        build(TrezaGenerateVideo, fake, max_cost_usd=0.10, raise_on_refusal=True)._run(prompt="a fox runs")


def test_the_agent_cannot_set_the_cap_or_the_key():
    for tool in (TrezaGenerateImage, TrezaGenerateVideo, TrezaRunPipeline):
        fields = set(tool(api_key="k").args_schema.model_fields)
        assert not fields & {"max_cost_usd", "api_key", "wait_seconds"}, fields


def test_image_passes_references():
    quote = {**QUOTE, "estimatedChargeUsd": 0, "model": "google/gemini-2.5-flash-image"}
    done = {"status": "success", "outputs": {"image": "https://media.trezalabs.com/i.png"}}
    fake = FakeTreza(make_image=[(200, quote)], run_pipeline=[(200, {"runId": "r1"})], get_run=[(200, done)])
    out = json.loads(build(TrezaGenerateImage, fake).run(prompt="a red kite", aspect_ratio="4:5", reference_image_urls=["https://media.trezalabs.com/ref.png"]))
    assert out["url"].endswith("i.png") and out["price_usd"] == 0
    assert fake.asked("make_image") == [{"prompt": "a red kite", "aspectRatio": "4:5", "referenceImageUrls": ["https://media.trezalabs.com/ref.png"]}]


def test_a_slow_render_comes_back_running_and_get_run_finishes_it():
    fake = FakeTreza(make_video=[(200, QUOTE)], run_pipeline=[(200, {"runId": "r1"})], get_run=[(200, {"status": "running"}), (200, DONE)])
    out = json.loads(build(TrezaGenerateVideo, fake, wait_seconds=0).run(prompt="a fox runs"))
    assert out["status"] == "running" and "url" not in out
    later = json.loads(build(TrezaGetRun, fake).run(pipeline_id=out["pipeline_id"], run_id=out["run_id"]))
    assert later["url"] == "https://media.trezalabs.com/v.mp4"


def test_pipeline_passes_inputs():
    quote = {"pipelineId": "pipeline_9", "quoteId": "q9", "estimatedChargeUsd": 1.5, "sufficient": True}
    done = {"status": "success", "outputs": {"final": "https://media.trezalabs.com/f.mp4"}}
    fake = FakeTreza(estimate_run_cost=[(200, quote)], run_pipeline=[(200, {"runId": "r9"})], get_run=[(200, done)])
    out = json.loads(build(TrezaRunPipeline, fake, max_cost_usd=2).run(pipeline_id=" pipeline_9 ", inputs={"prompt_1": "a lighthouse"}))
    assert out["url"] == "https://media.trezalabs.com/f.mp4" and out["price_usd"] == 1.5
    assert fake.asked("run_pipeline") == [{"pipelineId": "pipeline_9", "inputs": {"prompt_1": "a lighthouse"}, "quoteId": "q9"}]


def test_balance_and_a_bad_key():
    fake = FakeTreza(get_credit_balance=[(200, {"balanceUsd": 20.76}), (401, {"error": "Invalid API key."})])
    tool = build(TrezaBalance, fake)
    assert json.loads(tool.run())["balance_usd"] == 20.76
    assert "Check your Treza API key" in tool.run()


def test_key_comes_from_the_environment_and_is_not_printed(monkeypatch):
    monkeypatch.setenv("TREZA_API_KEY", "treza_live_env")
    tool = TrezaGenerateImage()
    assert tool.api_key.get_secret_value() == "treza_live_env"
    assert "treza_live_env" not in repr(tool) and "treza_live_env" not in tool.model_dump_json()
    monkeypatch.delenv("TREZA_API_KEY")
    with pytest.raises(ValueError, match="TREZA_API_KEY"):
        TrezaGenerateImage()


def test_treza_tools_shares_the_key_and_the_cap():
    tools = treza_tools(api_key="treza_live_test", max_cost_usd=1.0)
    assert [t.name for t in tools] == ["treza_generate_image", "treza_generate_video", "treza_run_pipeline", "treza_get_run", "treza_balance"]
    assert all(t.max_cost_usd == 1.0 and t.api_key.get_secret_value() == "treza_live_test" for t in tools)
    assert "treza_run_pipeline" not in [t.name for t in treza_tools(api_key="k", include_pipelines=False)]

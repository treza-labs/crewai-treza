"""Treza for CrewAI: generate AI pictures and videos and run Treza pipelines from a crew."""

from typing import Any, Optional

from crewai_treza.client import TrezaClient, TrezaError
from crewai_treza.tools import TrezaBalance, TrezaGenerateImage, TrezaGenerateVideo, TrezaGetRun, TrezaRunPipeline


def treza_tools(api_key: Optional[str] = None, max_cost_usd: Optional[float] = None, include_pipelines: bool = True, **settings: Any) -> list:
    """Every Treza tool with one key and one price cap, ready for ``Agent(tools=...)``.

    ``max_cost_usd`` is the most any one render may cost; None means no cap, so set one for an agent.
    """
    shared = {"max_cost_usd": max_cost_usd, **settings}
    if api_key:
        shared["api_key"] = api_key
    tools = [TrezaGenerateImage(**shared), TrezaGenerateVideo(**shared)]
    if include_pipelines:
        tools.append(TrezaRunPipeline(**shared))
    return [*tools, TrezaGetRun(**shared), TrezaBalance(**shared)]


__all__ = [
    "TrezaBalance",
    "TrezaClient",
    "TrezaError",
    "TrezaGenerateImage",
    "TrezaGenerateVideo",
    "TrezaGetRun",
    "TrezaRunPipeline",
    "treza_tools",
]

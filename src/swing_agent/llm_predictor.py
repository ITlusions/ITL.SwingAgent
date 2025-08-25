from __future__ import annotations
import os
from typing import Literal, Optional, List
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel

class LlmVote(BaseModel):
    trend_label: Literal["strong_up","up","sideways","down","strong_down"]
    entry_bias: Literal["long","short","none"] = "none"
    entry_window_low: Optional[float] = Field(default=None, gt=0)
    entry_window_high: Optional[float] = Field(default=None, gt=0)
    confidence: float = Field(ge=0, le=1)
    rationale: str

class LlmActionPlan(BaseModel):
    action_plan: str
    risk_notes: str
    scenarios: List[str] = []
    tone: Literal["conservative","balanced","aggressive"] = "balanced"

def _make_agent(model_name: str, system_prompt: str, out_model):
    model = OpenAIModel(model_name)
    return Agent[out_model](model=model, system_prompt=system_prompt)

def llm_extra_prediction(**features) -> LlmVote:
    model_name = os.getenv("SWING_LLM_MODEL", "gpt-4o-mini")
    sys = ("Disciplined 1–2 day swing-trading co-pilot. Return STRICT JSON matching LlmVote.")
    agent = _make_agent(model_name, sys, LlmVote)
    res = agent.run(user_message="Evaluate immediate trend, entry bias and a tight entry window if R/R is favorable.", input=features)
    return res.data

def llm_build_action_plan(*, signal_json: dict, style: str = "balanced") -> LlmActionPlan:
    model_name = os.getenv("SWING_LLM_MODEL", "gpt-4o-mini")
    sys = ("Execution coach for 1–2 day swing trades. "
           "Write a precise, checklist-style plan using trend, fib, R, priors, holds; include invalidations and 2–4 scenarios. "
           "Never invent prices.")
    agent = _make_agent(model_name, sys, LlmActionPlan)
    res = agent.run(user_message="Generate short plan + risks + scenarios. Use expected_win_hold_bars for patience if present.", input={"style": style, "signal": signal_json})
    return res.data

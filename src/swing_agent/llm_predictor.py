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
    """Create a Pydantic AI agent with specified model and output schema.
    
    Factory function for creating LLM agents with consistent configuration
    for different types of trading analysis (voting, action planning).
    
    Args:
        model_name: OpenAI model name (e.g., "gpt-4o-mini", "gpt-4").
        system_prompt: System prompt defining the agent's role and constraints.
        out_model: Pydantic model class for structured output validation.
        
    Returns:
        Agent: Configured Pydantic AI agent ready for inference.
        
    Example:
        >>> agent = _make_agent("gpt-4o-mini", "Trading analyst", LlmVote)
        >>> result = agent.run(user_message="Analyze trend", input=features)
        >>> vote = result.data  # Validated LlmVote object
        
    Note:
        Uses OpenAI models exclusively. Ensure OPENAI_API_KEY is set in environment.
    """
    model = OpenAIModel(model_name)
    return Agent[out_model](model=model, system_prompt=system_prompt)

def llm_extra_prediction(**features) -> LlmVote:
    """Generate LLM-based market analysis and entry bias prediction.
    
    Uses OpenAI models to analyze current market features and provide
    structured trading insights including trend assessment, entry bias,
    confidence scoring, and fundamental rationale.
    
    Args:
        **features: Market feature dictionary containing:
            - trend_label: Current trend classification
            - rsi_14: RSI indicator value
            - price_above_ema: Boolean price vs EMA position
            - fib_position: Position within Fibonacci levels
            - vol_regime: Volatility regime ("L", "M", "H")
            - Additional contextual features
            
    Returns:
        LlmVote: Structured prediction containing:
            - trend_label: LLM's trend assessment
            - entry_bias: Directional bias ("long", "short", "none")
            - entry_window_low/high: Suggested entry price range
            - confidence: Prediction confidence [0, 1]
            - rationale: Text explanation of the analysis
            
    Example:
        >>> features = {
        ...     "trend_label": "up",
        ...     "rsi_14": 65.2,
        ...     "price_above_ema": True,
        ...     "fib_position": 0.62,
        ...     "vol_regime": "M"
        ... }
        >>> vote = llm_extra_prediction(**features)
        >>> print(f"Bias: {vote.entry_bias}")
        >>> print(f"Confidence: {vote.confidence:.1%}")
        >>> print(f"Rationale: {vote.rationale}")
        
    Raises:
        Exception: On API errors, rate limits, or invalid model responses.
        
    Note:
        Model name controlled by SWING_LLM_MODEL environment variable.
        Defaults to "gpt-4o-mini" for cost efficiency.
    """
    model_name = os.getenv("SWING_LLM_MODEL", "gpt-4o-mini")
    sys = ("Disciplined 1–2 day swing-trading co-pilot. Return STRICT JSON matching LlmVote.")
    agent = _make_agent(model_name, sys, LlmVote)
    res = agent.run(user_message="Evaluate immediate trend, entry bias and a tight entry window if R/R is favorable.", input=features)
    return res.data

def llm_build_action_plan(*, signal_json: dict, style: str = "balanced") -> LlmActionPlan:
    """Generate detailed execution action plan for a trading signal.
    
    Creates comprehensive, checklist-style trading plans with risk management,
    scenario analysis, and execution guidance. Never invents prices - uses
    only signal data provided.
    
    Args:
        signal_json: Complete signal dictionary containing:
            - entry_plan: Entry price, stops, targets, R-multiple
            - trend_state: Market trend analysis
            - ml_expectations: Vector-based outcome predictions  
            - enrichments: Additional market context
            - llm_insights: Previous LLM analysis
        style: Plan style ("conservative", "balanced", "aggressive").
            
    Returns:
        LlmActionPlan: Structured action plan containing:
            - action_plan: Detailed execution checklist
            - risk_notes: Risk management considerations
            - scenarios: List of potential outcome scenarios
            - tone: Plan style/approach used
            
    Example:
        >>> signal = {
        ...     "entry_plan": {"side": "LONG", "entry_price": 150.0},
        ...     "trend_state": {"label": "up", "rsi_14": 65},
        ...     "ml_expectations": {"win_rate": 0.68}
        ... }
        >>> plan = llm_build_action_plan(
        ...     signal_json=signal,
        ...     style="conservative"
        ... )
        >>> print(plan.action_plan)
        >>> print("Risk considerations:", plan.risk_notes)
        >>> for i, scenario in enumerate(plan.scenarios, 1):
        ...     print(f"Scenario {i}: {scenario}")
        
    Note:
        Uses expected_win_hold_bars from ml_expectations for patience guidance.
        Plan style affects risk tolerance and position sizing recommendations.
    """
    model_name = os.getenv("SWING_LLM_MODEL", "gpt-4o-mini")
    sys = ("Execution coach for 1–2 day swing trades. "
           "Write a precise, checklist-style plan using trend, fib, R, priors, holds; include invalidations and 2–4 scenarios. "
           "Never invent prices.")
    agent = _make_agent(model_name, sys, LlmActionPlan)
    res = agent.run(user_message="Generate short plan + risks + scenarios. Use expected_win_hold_bars for patience if present.", input={"style": style, "signal": signal_json})
    return res.data

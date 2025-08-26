# ADR-003: LLM Integration Strategy

## Status

Accepted

## Context

SwingAgent integrates Large Language Models for:
- Market condition assessment and trend validation
- Entry bias determination and confidence scoring
- Action plan generation with risk scenarios
- Qualitative analysis to complement quantitative indicators

Key challenges:
- Cost control with commercial APIs
- Reliability and error handling
- Structured output validation
- Domain-specific prompt engineering

## Decision

We will use a structured LLM integration approach:

1. **OpenAI Models Only**: Focus on GPT-4 family for consistency
2. **Pydantic Validation**: All LLM outputs validated against structured schemas
3. **Dual-Purpose Usage**: 
   - **Voting**: Quick market assessment with confidence scoring
   - **Planning**: Detailed execution plans with scenario analysis
4. **Graceful Degradation**: System continues functioning if LLM unavailable
5. **Cost Optimization**: Default to cheaper models (gpt-4o-mini) with configuration override

## Integration Architecture

```python
# Structured output models
class LlmVote(BaseModel):
    trend_label: Literal["strong_up","up","sideways","down","strong_down"]
    entry_bias: Literal["long","short","none"] 
    confidence: float = Field(ge=0, le=1)
    rationale: str

class LlmActionPlan(BaseModel):
    action_plan: str
    risk_notes: str  
    scenarios: List[str]
    tone: Literal["conservative","balanced","aggressive"]
```

## Prompt Engineering Strategy

### Market Analysis Prompts
- **System Role**: "Disciplined 1–2 day swing-trading co-pilot"
- **Constraints**: Return only structured JSON, no price invention
- **Context**: Provide technical indicators, market features, ML priors

### Action Plan Prompts  
- **System Role**: "Execution coach for swing trades"
- **Output Style**: Checklist-based plans with invalidation levels
- **Risk Focus**: Include 2-4 scenarios with specific risk considerations

## Error Handling

```python
def safe_llm_prediction(**features) -> Optional[LlmVote]:
    try:
        return llm_extra_prediction(**features)
    except openai.RateLimitError:
        logging.warning("LLM rate limit hit")
        return None  # Graceful degradation
    except Exception as e:
        logging.error(f"LLM error: {e}")
        return None
```

## Configuration

```bash
# Environment variables
OPENAI_API_KEY=sk-...
SWING_LLM_MODEL=gpt-4o-mini  # Cost-effective default
```

## Consequences

### Positive

- **Structured Reliability**: Pydantic validation ensures consistent output format
- **Cost Control**: Cheaper models by default with selective upgrade capability
- **Domain Expertise**: Specialized prompts for trading context
- **Graceful Degradation**: System works without LLM for pure technical analysis
- **Rich Context**: LLM can synthesize complex market conditions humans might miss

### Negative

- **External Dependency**: Reliance on OpenAI API availability and pricing
- **Latency**: API calls add 1-3 seconds to signal generation
- **Cost Scaling**: Costs increase linearly with signal volume
- **Prompt Drift**: Model updates may change behavior over time

## Usage Guidelines

### When to Use LLM Voting
- Market conditions are ambiguous (sideways trends, mixed signals)
- Technical indicators give conflicting signals  
- High-conviction setups need validation

### When to Use Action Plans
- Live trading execution
- Complex multi-scenario setups
- Client reporting and documentation

## Monitoring

- API response times and error rates
- Token usage and cost tracking
- Output quality validation against backtests
- Prompt effectiveness measurement

## Future Enhancements

- Local model deployment for cost reduction
- Multi-model ensemble voting
- Fine-tuning on historical SwingAgent data
- Real-time market news integration
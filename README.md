# Model Evaluation with Gemini-2.0-Flash

## Challenges we found

1. **OpenAI Dependency**: deepeval uses OpenAI's API for evaluation metrics even when working with other models like Gemini for generation. This requires an OpenAI API key.

2. **API Key Permission Issues**: When using Gemini API for evaluation, we encountered "insufficient authentication scopes" errors.

3. **Complex Metric Requirements**: Some deepeval metrics like TaskCompletionMetric and HallucinationMetric require specific parameters (like 'tools_called' or 'context').

## Solutions

### 1. Manual Testing Approach (Recommended)

Create custom tests with simple assertions to verify that the model responses contain expected information:

```python
# test_no_deepeval.py
import pytest
import re
from generate_response import generate_response

def test_refund_policy():
    input_query = "What if these shoes don't fit?"
    model_output = generate_response(input_query)
    
    # Check that the response mentions a 30-day return policy
    assert re.search(r'30[ -]day', model_output.lower())
    
    # Check that it mentions returns or refunds
    assert any(word in model_output.lower() for word in ["return", "refund", "exchange"])
```

### 2. Use deepeval with OpenAI

If you want to use deepeval's metrics, you need to provide a valid OpenAI API key:

```python
import os
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"
```

### 3. Create a Custom Evaluator

We attempted to create a custom evaluator using Gemini, but encountered authentication issues:

```python
# custom_metrics.py
class GeminiEvaluator:
    def __init__(self, name, criteria, threshold=0.7):
        self.name = name
        self.criteria = criteria
        self.threshold = threshold
        # ...other initialization
    
    def measure(self, test_case):
        # Use Gemini API to evaluate responses
        # ...implementation
```

## Recommendations

1. For testing Gemini model responses without deepeval, use simple assertion-based tests as in `test_no_deepeval.py`.

2. If you need more sophisticated evaluation, consider:
   - Getting a valid OpenAI API key to use with deepeval's metrics
   - Setting up proper authentication for Google API to create custom evaluators

3. Focus on testing specific aspects of the responses rather than general quality metrics.
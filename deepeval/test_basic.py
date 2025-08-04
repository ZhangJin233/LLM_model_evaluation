import pytest
from deepeval import assert_test
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase
from deepeval.dataset import EvaluationDataset
from deepeval.metrics import AnswerRelevancyMetric
import os

from deepeval.models import GeminiModel
from deepeval.metrics import AnswerRelevancyMetric

model = GeminiModel(
    model_name="gemini-2.0-flash",
    api_key="YOUR_GOOGLE_API_KEY",  # Replace with your actual API key
    temperature=0,
)

answer_relevancy = AnswerRelevancyMetric(model=model)

first_test_case = LLMTestCase(
    input="What if these shoes don't fit?",
    actual_output="We offer a 30-day full refund at no extra costs.",
    context=["All customers are eligible for a 30 day full refund at no extra costs."],
)

dataset = EvaluationDataset(test_cases=[first_test_case])


@pytest.mark.parametrize(
    "test_case",
    dataset,
)
@pytest.mark.asyncio
def test_chat_app(test_case: LLMTestCase):
    answer_relevancy_metric = AnswerRelevancyMetric(
        threshold=0.7,
        model=model,
        # include_reason=True,
    )

    test_case = LLMTestCase(
        input="What if these shoes don't fit?",
        # Replace this with the actual output from your LLM application
        actual_output="You have 30 days to get a full refund at no extra cost.",
        expected_output="We offer a 30-day full refund at no extra costs.",
        retrieval_context=[
            "All customers are eligible for a 30 day full refund at no extra costs."
        ],
    )
    assert_test(test_case, [answer_relevancy_metric])

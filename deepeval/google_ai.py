from deepeval.models import GeminiModel
from deepeval.metrics import AnswerRelevancyMetric


class CustomGemini:
    def __init__(self):
        self.model = GeminiModel(
            model_name="gemini-2.0-flash",
            api_key="YOUR_GOOGLE_API_KEY",  # Replace with your actual API key
            temperature=0,
        )

    def get_model(self):
        return self.model

    def generate_response(self, prompt: str) -> str:
        response = self.model.generate(prompt)
        return response.text

    def answer_relevancy(
        self, input_text: str, actual_output: str, context: list
    ) -> float:
        answer_relevancy_metric = AnswerRelevancyMetric(model=self.model)
        test_case = {
            "input": input_text,
            "actual_output": actual_output,
            "context": context,
        }
        score = answer_relevancy_metric.score(test_case)
        return score

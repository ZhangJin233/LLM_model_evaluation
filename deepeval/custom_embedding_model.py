from typing import List, Optional
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from deepeval.models import DeepEvalBaseEmbeddingModel


class CustomEmbeddingModel(DeepEvalBaseEmbeddingModel):
    def __init__(self):
        pass

    def load_model(self):
        return GoogleGenerativeAIEmbeddings(
            GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY",
            model_name="models/gemini-2.0-flash",
        )

    def embed_text(self, text: str) -> List[float]:
        embedding_model = self.load_model()
        return embedding_model.embed_query(text)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        embedding_model = self.load_model()
        return embedding_model.embed_documents(texts)

    async def a_embed_text(self, text: str) -> List[float]:
        embedding_model = self.load_model()
        return await embedding_model.aembed_query(text)

    async def a_embed_texts(self, texts: List[str]) -> List[List[float]]:
        embedding_model = self.load_model()
        return await embedding_model.aembed_documents(texts)

    def get_model_name(self):
        "Custom Gemini Embedding Model"

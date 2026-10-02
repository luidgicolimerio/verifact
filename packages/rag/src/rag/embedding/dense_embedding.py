"""Generic Dense Embedding wrapper for OpenAI-compatible embedding APIs.

Suitable for models that return only dense embeddings (e.g., thenlper/gte-large,
BAAI/bge-base-en-v1.5, nomic-ai/modernbert-embed-base, etc.) served via Infinity
or any OpenAI-compatible embedding endpoint.
"""

from typing import Any

import httpx
from llama_index.core.bridge.pydantic import Field
from llama_index.core.schema import MetadataMode

from rag.embedding.base import BaseEmbedding
from rag.schema.base_node import EmbeddingItem
from rag.schema.node_schema import TextNode


class DenseEmbedding(BaseEmbedding):
    """Llama-Index wrapper for any OpenAI-compatible dense embedding API."""

    model_name: str = Field(default="BAAI/bge-base-en-v1.5", description="Model name.")
    embed_batch_size: int = Field(default=50, description="Embedding batch size.", gt=0)
    api_base: str = Field(default="http://embed.localhost", description="API base URL.")
    dense_name: str = Field(default="dense", description="Vector name key for dense embeddings.")
    default_vector_name: str = Field(default="dense", description="Default vector name.")
    timeout: float | None = Field(default=600.0, description="Timeout for API calls.")
    num_retries: int = Field(default=5, description="Number of retries for API calls.")
    metadata_mode: MetadataMode | str = Field(
        default=MetadataMode.NONE,
        description="Metadata mode for node formatting prior to embedding.",
    )
    num_workers: int | None = Field(default=4, description="Number of async embedding workers.")

    @classmethod
    def class_name(cls) -> str:
        return "DenseEmbedding"

    def _embed(self, sentences: list[str], num_retries: int | None = None) -> list[dict]:
        if num_retries is None:
            num_retries = self.num_retries
        if num_retries == 0:
            raise ConnectionError(f"Failed to generate embeddings after {self.num_retries} retries.")

        try:
            response = httpx.request(
                method="POST",
                url=self.api_base.rstrip("/") + "/v1/embeddings",
                json={"input": sentences, "model": self.model_name},
                timeout=self.timeout,
            )
        except httpx.TransportError:
            return self._embed(sentences, num_retries=num_retries - 1)

        if response.is_success:
            payload_list = response.json()["data"]
            return [
                {self.dense_name: EmbeddingItem(name=self.dense_name, embedding=x["embedding"], kind="dense")}
                for x in payload_list
            ]
        return self._embed(sentences, num_retries=num_retries - 1)

    def _get_query_embedding(self, query: str):
        return self._embed([query])[0]

    async def _aget_query_embedding(self, query: str):
        return self._get_query_embedding(query)

    def _get_text_embedding(self, text: str):
        return self._embed([text])[0]

    async def _aget_text_embedding(self, text: str):
        return self._get_text_embedding(text)

    def _get_text_embeddings(self, texts: list[str]):
        return self._embed(texts)

    def get_text_embedding(self, text: str):
        return self._embed([text])[0]

    def get_text_embeddings(self, texts: list[str]):
        return self._embed(texts)

    def __call__(self, nodes: list[TextNode], **kwargs: Any) -> list[TextNode]:
        metadata_mode = kwargs.pop("metadata_mode", self.metadata_mode)
        embeddings = self.get_text_embedding_batch(
            [node.get_content(metadata_mode=metadata_mode) for node in nodes], **kwargs
        )
        return self._associate_nodes_and_embeddings(nodes, embeddings)

    async def acall(self, nodes: list[TextNode], **kwargs: Any) -> list[TextNode]:
        metadata_mode = kwargs.pop("metadata_mode", self.metadata_mode)
        embeddings = await self.aget_text_embedding_batch(
            [node.get_content(metadata_mode=metadata_mode) for node in nodes], **kwargs
        )
        return self._associate_nodes_and_embeddings(nodes, embeddings)

    def _associate_nodes_and_embeddings(self, nodes, embeddings):
        import warnings
        for node, embedding in zip(nodes, embeddings, strict=False):
            try:
                node.set_embeddings(embeddings=embedding, default=self.default_vector_name)
            except Exception:
                warnings.warn("No method `set_embedding`, falling back to `embedding` attribute.", stacklevel=1)
                node.embedding = embedding
        return nodes

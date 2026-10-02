"""Azure OpenAI LLM implementation for VeriFact."""

import os
from typing import Any

from llama_index.core.llms import LLMMetadata
from llama_index.llms.azure_openai import AzureOpenAI as _AzureOpenAI
from pydantic import Field


class AzureOpenAILLM(_AzureOpenAI):
    """Thin wrapper around LlamaIndex AzureOpenAI that reads credentials from env vars.
    Bypasses llama-index model name validation so any model name (e.g. DeepSeek) works.
    """

    context_window: int = Field(default=128000, description="Context window size")

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        deployment_name: str = "gpt-4o-mini",
        azure_endpoint: str | None = None,
        api_key: str | None = None,
        api_version: str = "2024-12-01-preview",
        context_window: int | None = None,
        max_completion_tokens: int | None = None,
        **kwargs: Any,
    ) -> None:
        # Remove max_tokens from kwargs if present to avoid conflict
        kwargs.pop("max_tokens", None)

        super().__init__(
            model=model,
            engine=deployment_name,
            azure_endpoint=azure_endpoint or os.environ["AZURE_URL"],
            api_key=api_key or os.environ["AZURE_API_KEY"],
            api_version=api_version,
            max_tokens=max_completion_tokens,
            **kwargs,
        )
        if context_window is not None:
            self.context_window = context_window

    def _get_model_name(self) -> str:
        """Override to bypass llama-index's hardcoded OpenAI model name validation."""
        return self.model

    @property
    def metadata(self) -> LLMMetadata:
        """Override to bypass openai_modelname_to_contextsize validation for non-OpenAI models."""
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.max_tokens or -1,
            is_chat_model=True,
            is_function_calling_model=True,
            model_name=self.model,
        )

    @classmethod
    def class_name(cls) -> str:
        return "AzureOpenAILLM"

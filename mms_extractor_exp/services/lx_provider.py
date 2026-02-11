"""
LangChainProvider — Adapter bridging LLMFactory to langextract's BaseLanguageModel.

Translates langextract's `infer(batch_prompts) → Iterator[Sequence[ScoredOutput]]`
interface to LangChain's `.invoke(prompt) → AIMessage.content`.

Registered via @router.register() so that model IDs like 'ax', 'gpt', 'cld', 'gem',
'gen', 'opus' (and their full names like 'skt/ax4', 'azure/openai/...') route to this
provider automatically.
"""

import logging
from collections.abc import Iterator, Sequence

from langextract.core.base_model import BaseLanguageModel
from langextract.core import types as core_types
from langextract.providers import router

from utils.llm_factory import LLMFactory

logger = logging.getLogger(__name__)

# Regex patterns matching all model IDs that LLMFactory supports
_PATTERNS = [
    r"^(ax|gpt|gen|cld|gem|opus)$",  # short aliases
    r"^skt/",                          # SKT gateway models
    r"^azure/",                        # Azure gateway models
    r"^gcp/",                          # GCP gateway models
    r"^amazon/",                       # Amazon gateway models
    r"^claude-",                       # Direct Anthropic models
    r"^gemma",                         # Gemma models
]


@router.register(*_PATTERNS, priority=50)
class LangChainProvider(BaseLanguageModel):
    """langextract provider backed by LLMFactory (LangChain ChatOpenAI/ChatAnthropic).

    Usage via langextract:
        lx.extract(text, model_id="ax", ...)      # uses SKT ax4
        lx.extract(text, model_id="opus", ...)     # uses Claude Opus direct

    Usage standalone:
        provider = LangChainProvider(model_id="ax")
        for batch_outputs in provider.infer(["Hello"]):
            print(batch_outputs[0].output)
    """

    def __init__(self, model_id: str = "ax", **kwargs):
        super().__init__(**kwargs)
        self.model_id = model_id
        self._factory = LLMFactory()
        self._llm = self._factory.create_model(model_id)
        logger.info(f"LangChainProvider initialized: model_id={model_id}")

    def infer(
        self,
        batch_prompts: Sequence[str],
        **kwargs,
    ) -> Iterator[Sequence[core_types.ScoredOutput]]:
        """Run inference on a batch of prompts.

        Args:
            batch_prompts: List of prompt strings.
            **kwargs: Additional arguments (ignored, LLM config is set at init time).

        Yields:
            For each prompt, a list containing one ScoredOutput with score=1.0.
        """
        for prompt in batch_prompts:
            try:
                response = self._llm.invoke(prompt)
                text = response.content if hasattr(response, "content") else str(response)
                yield [core_types.ScoredOutput(score=1.0, output=text)]
            except Exception as e:
                logger.error(f"LangChainProvider.infer error: {e}")
                yield [core_types.ScoredOutput(score=0.0, output="")]

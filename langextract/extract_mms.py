"""
High-level extraction wrapper for Korean MMS entity extraction.

Wraps `langextract.extract()` with MMS-specific defaults: Korean prompt description,
few-shot examples, appropriate buffer size for MMS messages, and LangChainProvider
integration.
"""

import logging
from typing import Any

from langextract.core.data import AnnotatedDocument
from langextract.extraction import extract as lx_extract

# Import provider module to trigger @router.register() side-effect
from langextract import provider as _provider  # noqa: F401
from langextract.examples import build_mms_examples
from langextract.schemas import get_class_description_text

logger = logging.getLogger(__name__)

# Prompt description for MMS entity extraction
MMS_PROMPT_DESCRIPTION = f"""\
SK텔레콤 MMS 광고 메시지에서 핵심 엔티티(Core Offering Entities)를 추출하라.
핵심 오퍼링이란 광고가 고객에게 제안하는 구체적인 상품, 서비스, 매장, 이벤트, 채널, 목적을 의미한다.

## Entity Types
{get_class_description_text()}

## Extraction Rules
1. Zero-Translation: 원문에 등장하는 그대로 추출하라. 번역하지 말라.
2. Specificity: 구체적인 고유명사만 추출하라. 포괄적 카테고리명은 제외한다.
3. Store: 대리점명 + 지점명을 하나의 엔티티로 추출한다.
4. Voucher: 제휴 브랜드 + 혜택 설명을 결합하여 추출한다.
5. Strict Exclusions: 할인 금액 단독, URL/연락처, 네비게이션 라벨, 일반 용어는 제외한다.
"""


def extract_mms_entities(
    message: str,
    model_id: str = "ax",
    max_char_buffer: int = 5000,
    extraction_passes: int = 1,
    temperature: float | None = None,
    show_progress: bool = False,
    **kwargs: Any,
) -> AnnotatedDocument:
    """Extract structured entities from a Korean MMS advertisement message.

    Args:
        message: Korean MMS advertisement text.
        model_id: LLM model alias ('ax', 'gpt', 'gen', 'cld', 'gem', 'opus')
            or full model name (e.g. 'skt/ax4').
        max_char_buffer: Maximum characters per inference chunk. Korean MMS messages
            are typically <2000 chars, so 5000 avoids unnecessary chunking.
        extraction_passes: Number of extraction passes for improved recall.
        temperature: Sampling temperature (None uses model default, 0.0 for deterministic).
        show_progress: Whether to show progress bar.
        **kwargs: Additional arguments passed to langextract.extract().

    Returns:
        AnnotatedDocument with extracted entities.

    Example:
        >>> from langextract import extract_mms_entities
        >>> result = extract_mms_entities(
        ...     "T Day 혜택 안내! 5GX 프라임 요금제 가입 시 올리브영 기프트카드 증정",
        ...     model_id="ax"
        ... )
        >>> for ext in result.extractions:
        ...     print(f"{ext.extraction_class}: {ext.extraction_text}")
    """
    examples = build_mms_examples()

    result = lx_extract(
        text_or_documents=message,
        prompt_description=MMS_PROMPT_DESCRIPTION,
        examples=examples,
        model_id=model_id,
        max_char_buffer=max_char_buffer,
        extraction_passes=extraction_passes,
        temperature=temperature,
        show_progress=show_progress,
        fetch_urls=False,
        **kwargs,
    )

    return result

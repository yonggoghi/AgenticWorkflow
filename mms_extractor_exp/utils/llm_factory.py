"""
LLM Factory - LLM 모델 초기화 전담 클래스
============================================

📋 개요
-------
순환 의존성을 제거하고 LLM 초기화 로직을 중앙화합니다.
다양한 LLM 모델을 일관된 인터페이스로 생성하고 관리합니다.

🔗 의존성
---------
**사용하는 모듈:**
- `langchain_openai.ChatOpenAI`: LLM 모델 인터페이스
- `config.settings`: API 및 모델 설정 (API_CONFIG, MODEL_CONFIG)
- `os`: 환경변수 접근

**사용되는 곳:**
- `core.mms_extractor`: MMSExtractor 초기화 시 LLM 생성
- `services.result_builder`: ResultBuilder에서 동적 LLM 생성
- `services.entity_recognizer`: 멀티모델 엔티티 추출

🏗️ 주요 기능
------------
- **단일/복수 LLM 모델 생성**: 하나 또는 여러 모델을 동시에 초기화
- **모델명 매핑 관리**: 짧은 별칭(ax, gpt 등)을 실제 모델명으로 변환
- **일관된 설정 적용**: temperature, seed, max_tokens 통일

📊 지원 모델 매핑
----------------

| 별칭 | 실제 모델명 | 제공사 | 용도 |
|------|------------|--------|------|
| **ax** | skt/ax4 | SK Telecom | 기본 추출 모델 (빠름) |
| **gpt** | azure/openai/gpt-4o-2024-08-06 | OpenAI | 고품질 추출 |
| **gen** | gcp/gemini-2.5-flash | Google | 빠른 처리 |
| **cld** | amazon/anthropic/claude-sonnet-4 | Anthropic | 복잡한 추론 |
| **gem** | gemma-7b | Google | 경량 모델 |

### 모델 선택 가이드
- **빠른 처리**: ax, gen
- **높은 정확도**: gpt, cld
- **비용 효율**: gem, gen
- **멀티모델 앙상블**: [ax, gpt, gen]

💡 사용 예시
-----------
```python
from utils.llm_factory import LLMFactory

# 1. 기본 사용 (config.settings 자동 로드)
factory = LLMFactory()

# 단일 모델 생성
llm = factory.create_model('ax')
response = llm.invoke("광고 메시지 분석...")

# 2. 복수 모델 생성 (멀티모델 앙상블)
llms = factory.create_models(['ax', 'gpt', 'gen'])
for llm in llms:
    response = llm.invoke("...")

# 3. 커스텀 설정 사용
from config.settings import API_CONFIG, MODEL_CONFIG

factory = LLMFactory(
    api_config=API_CONFIG,
    model_config=MODEL_CONFIG
)
llm = factory.create_model('cld')

# 4. 환경변수만 사용 (config 없이)
factory = LLMFactory(api_config=None, model_config=None)
llm = factory.create_model('ax')  # 환경변수에서 API 키 로드
```

⚙️ 설정 우선순위
---------------
1. **API 키**: `api_config.llm_api_key` → `OPENAI_API_KEY` → `LLM_API_KEY`
2. **API URL**: `api_config.llm_api_url` → `LLM_BASE_URL`
3. **모델 설정**: `model_config` → 기본값 (max_tokens=4000, seed=42)

📝 참고사항
----------
- 모든 모델은 temperature=0.0 (결정론적 출력)
- 실패한 모델은 자동으로 건너뛰고 로그 기록
- 모델명 매핑은 `__init__`에서 설정 가능
- 하위 호환성을 위해 MMSExtractor의 `_initialize_multiple_llm_models` 유지

작성자: MMS 분석팀
최종 수정: 2024-12
버전: 2.1.0
"""

import logging
import os
from typing import List, Optional
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)


class LLMFactory:
    """
    LLM 모델 생성 Factory 클래스
    
    책임:
        - LLM 모델 인스턴스 생성 및 초기화
        - 모델명 별칭 관리 (ax, gpt, gen 등)
        - API 키 및 설정 관리
        - 멀티모델 생성 지원
    
    협력 객체:
        - **ChatOpenAI**: LangChain OpenAI 인터페이스
        - **config.settings**: API 및 모델 설정 제공
    
    Attributes:
        api_config: API 설정 객체 (llm_api_key, llm_api_url)
        model_config: 모델 설정 객체 (max_tokens, seed, 모델명)
        model_mapping (Dict[str, str]): 별칭 → 실제 모델명 매핑
    """
    
    def __init__(self, api_config=None, model_config=None):
        """
        LLMFactory 초기화
        
        Args:
            api_config: API 설정 객체 (선택사항, None이면 config.settings에서 로드)
            model_config: 모델 설정 객체 (선택사항, None이면 config.settings에서 로드)
        """
        # Config 로드
        if api_config is None or model_config is None:
            try:
                from config.settings import API_CONFIG, MODEL_CONFIG
                self.api_config = api_config or API_CONFIG
                self.model_config = model_config or MODEL_CONFIG
            except ImportError:
                logger.warning("설정 파일을 찾을 수 없습니다. 환경변수를 사용합니다.")
                self.api_config = None
                self.model_config = None
        else:
            self.api_config = api_config
            self.model_config = model_config
        
        # 모델명 매핑
        self.model_mapping = {
            "cld": self._get_config_value('anthropic_model', 'amazon/anthropic/claude-sonnet-4-20250514'),
            "ax": self._get_config_value('ax_model', 'skt/ax4'),
            "gpt": self._get_config_value('gpt_model', 'azure/openai/gpt-4o-2024-08-06'),
            "gen": self._get_config_value('gemini_model', 'gcp/gemini-2.5-flash'),
            "gem": self._get_config_value('gemma_model', 'gemma-7b'),
        }
    
    def _get_config_value(self, attr_name: str, default_value: str) -> str:
        """설정 값을 안전하게 가져오기"""
        if self.model_config:
            return getattr(self.model_config, attr_name, default_value)
        return default_value
    
    def create_model(self, model_name: str) -> ChatOpenAI:
        """
        단일 LLM 모델 생성
        
        Args:
            model_name: 모델 이름 (예: 'ax', 'gpt', 'gen', 'cld', 'gem')
        
        Returns:
            ChatOpenAI: 초기화된 LLM 모델 인스턴스
        
        Raises:
            Exception: 모델 생성 실패 시
        """
        actual_model_name = self.model_mapping.get(model_name, model_name)
        
        # API 키 및 URL 가져오기
        if self.api_config:
            api_key = getattr(self.api_config, 'llm_api_key', os.getenv('OPENAI_API_KEY'))
            api_base = getattr(self.api_config, 'llm_api_url', None)
        else:
            api_key = os.getenv('OPENAI_API_KEY') or os.getenv('LLM_API_KEY')
            api_base = os.getenv('LLM_BASE_URL')
        
        # 모델 설정 가져오기
        if self.model_config:
            max_tokens = getattr(self.model_config, 'llm_max_tokens', 4000)
            seed = getattr(self.model_config, 'llm_seed', 42)
        else:
            max_tokens = 4000
            seed = 42
        
        model_kwargs = {
            "temperature": 0.0,
            "openai_api_key": api_key,
            "openai_api_base": api_base,
            "model": actual_model_name,
            "max_tokens": max_tokens,
            "seed": seed
        }
        
        llm_model = ChatOpenAI(**model_kwargs)
        logger.info(f"✅ LLM 모델 생성 완료: {model_name} ({actual_model_name})")
        return llm_model
    
    def create_models(self, model_names: List[str]) -> List[ChatOpenAI]:
        """
        복수의 LLM 모델 생성
        
        Args:
            model_names: 모델 이름 리스트 (예: ['ax', 'gpt', 'gen'])
        
        Returns:
            List[ChatOpenAI]: 초기화된 LLM 모델 인스턴스 리스트
            
        Note:
            실패한 모델은 건너뛰고 성공한 모델만 반환합니다.
        """
        llm_models = []
        
        for model_name in model_names:
            try:
                llm_model = self.create_model(model_name)
                llm_models.append(llm_model)
            except Exception as e:
                logger.error(f"❌ LLM 모델 생성 실패: {model_name} - {e}")
                continue
        
        logger.info(f"✅ 총 {len(llm_models)}개 LLM 모델 생성 완료")
        return llm_models

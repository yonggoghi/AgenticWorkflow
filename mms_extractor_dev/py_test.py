from concurrent.futures import ThreadPoolExecutor
import time
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import json
import re
# from pygments import highlight
# from pygments.lexers import JsonLexer
# from pygments.formatters import HtmlFormatter
# from IPython.display import HTML
import pandas as pd
# from langchain.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from openai import OpenAI
from typing import List, Tuple, Union, Dict, Any
import ast
from rapidfuzz import fuzz, process
import re
import json
import glob
import os
from config import settings

pd.set_option('display.max_colwidth', 500)

llm_api_key = settings.API_CONFIG.llm_api_key
llm_api_url = settings.API_CONFIG.llm_api_url
client = OpenAI(
    api_key = llm_api_key,
    base_url = llm_api_url
)
# from langchain.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.schema import AIMessage, HumanMessage, SystemMessage

llm_gem = ChatOpenAI(
        temperature=0,
        openai_api_key=llm_api_key,
        openai_api_base=llm_api_url,
        model=settings.ModelConfig.gemma_model,
        max_tokens=settings.ModelConfig.llm_max_tokens,
        seed=42  
        )

llm_ax = ChatOpenAI(
        temperature=0,
        openai_api_key=llm_api_key,
        openai_api_base=llm_api_url,
        model=settings.ModelConfig.ax_model,
        max_tokens=settings.ModelConfig.llm_max_tokens,
        seed=42  
        )

llm_cld = ChatOpenAI(
        temperature=0,
        openai_api_key=llm_api_key,
        openai_api_base=llm_api_url,
        model=settings.ModelConfig.claude_model,
        max_tokens=settings.ModelConfig.llm_max_tokens,
        seed=42  
        )

llm_gen = ChatOpenAI(
        temperature=0,
        openai_api_key=llm_api_key,
        openai_api_base=llm_api_url,
        model=settings.ModelConfig.gemini_model,
        max_tokens=settings.ModelConfig.llm_max_tokens,
        seed=42  
        )

llm_gpt = ChatOpenAI(
        temperature=0,
        openai_api_key=llm_api_key,
        openai_api_base=llm_api_url,
        model=settings.ModelConfig.gpt_model,
        max_tokens=settings.ModelConfig.llm_max_tokens,
        seed=42  
        )

print(llm_ax.invoke(
"""
Analyze the advertisement to extract **User Action Paths**.
Output two distinct sections:
1. **ENTITY**: A list of independent Root Nodes.
2. **DAG**: A structured graph representing the flow from Root to Benefit.

## Crucial Language Rule
* **DO NOT TRANSLATE:** Extract entities **exactly as they appear** in the source text.
* **Preserve Original Script:** If the text says "아이폰 17", output "아이폰 17" (NOT "iPhone 17"). If it says "T Day", output "T Day".

## Part 1: Root Node Selection Hierarchy (Extract ALL Distinct Roots)
Identify logical starting points based on this priority. If multiple independent offers exist, extract all.

1.  **Physical Store (Highest):** Specific branch names.
    * *Match:* "새샘대리점 역곡점", "백색대리점 수성직영점"
2.  **Core Service (Plans/VAS):** Rate plans, Value-Added Services, Internet/IPTV.
    * *Match:* "5GX 프라임 요금제", "V컬러링", "로밍 baro 요금제"
3.  **Subscription/Event:** Membership signups or specific campaigns.
    * *Match:* "T 우주", "T Day", "0 day", "골드번호 프로모션"
4.  **App/Platform:** Apps requiring action.
    * *Match:* "A.(에이닷)", "PASS 앱", "T world"
5.  **Product (Hardware):** Device launches without a specific store focus.
    * *Match:* "iPhone 17", "갤럭시 Z 플립7"

## Part 2: DAG Construction Rules
Construct a Directed Acyclic Graph (DAG) for each identified Root Node.
* **Format:** `(Node:Action) -[Edge]-> (Node:Action)`
* **Nodes:**
    * **Root:** The entry point identified above (Original Text).
    * **Core:** The product/service being used or bought (Original Text).
    * **Value:** The final reward or benefit (Original Text).
* **Edges:**
    * **Definition:** A verb describing the relationship between two nodes.
    * **Purpose:** Represents the action or transition from one node to the next.
    * **Examples:**
        * `가입` (subscribe), `구매` (purchase), `사용` (use)
        * `획득` (obtain), `제공` (provide), `지급` (grant)
        * `방문` (visit), `다운로드` (download), `신청` (apply)
    * **Guidelines:** Use concise action verbs that clearly describe how the user moves from one step to the next in the flow.
* **Logic:** Represent the shortest path from the Root action to the Final Benefit.

## Strict Exclusions
* Ignore navigational labels ('바로 가기', '링크', 'Shortcut').
* Ignore generic partners ('스타벅스', 'CU') unless they are the main subscription target.

## Output Format
ENTITY: <comma-separated list of all Nodes in original text>
DAG:
<DAG representation line by line in original text>


## message:
message: '[SK텔레콤] 공식인증대리점 혜택 안내드립니다.  (광고)[SKT] 공식인증대리점 혜택 안내__고객님, 안녕하세요._SK텔레콤 공식인증대리점에서 상담받고 다양한 혜택을 누려 보세요.__■ 공식인증대리점 혜택_- T끼리 온가족할인, 선택약정으로 통신 요금 최대 55% 할인_- 갤럭시 폴더블/퀀텀, 아이폰 등 기기 할인 상담__■ T 멤버십 고객 감사제 안내_- 2025년 12월까지 매달 Big 3 제휴사 릴레이 할인(10일 단위)__궁금한 점이 있으면 가까운 T 월드 매장에 방문하거나 전화로 문의해 주세요.__▶ 가까운 매장 찾기: https://tworldfriends.co.kr/h/B11109__■ 문의: SKT 고객센터(1558, 무료)__SKT와 함께해 주셔서 감사합니다.__무료 수신거부 1504',
"""

).content)
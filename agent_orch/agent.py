import os
import logging
import sys
from typing import Dict, Any, List
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from langchain_core.tools import tool
from agent_orch.tools import EntitySearchTool


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure mms_extractor_exp is in path for imports
mms_exp_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "mms_extractor_exp")
sys.path.insert(0, mms_exp_path)

try:
    from config.settings import API_CONFIG, MODEL_CONFIG
except ImportError:
    # Fallback if not found or path issue
    logging.warning("Could not import settings from config. Using defaults.")
    API_CONFIG = None
    MODEL_CONFIG = None

# Load environment variables
load_dotenv("/Users/yongwook/workspace/AgenticWorkflow/mms_extractor_exp/.env")


# Load configuration
# Ensure we use absolute path for the CSV
workspace_root = "/Users/yongwook/workspace/AgenticWorkflow"
default_csv_path = os.path.join(workspace_root, "mms_extractor_exp/data/offer_master_data.csv")

env_csv_path = os.getenv("OFFER_DATA_PATH")
if env_csv_path:
    # Try to resolve relative path
    if not os.path.isabs(env_csv_path):
        # Try relative to workspace root
        candidate_1 = os.path.join(workspace_root, env_csv_path.lstrip("./"))
        # Try relative to mms_extractor_exp
        candidate_2 = os.path.join(workspace_root, "mms_extractor_exp", env_csv_path.lstrip("./"))
        
        if os.path.exists(candidate_1):
            CSV_PATH = candidate_1
        elif os.path.exists(candidate_2):
            CSV_PATH = candidate_2
        else:
            # Fallback to default
            CSV_PATH = default_csv_path
    else:
        CSV_PATH = env_csv_path
else:
    CSV_PATH = default_csv_path

# Initialize the search tool instance
search_tool_instance = EntitySearchTool(CSV_PATH)

from pydantic import BaseModel, Field

class SearchEntityArgs(BaseModel):
    query: str = Field(description="The search query to find the entity. e.g. 'iPhone 15', 'Netflix', '우주패스'")

@tool(args_schema=SearchEntityArgs)
def search_entity(query: str) -> List[Dict[str, Any]]:
    """
    Search for entities in the database using a fuzzy search.
    Returns a list of potential matches with scores.
    """
    return search_tool_instance.search(query, limit=10)

class EntityExtractionAgent:
    def __init__(self, model_name: str = None):
        self._initialize_llm(model_name)
        self.tools = [search_entity]
        
        # Define the system prompt
        self.system_message = """You are an expert entity extractor for MMS marketing messages.
Your goal is to extract product or service entities and their associated customer actions.

You have access to a `search_entity` tool that returns a list of potential matches from a database.
The database contains aliases (e.g., "iPhone" -> "아이폰").

Follow these steps:
1. Analyze the message to identify ALL potential product or service names.
2. For each identified name, use the `search_entity` tool to find potential matches in the database.
   - You are encouraged to **expand or modify the search query** if you believe it will improve results (e.g., translating "iPhone" to "아이폰", adding "series", or removing generic words).
   - The database contains aliases, so try to find the best match.
3. Map each entity to one of the following actions: ["구매", "가입", "사용", "방문", "참여", "코드입력", "쿠폰다운로드", "기타"].
4. Aggregate the results into the following JSON format:

{
    "product": [
        {
            "item_nm": "The standardized item name from the database (or best guess if not found)",
            "item_id": [
                "List of matching ITEM_IDs from the search results"
            ],
            "item_name_in_msg": [
                "List of EXACT phrases from the original message. Do NOT use the expanded query here."
            ],
            "expected_action": [
                "List of expected actions (e.g. '구매', '가입')"
            ]
        }
    ]
}

If no entities are found, return {"product": []}.

Minimize the number of tool calls. Try to find the best match in one go.
"""
        # Bind tools to LLM
        self.llm_with_tools = self.llm.bind_tools(self.tools)

    def _initialize_llm(self, model_name_arg: str = None):
        """Initialize LLM using logic from mms_extractor_data.py"""
        try:
            if MODEL_CONFIG and API_CONFIG:
                # Logic from mms_extractor_data.py
                model_mapping = {
                    "gemma": getattr(MODEL_CONFIG, 'gemma_model', 'gemma-7b'),
                    "gem": getattr(MODEL_CONFIG, 'gemma_model', 'gemma-7b'),
                    "ax": getattr(MODEL_CONFIG, 'ax_model', 'ax-4'),
                    "claude": getattr(MODEL_CONFIG, 'claude_model', 'claude-4'),
                    "cld": getattr(MODEL_CONFIG, 'claude_model', 'claude-4'),
                    "gemini": getattr(MODEL_CONFIG, 'gemini_model', 'gemini-pro'),
                    "gen": getattr(MODEL_CONFIG, 'gemini_model', 'gemini-pro'),
                    "gpt": getattr(MODEL_CONFIG, 'gpt_model', 'gpt-4')
                }
                
                # Use argument if provided, otherwise use config default
                target_model_alias = model_name_arg if model_name_arg else getattr(MODEL_CONFIG, 'llm_model', 'gemini-pro')
                actual_model_name = model_mapping.get(target_model_alias, target_model_alias)
                
                model_kwargs = {
                    "temperature": 0.0,
                    "openai_api_key": getattr(API_CONFIG, 'llm_api_key', os.getenv('OPENAI_API_KEY')),
                    "openai_api_base": getattr(API_CONFIG, 'llm_api_url', None),
                    "model": actual_model_name,
                    "max_tokens": getattr(MODEL_CONFIG, 'llm_max_tokens', 4000)
                }
                
                if 'gpt' in actual_model_name.lower():
                    model_kwargs["seed"] = 42
                    
                self.llm = ChatOpenAI(**model_kwargs)
                logger.info(f"LLM initialized: {target_model_alias} ({actual_model_name})")
            else:
                # Fallback if config not available
                logger.warning("Config not available, using default ChatOpenAI init")
                self.llm = ChatOpenAI(model=model_name_arg or "gpt-4o", temperature=0)
                
        except Exception as e:
            logger.error(f"LLM initialization failed: {e}")
            # Fallback
            self.llm = ChatOpenAI(model="gen", temperature=0)

    def process_message(self, message: str) -> Dict[str, Any]:
        """
        Process the message and extract the entity.
        """
        try:
            # Initial messages
            messages = [
                SystemMessage(content=self.system_message),
                HumanMessage(content=message)
            ]
            
            # First turn: Get tool calls
            response = self.llm_with_tools.invoke(messages)
            
            if response.tool_calls:
                # Execute tools
                tool_results = []
                for tool_call in response.tool_calls:
                    if tool_call['name'] == 'search_entity':
                        # Execute search
                        query = tool_call['args'].get('query')
                        logger.info(f"Agent calling search_entity with query: '{query}'")
                        result = search_entity.invoke(tool_call['args'])
                        logger.info(f"Search result length: {len(result)}")
                        tool_results.append(f"Tool 'search_entity' with query '{query}' returned: {result}")
                
                # Workaround for Gemini/Proxy 400 Error:
                # Instead of appending AIMessage(tool_calls) and ToolMessage(result),
                # we append a HumanMessage summarizing the tool result.
                # This avoids sending the complex tool_calls history which the API seems to reject.
                
                summary = "\n".join(tool_results)
                messages.append(HumanMessage(content=f"System Notification: The tools were executed. Results:\n{summary}\n\nPlease use this information to extract the entity as requested."))
                
                # Second turn: Get final answer
                final_response = self.llm_with_tools.invoke(messages)
                return final_response.content
            else:
                # No tool calls, return direct response
                return response.content
                
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return {"error": str(e)}

if __name__ == "__main__":
    # Test
    agent = EntityExtractionAgent()
    msg = """
[SK텔레콤] 반가워요 5G 아이폰17/ 17 Pro 사전예약 안내\n(광고)[SKT] 아이폰 17/17 Pro 사전예약 안내  #04 고객님, 안녕하세요. 최고의 스마트폰 칩과 카메라, 견고한 세라믹 실드에 5G 기술까지! 이 모든 것을 갖춘 아이폰 17/17 Pro를 만나 보세요.  ▶ 혜택받고 사전예약하기: http://t-mms.kr/t.do?m=#61&u=https://bit.ly/2HeWcdx   ■ 사전예약 기간 - 2020년 10월 23일(금)~10월 29일(목) * 2020년 10월 30일(금)부터 순서대로 배송 후 개통 진행  ■ 아이폰 17/17 Pro 스펙 - Hi, Speed. 아이폰 최초의 5G 지원 - 스마트폰 사상 가장 빠른 A14 Bionic 칩 - 매끈하고 강화된 내구성을 가진 세라믹 글라스 적용 디자인 - 저조도 사진의 품질을 한 차원 끌어올려 주는 카메라 시스템 - 최초의 Dolby Vision 영상 카메라 탑재   ■ T다이렉트샵 특별 사은품(택1) ① [사죠영] 죠르디 한정판 기프트 ② [프리디] 멀티 무선 충전기 ③ [에이프릴스톤] 보조배터리+멀티백 ④ [크레앙] 3in1 무선 충전 살균기  ※ 이 외에도 더 많은 T기프트가 있습니다. .  ▶ T다이렉트샵 카카오톡 상담하기: http://t-mms.kr/t.do?m=#61&u=https://bit.ly/3o7zOnA  ■ 문의: SKT 고객센터(1558, 무료)   ※ 코로나19 확산으로 고객센터에 문의가 증가하고 있습니다. 고객센터와 전화 연결이 원활하지 않을 수 있으니 양해 바랍니다.  SKT와 함께해주셔서 감사합니다. 무료 수신거부 1504
    """
    print(agent.process_message(msg))

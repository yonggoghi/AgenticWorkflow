
import pandas as pd
import re
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from config import settings

# Mock settings if needed, or assume environment is set
llm_api_key = settings.API_CONFIG.llm_api_key
llm_api_url = settings.API_CONFIG.llm_api_url

llm_model = ChatOpenAI(
    temperature=0,
    openai_api_key=llm_api_key,
    openai_api_base=llm_api_url,
    model=settings.ModelConfig.gpt_model, # Using GPT for testing
    max_tokens=1024
)

def extract_entities_single_shot(llm_model, msg_text, item_pdf_all, stop_item_names=[], rank_limit=200):
    """
    Single-shot entity extraction with combined Prompt (COT + Entity + DAG).
    """
    
    # 1. Combined Prompt
    COMBINED_PROMPT = """
Analyze the advertisement to extract **User Action Paths** and **Promoted Entities**.

## Goal
1. Identify the core products/services being explicitly promoted.
2. Construct a DAG representing the user's action flow.

## Guidelines
1. **Analyze First (COT):** Think step-by-step. 
    - Distinguish between the *main offer* (e.g., "iPhone 17") and *generic terms* or *partners* (e.g., "Starbucks" coupon).
    - Only select entities that are the *primary subject* of the promotion or a *specific benefit*.
    - **Crucial:** Do not include navigational terms like 'link', 'click', 'shortcut'.
2. **Entity Extraction:**
    - Extract entities **exactly as they appear** in the text.
    - If the text says "T Day", output "T Day".
3. **DAG Construction:**
    - Format: `(Node:Action) -[Edge]-> (Node:Action)`
    - Root: Entry point (Store, App, Plan).
    - Value: Final benefit.

## Output Format
REASON: <Short analysis of what is being promoted vs. what is just context>
ENTITY: <comma-separated list of CORE promoted entities>
DAG:
<DAG representation>
"""

    prompt = f"""
    {COMBINED_PROMPT}

    ## message:
    {msg_text}
    """

    # 2. Call LLM
    try:
        print(f"Sending prompt to LLM...")
        chain = PromptTemplate(template="{prompt}", input_variables=["prompt"]) | llm_model
        response = chain.invoke({"prompt": prompt}).content
        
        print(f"\n{'='*40}")
        print(f"[LLM Response]")
        print(f"{'='*40}")
        print(response)
        print(f"{'='*40}\n")
        
        # 3. Parse Response
        entity_list = []
        
        # Parse ENTITY: line
        # More robust regex to catch multi-line or slightly malformed output
        entity_match = re.search(r'ENTITY:\s*(.*?)(?:\n|$)', response, re.IGNORECASE | re.DOTALL)
        if entity_match:
            raw_entities = entity_match.group(1).split(',')
            entity_list = [e.strip() for e in raw_entities if e.strip() and e.strip().lower() not in ['none', 'null', 'empty']]

        # Filter stop words
        entity_list = [e for e in entity_list if e not in stop_item_names]
        
        print(f"Parsed Entities: {entity_list}")
        
        if not entity_list:
            print("No entities found.")
            return pd.DataFrame()

        # 4. Fuzzy Matching (Mocking the dependency for this snippet test)
        # In a real scenario, we would import parallel_fuzzy_similarity etc.
        # Here we just check if they exist in our mock item_pdf_all
        
        # Mock item_pdf_all for testing
        # Assuming item_pdf_all has 'item_nm_alias' column
        
        valid_entities = []
        if not item_pdf_all.empty:
             known_aliases = item_pdf_all['item_nm_alias'].unique()
             # Simple exact match for this test, or simple containment
             for ent in entity_list:
                 # Check if ent is roughly in known_aliases (simulating fuzzy match)
                 # For test purposes, we'll just accept it if it looks valid
                 valid_entities.append(ent)
        else:
            valid_entities = entity_list

        return valid_entities

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()

# Test Data
msg_text = """
[SKT] T 우주패스 쇼핑 출시! 
지금 링크를 눌러 가입하면 첫 달 1,000원에 이용 가능합니다. 
가입 고객 전원에게 11번가 포인트 3,000P와 아마존 무료배송 쿠폰을 드립니다.
"""

# Mock DataFrame
item_pdf_all = pd.DataFrame({'item_nm_alias': ['T 우주패스', '11번가', '아마존', 'T 우주패스 쇼핑']})

# Run Test
result = extract_entities_single_shot(llm_model, msg_text, item_pdf_all)
print(f"\nFinal Result: {result}")

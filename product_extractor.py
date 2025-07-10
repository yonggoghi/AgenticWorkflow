import pandas as pd
import json
import requests
from typing import List, Dict, Any
import re
from concurrent.futures import ThreadPoolExecutor
import logging
from dataclasses import dataclass
from tqdm import tqdm
from openai import OpenAI

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class LLMConfig:
    """Configuration for LLM API endpoints"""
    url: str
    api_key: str
    poor_model_name: str
    good_model_name: str

def extract_json_objects(text):
    """Extract JSON objects from text using regex and parsing."""
    pattern = r'(\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\}))*\}))*\})'
    
    result = []
    for match in re.finditer(pattern, text):
        potential_json = match.group(0)
        try:
            json_obj = json.loads(potential_json)
            result.append(json_obj)
        except:
            pass
    
    return result

class ProductExtractor:
    def __init__(self, llm_config: LLMConfig):
        """Initialize the product extractor with LLM configurations"""
        self.product_metadata = self._load_product_metadata()
        self.llm_config = llm_config
        self.client = OpenAI(
            api_key = llm_config.api_key,
            base_url = llm_config.url
        )

    def _load_product_metadata(self) -> pd.DataFrame:
        """Load product metadata from CSV file"""
        try:
            return pd.read_csv('data/item_info_all_250527.csv')
        except FileNotFoundError:
            logger.warning("Product metadata file not found. Creating empty DataFrame.")
            return pd.DataFrame(columns=['item_nm', 'item_id', 'description'])
    
    def _call_llm(self, client: OpenAI, model_name: str, message: str, system_prompt: str = None) -> Dict[str, Any]:
        schema = {
            'product': {
                'type': 'array',
                'items': {
                    'type': 'object',
                    'properties': {
                        'name': {'type': 'string', 'description': '광고하는 제품이나 서비스 이름'},
                        'action': {'type': 'string', 'description': '고객에게 기대하는 행동: [구매, 가입, 사용, 방문, 참여, 코드입력, 쿠폰다운로드, 기타] 중에서 선택'}
                    }
                }
            }, 
            'objectType': 'object'
        }

        if system_prompt is None:
            system_prompt = "당신은 SKT 캠페인 메시지에서 정확한 정보를 추출하는 전문가입니다."

        user_message = f"""
{system_prompt}

### 분석 대상 광고 메세지 ###
{message}

### 결과 Schema ###
{json.dumps(schema, indent=2, ensure_ascii=False)}
"""

        try:
            response = client.chat.completions.create(
                model=model_name,
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.0,
                max_tokens=4000,
                top_p=0.95,
                frequency_penalty=0.0,
                presence_penalty=0.0,
                response_format={"type": "json_object"}
            )
            
            # Extract the JSON from the response
            result_json_text = response.choices[0].message.content
            json_objects = extract_json_objects(result_json_text)[0]
            
            # Extract the content from the response
            if "product" in json_objects and len(json_objects["product"]) > 0:
                
                if json_objects:
                    # Use the first valid JSON object found
                    result = json_objects
                    
                    # Ensure the result has the required structure
                    if not isinstance(result, list):
                        result = [result]
                    
                    # Validate and clean each item in the result
                    cleaned_result = []
                    for item in result:
                        if isinstance(item, dict):
                            cleaned_item = {
                                "name": str(item.get("name", "")),
                                "action": str(item.get("action", "기타"))
                            }
                            # Validate action is one of the allowed values
                            allowed_actions = ["구매", "가입", "사용", "방문", "참여", "코드입력", "쿠폰다운로드", "기타"]
                            if cleaned_item["action"] not in allowed_actions:
                                cleaned_item["action"] = "기타"
                            cleaned_result.append(cleaned_item)
                    
                    return cleaned_result
                else:
                    logger.error(f"No valid JSON found in response: {result_json_text}")
                    return []
            else:
                logger.error(f"Unexpected response format: {result_json_text}")
                return []
                
        except Exception as e:
            logger.error(f"Error calling LLM API: {str(e)}")
            return []
    
    def _extract_products_poor_llm(self, message: str, system_prompt: str = None) -> Dict[str, Any]:
        """Extract products using the poor LLM"""
        response = self._call_llm(self.client, self.llm_config.poor_model_name, message, system_prompt)
        if "error" in response:
            return {"products": [], "expected_action": "unknown"}
        return response
    
    def _extract_products_good_llm(self, message: str) -> Dict[str, Any]:
        """Extract products using the good LLM with a constant prompt"""
        base_prompt = """당신은 SKT 캠페인 메시지에서 정확한 정보를 추출하는 전문가입니다.
1. 광고 메시지에서 제품명을 정확히 식별하세요.
2. 고객 행동은 [구매, 가입, 사용, 방문, 참여, 코드입력, 쿠폰다운로드, 기타] 중에서만 선택하세요.
3. 제품명과 행동을 정확하게 매칭하세요."""
        
        response = self._call_llm(self.client, self.llm_config.good_model_name, message, base_prompt)
        if "error" in response:
            return {"products": [], "expected_action": "unknown"}
        return response
    
    def _calculate_reward(self, poor_result: List[Dict[str, str]], good_result: List[Dict[str, str]]) -> float:
        """Calculate reward based on the difference between poor and good LLM results"""
        if not poor_result and not good_result:
            return 1.0
        
        # Extract product names and actions
        poor_products = {(item["name"], item["action"]) for item in poor_result}
        good_products = {(item["name"], item["action"]) for item in good_result}
        
        # Calculate metrics
        if not poor_products and not good_products:
            return 1.0
        
        precision = len(poor_products & good_products) / len(poor_products) if poor_products else 0
        recall = len(poor_products & good_products) / len(good_products) if good_products else 0
        
        if precision + recall == 0:
            return 0.0
            
        f1_score = 2 * (precision * recall) / (precision + recall)
        
        # Log detailed comparison
        logger.debug(f"Poor LLM found {len(poor_products)} products")
        logger.debug(f"Good LLM found {len(good_products)} products")
        logger.debug(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1_score:.4f}")
        
        if poor_products != good_products:
            missing = good_products - poor_products
            extra = poor_products - good_products
            if missing:
                logger.debug(f"Products missed by poor LLM: {missing}")
            if extra:
                logger.debug(f"Extra products found by poor LLM: {extra}")
        
        return f1_score
    
    def _evolve_prompt(self, current_prompt: str, reward: float, poor_result: List[Dict[str, str]], good_result: List[Dict[str, str]], 
                      iteration_history: List[Dict[str, Any]] = None) -> str:
        """Evolve the prompt by refining extraction instructions based on performance and history"""
        base_prompt = """당신은 SKT 캠페인 메시지에서 정확한 정보를 추출하는 전문가입니다.
1. 광고 메시지에서 제품명을 정확히 식별하세요.
2. 고객 행동은 [구매, 가입, 사용, 방문, 참여, 코드입력, 쿠폰다운로드, 기타] 중에서만 선택하세요.
3. 제품명과 행동을 정확하게 매칭하세요."""

        # Extract current instructions (everything after the base prompt)
        current_instructions = current_prompt[len(base_prompt):].strip() if current_prompt.startswith(base_prompt) else ""
        
        if reward < 0.3:
            new_instructions = "\n추출 지침:\n- 제품명은 광고 메시지에서 직접적으로 언급된 이름을 사용하세요.\n- 행동은 메시지의 핵심 목적을 파악하여 선택하세요."
        elif reward < 0.5:
            # Compare results to identify specific issues
            poor_products = {(item["name"], item["action"]) for item in poor_result}
            good_products = {(item["name"], item["action"]) for item in good_result}
            
            missing_products = good_products - poor_products
            extra_products = poor_products - good_products
            
            # Analyze previous iterations if available
            if iteration_history:
                # Check if we've been trying similar instructions before
                similar_instructions = [h for h in iteration_history if h['reward'] < 0.5]
                if similar_instructions:
                    # If we've been trying similar approaches without success, try a different strategy
                    if missing_products:
                        new_instructions = "\n추출 지침:\n- 제품명이 부분적으로 언급되어도 포함하세요.\n- 광고 메시지의 모든 제품 관련 정보를 주의 깊게 확인하세요.\n- 이전에 놓친 제품들을 다시 한번 확인하세요."
                    elif extra_products:
                        new_instructions = "\n추출 지침:\n- 직접적으로 언급된 제품만 포함하세요.\n- 추측이나 일반적인 제품명은 제외하세요.\n- 이전에 잘못 포함된 제품들을 주의하세요."
                    else:
                        new_instructions = "\n추출 지침:\n- 제품명과 행동의 관계를 더 정확하게 파악하세요.\n- 행동은 메시지의 주요 목적에 맞게 선택하세요.\n- 이전 시도에서의 실수를 피하세요."
                else:
                    # First time trying these instructions
                    if missing_products:
                        new_instructions = "\n추출 지침:\n- 제품명이 부분적으로 언급되어도 포함하세요.\n- 광고 메시지의 모든 제품 관련 정보를 주의 깊게 확인하세요."
                    elif extra_products:
                        new_instructions = "\n추출 지침:\n- 직접적으로 언급된 제품만 포함하세요.\n- 추측이나 일반적인 제품명은 제외하세요."
                    else:
                        new_instructions = "\n추출 지침:\n- 제품명과 행동의 관계를 더 정확하게 파악하세요.\n- 행동은 메시지의 주요 목적에 맞게 선택하세요."
            else:
                # No history available, use basic instructions
                if missing_products:
                    new_instructions = "\n추출 지침:\n- 제품명이 부분적으로 언급되어도 포함하세요.\n- 광고 메시지의 모든 제품 관련 정보를 주의 깊게 확인하세요."
                elif extra_products:
                    new_instructions = "\n추출 지침:\n- 직접적으로 언급된 제품만 포함하세요.\n- 추측이나 일반적인 제품명은 제외하세요."
                else:
                    new_instructions = "\n추출 지침:\n- 제품명과 행동의 관계를 더 정확하게 파악하세요.\n- 행동은 메시지의 주요 목적에 맞게 선택하세요."
        elif reward < 0.7:
            # Check if we've been improving
            if iteration_history and iteration_history[-1]['reward'] < reward:
                new_instructions = "\n추출 지침:\n- 현재 접근 방식이 효과적이니 계속하세요.\n- 광고 메시지의 핵심 정보를 더 세밀하게 분석하세요.\n- 제품명과 행동의 관계를 더 정확하게 파악하세요.\n- 메시지의 주요 목적에 집중하세요."
            else:
                new_instructions = "\n추출 지침:\n- 광고 메시지의 핵심 정보를 더 세밀하게 분석하세요.\n- 제품명과 행동의 관계를 더 정확하게 파악하세요.\n- 메시지의 주요 목적에 집중하세요."
        else:
            return current_prompt  # Keep current instructions if performance is good
        
        # Combine base prompt with new instructions
        return base_prompt + new_instructions

    def process_messages(self, messages: List[str], num_iterations: int = 10) -> List[Dict[str, Any]]:
        """Process messages using the AlphaEvolve-inspired approach"""
        results = []
        # Initial prompt for poor LLM
        current_prompt = """당신은 SKT 캠페인 메시지에서 정확한 정보를 추출하는 전문가입니다.
1. 광고 메시지에서 제품명을 정확히 식별하세요.
2. 고객 행동은 [구매, 가입, 사용, 방문, 참여, 코드입력, 쿠폰다운로드, 기타] 중에서만 선택하세요.
3. 제품명과 행동을 정확하게 매칭하세요.
추출 지침:
- 제품명은 광고 메시지에서 직접적으로 언급된 이름을 사용하세요.
- 행동은 메시지의 핵심 목적을 파악하여 선택하세요."""
        
        logger.info(f"Starting message processing with initial prompt for poor LLM: {current_prompt}")
        
        # Track overall statistics
        total_messages = len(messages)
        successful_evolutions = 0
        best_rewards = []
        
        for message_idx, message in enumerate(tqdm(messages, desc="Processing messages")):
            logger.info(f"\n{'='*50}")
            logger.info(f"Processing message {message_idx + 1}/{total_messages}")
            logger.info(f"{'='*50}")
            best_reward = 0.0
            best_result = None
            best_prompt = current_prompt
            message_evolutions = 0
            iteration_history = []  # Track history for this message
            
            for iteration in range(num_iterations):
                logger.info(f"\n{'-'*50}")
                logger.info(f"Iteration {iteration + 1}/{num_iterations}")
                logger.info(f"{'-'*50}")
                logger.info("Current prompt:")
                logger.info(f"{current_prompt}")
                
                # Get results from both LLMs
                poor_result = self._extract_products_poor_llm(message, current_prompt)
                good_result = self._extract_products_good_llm(message)  # Uses constant prompt
                
                # Calculate reward
                reward = self._calculate_reward(poor_result, good_result)
                logger.info(f"Reward: {reward:.4f}")
                
                # Update best result if current reward is better
                if reward > best_reward:
                    best_reward = reward
                    best_result = poor_result
                    best_prompt = current_prompt
                    logger.info(f"New best reward achieved!")
                
                # Store iteration results
                iteration_history.append({
                    'prompt': current_prompt,
                    'reward': reward,
                    'poor_result': poor_result,
                    'good_result': good_result
                })
                
                # Evolve prompt for poor LLM only
                old_prompt = current_prompt
                current_prompt = self._evolve_prompt(current_prompt, reward, poor_result, good_result, iteration_history)
                if old_prompt != current_prompt:
                    message_evolutions += 1
                    successful_evolutions += 1
                    logger.info(f"\nPrompt evolved:")
                    logger.info("Old prompt:")
                    logger.info(f"{old_prompt}")
                    logger.info("\nNew prompt:")
                    logger.info(f"{current_prompt}")
            
            best_rewards.append(best_reward)
            logger.info(f"\n{'-'*50}")
            logger.info(f"Completed message {message_idx + 1}:")
            logger.info(f"Final reward: {best_reward:.4f}")
            logger.info(f"Number of prompt evolutions: {message_evolutions}")
            logger.info("\nBest prompt:")
            logger.info(f"{best_prompt}")
            results.append(best_result)
        
        # Log overall statistics
        avg_reward = sum(best_rewards) / len(best_rewards)
        logger.info(f"\n{'='*50}")
        logger.info("Processing Summary:")
        logger.info(f"{'='*50}")
        logger.info(f"Total messages processed: {total_messages}")
        logger.info(f"Average best reward: {avg_reward:.4f}")
        logger.info(f"Total successful prompt evolutions: {successful_evolutions}")
        logger.info(f"Average evolutions per message: {successful_evolutions/total_messages:.2f}")
        
        return results

def main():
    # LLM configurations
    llm_config = LLMConfig(
        url="https://api.platform.a15t.com/v1",
        api_key="sk-gapk-Y70vdkPbXPRMWHK0dtaYU30hw-bi7B5C",
        poor_model_name="skt/a.x-3-lg",
        good_model_name="skt/claude-3-5-sonnet-20241022"
    )
    
    # Initialize extractor
    extractor = ProductExtractor(llm_config)
    
    # Load messages
    messages_df = pd.read_csv('data/mms_sample.csv')
    messages = messages_df['msg'].tolist()
    
    # Process messages
    results = extractor.process_messages(messages)
    
    # Save results
    with open('results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main() 
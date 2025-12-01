"""
Simple MMS Agent
Direct tool calling approach for maximum compatibility
"""

from typing import Dict, Any, List
import json
from ..core.llm_client import get_llm
from ..tools import (
    # Non-LLM tools
    search_entities_kiwi,
    search_entities_fuzzy,
    classify_program,
    match_store_info,
    validate_entities,
    # LLM tools
    extract_entities_llm,
    extract_main_info,
    extract_entity_dag
)


class SimpleMMSAgent:
    """
    Simple MMS Agent with direct orchestration
    
    Follows a predefined strategy:
    1. Program classification
    2. Entity extraction (Kiwi first, then LLM if needed)
    3. Main information extraction
    4. Store matching (if detected)
    5. DAG extraction (optional)
    """
    
    def __init__(self, verbose: bool = True):
        """Initialize agent"""
        self.verbose = verbose
        self.llm = get_llm()
    
    def _log(self, message: str):
        """Log if verbose"""
        if self.verbose:
            print(f"[Agent] {message}")
    
    def process_message(self, message: str) -> Dict[str, Any]:
        """
        Process MMS message with orchestrated tool calls
        
        Args:
            message: MMS advertisement message
            
        Returns:
            Extracted information
        """
        results = {
            "message": message,
            "program_info": None,
            "entities": None,
            "main_info": None,
            "store_info": None,
            "dag": None
        }
        
        try:
            # Step 1: Program classification
            self._log("Step 1: Program classification...")
            results["program_info"] = classify_program.invoke({
                "message": message,
                "top_k": 5
            })
            self._log(f"  → Found {len(results['program_info'].get('programs', []))} programs")
            
            # Step 2: Entity extraction (Kiwi first)
            self._log("Step 2: Entity extraction (Kiwi)...")
            kiwi_result = search_entities_kiwi.invoke({"message": message})
            entity_count = len(kiwi_result.get('entities', []))
            self._log(f"  → Kiwi found {entity_count} entities")
            
            # Decision: Need LLM entity extraction?
            if entity_count < 30000:
                self._log("Step 2b: Additional LLM entity extraction...")
                llm_entities_str = extract_entities_llm.invoke({"message": message})
                llm_entities = json.loads(llm_entities_str)
                self._log(f"  → LLM found {len(llm_entities) if isinstance(llm_entities, list) else 0} additional entities")
                results["entities"] = llm_entities
            else:
                self._log("  → Kiwi entities sufficient")
                # Convert to expected format
                entities_str = search_entities_fuzzy.invoke({
                    "entities": ",".join(kiwi_result.get('entities', [])),
                    "threshold": 0.5
                })
                results["entities"] = json.loads(entities_str)
            
            # Step 3: Main information extraction
            self._log("Step 3: Main information extraction...")
            
            # Build context from program info
            context = f"# 프로그램 분류 결과\n{results['program_info'].get('context', '')}"
            
            # Choose mode based on entities
            if entity_count >= 3:
                mode = "rag"  # We have candidates
                self._log(f"  → Using RAG mode (엔티티 {entity_count}개)")
            else:
                mode = "llm"  # Free extraction
                self._log("  → Using LLM mode (자유 추출)")
            
            main_info_str = extract_main_info.invoke({
                "message": message,
                "mode": mode,
                "context": context
            })
            results["main_info"] = json.loads(main_info_str)
            self._log("  → Main info extracted")
            
            # Step 4: Store matching (if detected)
            # Look for store keywords in message
            store_keywords = ["대리점", "직영점", "매장", "지점"]
            if any(kw in message for kw in store_keywords):
                self._log("Step 4: Store info matching...")
                # Extract store name (simple heuristic)
                import re
                store_pattern = r'([가-힣\s]+대리점|[가-힣\s]+직영점)'
                store_match = re.search(store_pattern, message)
                if store_match:
                    store_name = store_match.group(1).strip()
                    store_info_str = match_store_info.invoke({"store_name": store_name})
                    results["store_info"] = json.loads(store_info_str)
                    self._log(f"  → Found {len(results['store_info']) if isinstance(results['store_info'], list) else 0} stores")
            
            # Step 5: DAG extraction (optional - check complexity)
            # Simple heuristic: if message is long and has multiple products
            if len(message) > 200 and len(results.get("entities", [])) > 3:
                self._log("Step 5: DAG extraction (complex message detected)...")
                dag_str = extract_entity_dag.invoke({"message": message})
                results["dag"] = json.loads(dag_str)
                self._log("  → DAG extracted")
            
            self._log("✅ Processing complete!")
            return {
                "success": True,
                "results": results
            }
            
        except Exception as e:
            self._log(f"❌ Error: {e}")
            return {
                "success": False,
                "error": str(e),
                "partial_results": results
            }


# Convenience function
def process_mms_message(message: str, verbose: bool = True) -> Dict[str, Any]:
    """
    Process MMS message using Simple Agent
    
    Args:
        message: MMS advertisement message
        verbose: Print progress
        
    Returns:
        Extracted information
    """
    agent = SimpleMMSAgent(verbose=verbose)
    return agent.process_message(message)


if __name__ == "__main__":
    # Test
    test_message = """
갤럭시 Z 플립7/폴드7 구매 혜택
- 최대 할인 제공
- 갤럭시 워치 무료 증정(5GX 프라임 요금제 이용 시)

아이폰 신제품 구매 혜택
- 최대 할인 및 쓰던 폰 반납 시 최대 보상 제공
- 아이폰 에어 구매 시 에어팟 증정(5GX 프라임 요금제 이용 시)

문의: SKT 고객센터(1558, 무료)"""
    
    print("🤖 Simple MMS Agent 테스트\n")
    print("=" * 60)
    
    result = process_mms_message(test_message)
    
    print("\n" + "=" * 60)
    if result['success']:
        print("✅ 성공!\n")
        print(json.dumps(result['results'], indent=2, ensure_ascii=False))
    else:
        print(f"❌ 실패: {result['error']}\n")
        if 'partial_results' in result:
            print("부분 결과:")
            print(json.dumps(result['partial_results'], indent=2, ensure_ascii=False))

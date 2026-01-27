#!/usr/bin/env python3
"""
Simplified test for architectural improvements:
1. Typed WorkflowState dataclass
2. Conditional DAG step registration (will check code structure)
"""

import sys
sys.path.insert(0, '.')

from core.workflow_core import WorkflowState
import pandas as pd

def test_typed_state():
    """Test typed WorkflowState functionality"""
    print("\n" + "="*60)
    print("TEST 1: Typed WorkflowState Dataclass")
    print("="*60)
    
    # Test 1: Create with required fields
    print("\n✓ Creating state with required fields...")
    state = WorkflowState(mms_msg="test message", extractor=None)
    assert state.mms_msg == "test message"
    assert state.extractor is None
    print(f"  ✅ Required fields: mms_msg={state.mms_msg}, extractor={state.extractor}")
    
    # Test 2: Default values for optional fields
    print("\n✓ Checking default values...")
    assert state.msg == ""
    assert state.is_fallback == False
    assert isinstance(state.entities_from_kiwi, list)
    assert len(state.entities_from_kiwi) == 0
    assert isinstance(state.cand_item_list, pd.DataFrame)
    assert state.cand_item_list.empty
    print(f"  ✅ Optional fields have correct defaults")
    
    # Test 3: Direct attribute access (typed)
    print("\n✓ Testing typed attribute access...")
    state.msg = "validated message"
    state.is_fallback = True
    state.entities_from_kiwi = ["entity1", "entity2"]
    assert state.msg == "validated message"
    assert state.is_fallback == True
    assert len(state.entities_from_kiwi) == 2
    print(f"  ✅ Direct attribute access works")
    
    # Test 4: Backward compatible get/set methods
    print("\n✓ Testing backward compatible methods...")
    assert state.get("mms_msg") == "test message"
    assert state.get("msg") == "validated message"
    assert state.get("nonexistent", "default") == "default"
    
    state.set("new_field", "new_value")
    assert state.get("new_field") == "new_value"
    assert state.new_field == "new_value"  # Also accessible as attribute
    print(f"  ✅ Backward compatible get/set works")
    
    # Test 5: Error tracking
    print("\n✓ Testing error tracking...")
    assert not state.has_error()
    state.add_error("Test error 1")
    state.add_error("Test error 2")
    assert state.has_error()
    assert len(state.get_errors()) == 2
    print(f"  ✅ Error tracking works: {state.get_errors()}")
    
    # Test 6: History tracking
    print("\n✓ Testing history tracking...")
    state.add_history("Step1", 1.5, "success")
    state.add_history("Step2", 0.8, "success")
    assert len(state.get_history()) == 2
    print(f"  ✅ History tracking works: {len(state.get_history())} entries")
    
    # Test 7: Dataclass repr
    print("\n✓ Testing dataclass representation...")
    repr_str = repr(state)
    print(f"  State repr: {repr_str}")
    assert "WorkflowState" in repr_str
    assert "mms_msg" in repr_str
    print(f"  ✅ Dataclass repr works")
    
    print("\n" + "="*60)
    print("✅ ALL TYPED STATE TESTS PASSED!")
    print("="*60)
    return True

def test_conditional_dag_code_structure():
    """Test that code structure supports conditional DAG registration"""
    print("\n" + "="*60)
    print("TEST 2: Conditional DAG Registration (Code Review)")
    print("="*60)
    
    # Read the mms_extractor.py file
    with open('mms_extractor.py', 'r') as f:
        content = f.read()
    
    # Check for conditional DAG registration
    print("\n✓ Checking for conditional DAG registration...")
    assert 'if self.extract_entity_dag:' in content
    assert 'self.workflow_engine.add_step(DAGExtractionStep())' in content
    print("  ✅ Found conditional DAG registration code")
    
    # Verify it's not unconditionally adding DAG step
    lines = content.split('\n')
    dag_add_lines = [i for i, line in enumerate(lines) if 'DAGExtractionStep()' in line]
    
    for line_num in dag_add_lines:
        # Check previous lines for 'if self.extract_entity_dag'
        context = '\n'.join(lines[max(0, line_num-5):line_num+1])
        if 'add_step(DAGExtractionStep())' in context:
            assert 'if self.extract_entity_dag:' in context, \
                f"DAGExtractionStep added unconditionally at line {line_num}"
            print(f"  ✅ DAGExtractionStep at line {line_num+1} is conditional")
    
    print("\n" + "="*60)
    print("✅ CONDITIONAL DAG REGISTRATION VERIFIED!")
    print("="*60)
    return True

if __name__ == "__main__":
    try:
        success = True
        
        # Run tests
        success &= test_typed_state()
        success &= test_conditional_dag_code_structure()
        
        print("\n" + "="*60)
        print("="*60)
        if success:
            print("🎉 ALL ARCHITECTURAL IMPROVEMENTS VERIFIED!")
            print("\nImplemented:")
            print("  1. ✅ Typed WorkflowState with dataclass")
            print("  2. ✅ Conditional DAG step registration")
            print("\nBenefits:")
            print("  • Type safety and IDE autocomplete")
            print("  • Cleaner workflow logic")
            print("  • No unnecessary step execution")
        else:
            print("❌ SOME TESTS FAILED")
        print("="*60)
        print("="*60)
        
        sys.exit(0 if success else 1)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

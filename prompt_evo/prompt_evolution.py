#!/usr/bin/env python3
"""
Prompt Evolution System for SK Telecom MMS Message Analysis

Automatically evolves a Student LLM's system prompt to match Teacher LLM outputs
with anti-overfitting mechanisms (batch-based updates, anchor regression testing, rollback).
"""

import argparse
import json
import os
import random
import signal
import sys
from datetime import datetime
from pathlib import Path
from time import sleep
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


# ============================================================================
# Pydantic Schemas
# ============================================================================

class PromptEvolutionResult(BaseModel):
    """Result from Evaluator LLM for prompt evolution."""
    reasoning: str = Field(description="Analysis and modification rationale")
    identified_patterns: List[str] = Field(description="Common patterns found")
    has_changed: bool = Field(description="Whether prompt was modified")
    new_prompt_text: str = Field(description="Modified or original prompt")
    added_rules: List[str] = Field(description="Newly added rules")


class SimilarityScore(BaseModel):
    """Similarity evaluation between Student and Teacher outputs."""
    score: float = Field(ge=0.0, le=1.0, description="Similarity score 0.0-1.0")
    explanation: str = Field(description="Scoring rationale")


# ============================================================================
# Constants and Defaults
# ============================================================================

DEFAULT_INITIAL_PROMPT = """당신은 SK텔레콤 MMS 마케팅 메시지를 분석하는 AI입니다.

입력된 MMS 메시지를 분석하여 핵심 정보를 추출하시오.
"""

DEFAULT_PROMPTS = {
    "evolution": None,  # Will be loaded from file
    "similarity": None,  # Will be loaded from file
}


# ============================================================================
# Global State for Signal Handling
# ============================================================================

interrupted = False
best_prompt_backup = None


def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully."""
    global interrupted
    interrupted = True
    print("\n\n⚠️  Interrupt received. Saving current best prompt...")


signal.signal(signal.SIGINT, signal_handler)


# ============================================================================
# Utility Functions
# ============================================================================

def log(message: str, verbose: bool = False, force: bool = False):
    """Print log message if verbose or force."""
    if force or verbose:
        print(message)


def load_evaluator_prompt(prompt_type: str) -> str:
    """Load evaluator prompt from file or use default."""
    file_path = f"evaluator_prompts/{prompt_type}_prompt.txt"
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    # Fallback to default (should not happen if files are created)
    raise FileNotFoundError(
        f"Evaluator prompt file not found: {file_path}\n"
        f"Please ensure evaluator_prompts/{prompt_type}_prompt.txt exists."
    )


def load_initial_prompt(file_path: str) -> str:
    """Load initial prompt from file. Use default if empty."""
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        if content:
            return content
        else:
            print(f"⚠️  Warning: {file_path} is empty. Using default initial prompt.")
            return DEFAULT_INITIAL_PROMPT
    raise FileNotFoundError(f"Prompt file not found: {file_path}")


def load_messages(file_path: str) -> List[str]:
    """Load and parse MMS messages from file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split by newlines, filter empty lines
    messages = [msg.strip() for msg in content.split('\n') if msg.strip()]
    return messages


def split_data(messages: List[str], train_ratio: float, seed: int) -> Tuple[List[str], List[str]]:
    """Split messages into train and validation sets."""
    random.seed(seed)
    shuffled = messages.copy()
    random.shuffle(shuffled)
    
    split_idx = int(len(shuffled) * train_ratio)
    train_set = shuffled[:split_idx]
    val_set = shuffled[split_idx:]
    
    return train_set, val_set


def select_anchor_samples(
    train_set: List[str],
    anchor_count: int,
    llm_teacher: Any,
    seed: int
) -> List[Dict[str, str]]:
    """
    Select diverse anchor samples from training set.
    
    Strategy:
    1. Ensure variety by selecting from different positions
    2. Pre-compute Teacher responses for anchors
    """
    if len(train_set) <= anchor_count:
        print(f"⚠️  Warning: train_set size ({len(train_set)}) <= anchor_count ({anchor_count})")
        print(f"    Using all training samples as anchors.")
        anchor_indices = list(range(len(train_set)))
    else:
        # Select evenly distributed samples
        random.seed(seed)
        step = len(train_set) // anchor_count
        anchor_indices = [i * step for i in range(anchor_count)]
    
    anchors = []
    for idx in anchor_indices:
        message = train_set[idx]
        # Pre-compute Teacher response
        teacher_response = llm_teacher.invoke([HumanMessage(content=message)])
        
        anchors.append({
            "index": idx,
            "message": message,
            "teacher_response": teacher_response.content
        })
    
    return anchors


def create_batches(
    train_set: List[str],
    anchor_indices: List[int],
    batch_size: int,
    seed: int
) -> List[List[str]]:
    """Create batches from training set, excluding anchor samples."""
    # Exclude anchors
    non_anchor_messages = [
        msg for i, msg in enumerate(train_set) if i not in anchor_indices
    ]
    
    # Shuffle for variety
    random.seed(seed + 1)  # Different seed for batch shuffling
    random.shuffle(non_anchor_messages)
    
    # Create batches
    batches = []
    for i in range(0, len(non_anchor_messages), batch_size):
        batch = non_anchor_messages[i:i + batch_size]
        batches.append(batch)
    
    return batches


def retry_with_backoff(func, max_retries: int = 3, initial_delay: float = 1.0):
    """Retry function with exponential backoff."""
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            delay = initial_delay * (2 ** attempt)
            print(f"⚠️  API error: {e}. Retrying in {delay}s...")
            sleep(delay)


def extract_json_from_response(response_text: str) -> str:
    """Extract JSON from LLM response, handling markdown code blocks."""
    import re
    
    # Try to extract JSON from markdown code block
    json_match = re.search(r'```(?:json)?\s*\n(.+?)\n```', response_text, re.DOTALL)
    if json_match:
        return json_match.group(1).strip()
    
    # Return as-is if no code block found
    return response_text.strip()



# ============================================================================
# LLM Interaction Functions
# ============================================================================

def initialize_llms(args) -> Tuple[Any, Any, Any]:
    """Initialize Student, Teacher, and Evaluator LLM clients."""
    # Load environment variables
    load_dotenv()
    
    # Get API credentials from .env
    llm_api_key = os.getenv("LLM_API_KEY")
    llm_base_url = os.getenv("LLM_BASE_URL")
    
    if not llm_api_key:
        raise ValueError(
            "LLM_API_KEY not found in environment.\n"
            "Please set it in .env file (format: LLM_API_KEY=sk-...)."
        )
    
    if not llm_base_url:
        raise ValueError(
            "LLM_BASE_URL not found in environment.\n"
            "Please set it in .env file (format: LLM_BASE_URL=https://...)."
        )
    
    # Initialize LLMs with consistent parameters
    llm_student = ChatOpenAI(
        model=args.student_model,
        temperature=0,
        openai_api_key=llm_api_key,
        openai_api_base=llm_base_url,
        max_tokens=4000,
        seed=args.seed
    )
    
    llm_teacher = ChatOpenAI(
        model=args.teacher_model,
        temperature=0,
        openai_api_key=llm_api_key,
        openai_api_base=llm_base_url,
        max_tokens=4000,
        seed=args.seed
    )
    
    llm_evaluator = ChatOpenAI(
        model=args.evaluator_model,
        temperature=0,
        openai_api_key=llm_api_key,
        openai_api_base=llm_base_url,
        max_tokens=4000,
        seed=args.seed
    )
    
    return llm_student, llm_teacher, llm_evaluator


def evaluate_similarity(
    student_response: str,
    teacher_response: str,
    llm_evaluator: Any,
    similarity_prompt: str
) -> float:
    """Evaluate similarity between Student and Teacher responses."""
    prompt = f"""{similarity_prompt}

### 입력
Student 응답:
\"\"\"
{student_response}
\"\"\"

Teacher 응답:
\"\"\"
{teacher_response}
\"\"\"

### 출력 (JSON 형식으로만 응답):
"""
    
    def _call():
        # Get raw response first
        response = llm_evaluator.invoke([HumanMessage(content=prompt)])
        # Extract JSON from potential markdown wrapper
        json_text = extract_json_from_response(response.content)
        # Parse with Pydantic
        import json
        data = json.loads(json_text)
        return SimilarityScore(**data)
    
    result = retry_with_backoff(_call)
    return result.score


def evaluate_anchors(
    prompt: str,
    anchor_set: List[Dict[str, str]],
    llm_student: Any,
    llm_evaluator: Any,
    similarity_prompt: str,
    verbose: bool = False
) -> float:
    """Evaluate prompt performance on anchor samples."""
    scores = []
    
    for anchor in anchor_set:
        # Get Student response with current prompt
        def _call_student():
            return llm_student.invoke([
                SystemMessage(content=prompt),
                HumanMessage(content=anchor["message"])
            ])
        
        res_student = retry_with_backoff(_call_student)
        
        # Evaluate similarity
        score = evaluate_similarity(
            res_student.content,
            anchor["teacher_response"],
            llm_evaluator,
            similarity_prompt
        )
        scores.append(score)
        
        if verbose:
            log(f"    Anchor {anchor['index']}: {score:.2%}", verbose=True)
    
    avg_score = sum(scores) / len(scores)
    return avg_score


def evolve_prompt(
    current_prompt: str,
    batch_results: List[Dict[str, str]],
    llm_evaluator: Any,
    evolution_prompt: str
) -> PromptEvolutionResult:
    """Request prompt evolution from Evaluator LLM."""
    # Format batch results
    batch_text = ""
    for i, result in enumerate(batch_results, 1):
        batch_text += f"""[케이스 {i}]
MMS 메시지: {result['message']}
Student: {result['student']}
Teacher: {result['teacher']}

"""
    
    prompt = f"""{evolution_prompt}

### 입력
현재 System Prompt:
\"\"\"
{current_prompt}
\"\"\"

배치 결과:
{batch_text}

### 출력 (JSON 형식으로만 응답):
"""
    
    def _call():
        # Get raw response first
        response = llm_evaluator.invoke([HumanMessage(content=prompt)])
        # Extract JSON from potential markdown wrapper
        json_text = extract_json_from_response(response.content)
        # Parse with Pydantic
        import json
        data = json.loads(json_text)
        return PromptEvolutionResult(**data)
    
    result = retry_with_backoff(_call)
    return result


# ============================================================================
# Main Evolution Loop
# ============================================================================

def run_evolution(args):
    """Main evolution loop."""
    global best_prompt_backup
    
    # Setup
    verbose = args.verbose
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(exist_ok=True)
    
    log("=" * 80, force=True)
    log("Prompt Evolution System", force=True)
    log("=" * 80, force=True)
    
    # Load prompts
    log("\n📝 Loading prompts...", force=True)
    current_prompt = load_initial_prompt(args.prompt_file)
    evolution_prompt = load_evaluator_prompt("evolution")
    similarity_prompt = load_evaluator_prompt("similarity")
    log(f"✓ Initial prompt loaded from: {args.prompt_file}", verbose=verbose)
    
    # Load and split data
    log("\n📊 Loading data...", force=True)
    messages = load_messages(args.data_file)
    log(f"✓ Loaded {len(messages)} messages from: {args.data_file}", force=True)
    
    train_set, val_set = split_data(messages, args.train_ratio, args.seed)
    log(f"✓ Train: {len(train_set)} messages, Validation: {len(val_set)} messages", force=True)
    
    # Initialize LLMs
    log("\n🤖 Initializing LLMs...", force=True)
    llm_student, llm_teacher, llm_evaluator = initialize_llms(args)
    log(f"✓ Student: {args.student_model}", force=True)
    log(f"✓ Teacher: {args.teacher_model}", force=True)
    log(f"✓ Evaluator: {args.evaluator_model}", force=True)
    
    # Select anchor samples
    log("\n⚓ Selecting anchor samples...", force=True)
    anchor_set = select_anchor_samples(train_set, args.anchor_count, llm_teacher, args.seed)
    anchor_indices = [a["index"] for a in anchor_set]
    log(f"✓ Selected {len(anchor_set)} anchor samples", force=True)
    
    # Save anchor samples
    with open(output_dir / "anchor_samples.json", 'w', encoding='utf-8') as f:
        json.dump([{
            "index": a["index"],
            "message": a["message"][:100] + "..." if len(a["message"]) > 100 else a["message"]
        } for a in anchor_set], f, ensure_ascii=False, indent=2)
    
    # Create batches
    batches = create_batches(train_set, anchor_indices, args.batch_size, args.seed)
    total_batches = len(batches)
    if args.max_iterations:
        batches = batches[:args.max_iterations]
    
    log(f"✓ Created {len(batches)} batches (batch_size={args.batch_size})", force=True)
    
    # Initialize evolution state
    log("\n🔄 Starting evolution loop...", force=True)
    log("=" * 80, force=True)
    
    best_prompt = current_prompt
    best_anchor_score = evaluate_anchors(
        current_prompt, anchor_set, llm_student, llm_evaluator, similarity_prompt, verbose
    )
    best_prompt_backup = best_prompt
    
    log(f"\n📊 Initial anchor score: {best_anchor_score:.2%}", force=True)
    
    history = [{
        "batch_idx": -1,
        "action": "initial",
        "prompt": current_prompt,
        "anchor_score": best_anchor_score
    }]
    
    # Evolution log file
    log_file = output_dir / "evolution_log.jsonl"
    
    accepted_count = 0
    rejected_count = 0
    
    # Main loop
    for batch_idx, batch in enumerate(batches):
        if interrupted:
            log("\n⚠️  Evolution interrupted by user.", force=True)
            break
        
        log(f"\n{'─' * 80}", force=True)
        log(f"Batch {batch_idx + 1}/{len(batches)}", force=True)
        log(f"{'─' * 80}", force=True)
        
        # Step 1: Batch inference
        log(f"  🔍 Running batch inference ({len(batch)} messages)...", verbose=verbose)
        batch_results = []
        
        for message in batch:
            # Student response
            def _call_student():
                return llm_student.invoke([
                    SystemMessage(content=current_prompt),
                    HumanMessage(content=message)
                ])
            
            res_student = retry_with_backoff(_call_student)
            
            # Teacher response
            def _call_teacher():
                return llm_teacher.invoke([HumanMessage(content=message)])
            
            res_teacher = retry_with_backoff(_call_teacher)
            
            batch_results.append({
                "message": message,
                "student": res_student.content,
                "teacher": res_teacher.content
            })
        
        log(f"  ✓ Batch inference complete", verbose=verbose)
        
        # Step 2: Prompt evolution
        log(f"  🧬 Requesting prompt evolution...", verbose=verbose)
        evolution_result = evolve_prompt(
            current_prompt, batch_results, llm_evaluator, evolution_prompt
        )
        
        if not evolution_result.has_changed:
            log(f"  ℹ️  No changes proposed", force=True)
            
            # Log to file
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps({
                    "batch_idx": batch_idx,
                    "action": "no_change",
                    "anchor_score": best_anchor_score
                }, ensure_ascii=False) + '\n')
            
            continue
        
        log(f"  ✓ Evolution proposed", verbose=verbose)
        if verbose:
            log(f"    Patterns: {evolution_result.identified_patterns}", verbose=True)
            log(f"    Added rules: {evolution_result.added_rules}", verbose=True)
        
        # Step 3: Anchor test
        log(f"  ⚓ Testing on anchor samples...", verbose=verbose)
        proposed_anchor_score = evaluate_anchors(
            evolution_result.new_prompt_text,
            anchor_set,
            llm_student,
            llm_evaluator,
            similarity_prompt,
            verbose
        )
        
        log(f"  📊 Anchor score: {proposed_anchor_score:.2%} (best: {best_anchor_score:.2%})", verbose=verbose)
        
        # Step 4: Accept/Reject decision
        threshold = best_anchor_score * args.anchor_threshold
        
        if proposed_anchor_score >= threshold:
            # Accept
            current_prompt = evolution_result.new_prompt_text
            
            if proposed_anchor_score > best_anchor_score:
                best_anchor_score = proposed_anchor_score
                best_prompt = current_prompt
                best_prompt_backup = best_prompt
            
            accepted_count += 1
            
            log(f"  ✅ ACCEPTED (anchor: {proposed_anchor_score:.2%})", force=True)
            log(f"     Added rules: {evolution_result.added_rules}", force=True)
            
            history.append({
                "batch_idx": batch_idx,
                "action": "accepted",
                "prompt": current_prompt,
                "anchor_score": proposed_anchor_score,
                "added_rules": evolution_result.added_rules,
                "reasoning": evolution_result.reasoning
            })
            
            # Log to file
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps({
                    "batch_idx": batch_idx,
                    "action": "accepted",
                    "anchor_score": proposed_anchor_score,
                    "added_rules": evolution_result.added_rules
                }, ensure_ascii=False) + '\n')
        else:
            # Reject
            rejected_count += 1
            
            log(f"  ❌ REJECTED - anchor regression", force=True)
            log(f"     Current best: {best_anchor_score:.2%}, Proposed: {proposed_anchor_score:.2%}", force=True)
            log(f"     Threshold: {threshold:.2%}", force=True)
            
            # Log to file
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps({
                    "batch_idx": batch_idx,
                    "action": "rejected",
                    "anchor_score": proposed_anchor_score,
                    "reason": "anchor_regression"
                }, ensure_ascii=False) + '\n')
        
        # Step 5: Checkpoint
        if (batch_idx + 1) % args.checkpoint_every == 0:
            checkpoint_path = output_dir / "checkpoints" / f"batch_{batch_idx}.json"
            with open(checkpoint_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "batch_idx": batch_idx,
                    "current_prompt": current_prompt,
                    "best_prompt": best_prompt,
                    "best_anchor_score": best_anchor_score,
                    "history": history
                }, f, ensure_ascii=False, indent=2)
            
            log(f"  💾 Checkpoint saved: {checkpoint_path.name}", verbose=verbose)
    
    # Final validation
    log("\n" + "=" * 80, force=True)
    log("Final Validation", force=True)
    log("=" * 80, force=True)
    
    log(f"\n🔍 Running validation on {len(val_set)} samples...", force=True)
    
    validation_scores = []
    for message in val_set:
        # Student response with best prompt
        def _call_student():
            return llm_student.invoke([
                SystemMessage(content=best_prompt),
                HumanMessage(content=message)
            ])
        
        res_student = retry_with_backoff(_call_student)
        
        # Teacher response
        def _call_teacher():
            return llm_teacher.invoke([HumanMessage(content=message)])
        
        res_teacher = retry_with_backoff(_call_teacher)
        
        # Evaluate
        score = evaluate_similarity(
            res_student.content,
            res_teacher.content,
            llm_evaluator,
            similarity_prompt
        )
        validation_scores.append(score)
    
    avg_val_score = sum(validation_scores) / len(validation_scores)
    min_val_score = min(validation_scores)
    max_val_score = max(validation_scores)
    
    # Calculate std
    mean = avg_val_score
    variance = sum((s - mean) ** 2 for s in validation_scores) / len(validation_scores)
    std_val_score = variance ** 0.5
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save best prompt
    final_prompt_path = output_dir / f"final_prompt_{timestamp}.txt"
    with open(final_prompt_path, 'w', encoding='utf-8') as f:
        f.write(best_prompt)
    
    # Save validation results
    validation_results = {
        "timestamp": timestamp,
        "best_anchor_score": best_anchor_score,
        "validation_metrics": {
            "average": avg_val_score,
            "min": min_val_score,
            "max": max_val_score,
            "std": std_val_score,
            "sample_count": len(val_set)
        },
        "evolution_summary": {
            "total_batches": len(batches),
            "accepted_updates": accepted_count,
            "rejected_updates": rejected_count
        },
        "configuration": {
            "student_model": args.student_model,
            "teacher_model": args.teacher_model,
            "evaluator_model": args.evaluator_model,
            "batch_size": args.batch_size,
            "anchor_count": args.anchor_count,
            "anchor_threshold": args.anchor_threshold,
            "train_ratio": args.train_ratio,
            "seed": args.seed
        }
    }
    
    with open(output_dir / "validation_results.json", 'w', encoding='utf-8') as f:
        json.dump(validation_results, f, ensure_ascii=False, indent=2)
    
    # Print summary
    log("\n" + "=" * 80, force=True)
    log("Evolution Summary", force=True)
    log("=" * 80, force=True)
    
    log(f"\n📊 Final Validation Results:", force=True)
    log(f"   Prompt version: best (anchor score: {best_anchor_score:.2%})", force=True)
    log(f"   Total samples: {len(val_set)}", force=True)
    log(f"   Average similarity: {avg_val_score:.2%}", force=True)
    log(f"   Min: {min_val_score:.2%}, Max: {max_val_score:.2%}", force=True)
    log(f"   Std: {std_val_score:.2%}", force=True)
    
    log(f"\n🔄 Evolution Summary:", force=True)
    log(f"   Total batches processed: {len(batches)}", force=True)
    log(f"   Accepted updates: {accepted_count}", force=True)
    log(f"   Rejected updates: {rejected_count}", force=True)
    
    log(f"\n💾 Output files:", force=True)
    log(f"   Final prompt: {final_prompt_path}", force=True)
    log(f"   Evolution log: {log_file}", force=True)
    log(f"   Validation results: {output_dir / 'validation_results.json'}", force=True)
    log(f"   Anchor samples: {output_dir / 'anchor_samples.json'}", force=True)
    
    log("\n✅ Evolution complete!", force=True)


# ============================================================================
# CLI Entry Point
# ============================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Prompt Evolution System for SK Telecom MMS Message Analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Prompt and model configuration
    parser.add_argument(
        "--prompt_file",
        type=str,
        default="prompt.txt",
        help="Initial prompt file path"
    )
    parser.add_argument(
        "--student_model",
        type=str,
        default="skt/ax4",
        help="Student model name"
    )
    parser.add_argument(
        "--teacher_model",
        type=str,
        default="gcp/gemini-2.5-flash",
        help="Teacher model name"
    )
    parser.add_argument(
        "--evaluator_model",
        type=str,
        default="amazon/anthropic/claude-sonnet-4-20250514",
        help="Evaluator model name"
    )
    
    # Data configuration
    parser.add_argument(
        "--data_file",
        type=str,
        default="reg_test.txt",
        help="Training/validation data file"
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.7,
        help="Training data ratio (0.0-1.0)"
    )
    
    # Output configuration
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Output directory"
    )
    
    # Evolution parameters
    parser.add_argument(
        "--batch_size",
        type=int,
        default=3,
        help="Messages per batch"
    )
    parser.add_argument(
        "--anchor_count",
        type=int,
        default=3,
        help="Number of anchor samples"
    )
    parser.add_argument(
        "--anchor_threshold",
        type=float,
        default=0.90,
        help="Anchor score threshold (0.0-1.0)"
    )
    parser.add_argument(
        "--max_iterations",
        type=int,
        default=None,
        help="Maximum batch iterations (None=all)"
    )
    parser.add_argument(
        "--checkpoint_every",
        type=int,
        default=2,
        help="Checkpoint save interval (batches)"
    )
    
    # Reproducibility
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    # Logging
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    try:
        run_evolution(args)
    except KeyboardInterrupt:
        print("\n\n⚠️  Evolution interrupted by user.")
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Save interrupted prompt if available
        if interrupted and best_prompt_backup:
            interrupted_path = Path(args.output_dir) / "interrupted_prompt.txt"
            with open(interrupted_path, 'w', encoding='utf-8') as f:
                f.write(best_prompt_backup)
            print(f"💾 Interrupted prompt saved to: {interrupted_path}")


if __name__ == "__main__":
    main()

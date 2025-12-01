import argparse
import logging
import sys
import os

# Add the current directory to sys.path to ensure agent_orch can be imported
sys.path.append(os.getcwd())

from agent_orch.agent import EntityExtractionAgent

def main():
    parser = argparse.ArgumentParser(description="Agent-based Entity Extraction")
    parser.add_argument("--message", type=str, help="MMS message to process", required=False)
    parser.add_argument("--batch-file", type=str, help="Path to a file containing messages (one per line) for batch processing")
    parser.add_argument("--model", type=str, default="gen", help="LLM model name")
    args = parser.parse_args()

    # Validate arguments
    if not args.message and not args.batch_file:
        print("Error: Either --message or --batch-file must be provided.")
        return

    try:
        agent = EntityExtractionAgent(model_name=args.model)
        
        if args.batch_file:
            if not os.path.exists(args.batch_file):
                print(f"Error: Batch file not found at {args.batch_file}")
                return
                
            print(f"Processing batch file: {args.batch_file}")
            with open(args.batch_file, 'r', encoding='utf-8') as f:
                messages = [line.strip() for line in f if line.strip()]
            
            print(f"Found {len(messages)} messages.")
            for i, msg in enumerate(messages, 1):
                print(f"\n--- Processing Message {i}/{len(messages)} ---")
                print(f"Message: {msg[:50]}..." if len(msg) > 50 else f"Message: {msg}")
                result = agent.process_message(msg)
                print("Result:")
                print(result)
                
        elif args.message:
            print(f"Processing message: {args.message}")
            result = agent.process_message(args.message)
            print("\nResult:")
            print(result)
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()

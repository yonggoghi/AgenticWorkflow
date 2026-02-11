"""
CLI entry point for Korean MMS entity extraction via langextract.

Usage:
    python langextract/cli.py --message "광고 메시지" --model ax
    python langextract/cli.py --file messages.txt --model gpt --output results.jsonl
    python langextract/cli.py --message "메시지" --model opus --format json
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Path setup: mms_extractor_exp MUST be before AgenticWorkflow/ on sys.path
# because AgenticWorkflow/config.py would shadow mms_extractor_exp/config/ package.
_our_dir = str(Path(__file__).parent.parent)
_mms_exp_path = str(Path(__file__).parent.parent / "mms_extractor_exp")
for p in (_our_dir, _mms_exp_path):
    if p in sys.path:
        sys.path.remove(p)
sys.path.insert(0, _our_dir)       # position 1 after insert below
sys.path.insert(0, _mms_exp_path)  # position 0 (highest priority)

from langextract.core.data import AnnotatedDocument
from langextract.extract_mms import extract_mms_entities


def format_result_text(doc: AnnotatedDocument, message: str = "") -> str:
    """Format extraction result as human-readable text."""
    lines = []
    if message:
        lines.append(f"Message: {message[:100]}{'...' if len(message) > 100 else ''}")
        lines.append("")

    if not doc.extractions:
        lines.append("  (no extractions)")
        return "\n".join(lines)

    for ext in doc.extractions:
        attrs = ""
        if ext.attributes:
            attrs = f"  {ext.attributes}"
        lines.append(f"  [{ext.extraction_class}] {ext.extraction_text}{attrs}")

    return "\n".join(lines)


def format_result_json(doc: AnnotatedDocument) -> dict:
    """Format extraction result as JSON-serializable dict."""
    extractions = []
    for ext in (doc.extractions or []):
        entry = {
            "class": ext.extraction_class,
            "text": ext.extraction_text,
        }
        if ext.attributes:
            entry["attributes"] = ext.attributes
        extractions.append(entry)
    return {
        "text": doc.text,
        "extractions": extractions,
    }


def process_single(message: str, model_id: str, fmt: str, extraction_passes: int) -> str:
    """Process a single message and return formatted output."""
    start = time.time()
    result = extract_mms_entities(
        message,
        model_id=model_id,
        extraction_passes=extraction_passes,
    )
    elapsed = time.time() - start

    if fmt == "json":
        output = json.dumps(format_result_json(result), ensure_ascii=False, indent=2)
    else:
        output = format_result_text(result, message)
        output += f"\n  ({elapsed:.2f}s)"

    return output


def main():
    parser = argparse.ArgumentParser(
        description="Extract entities from Korean MMS ads using langextract + LLMFactory"
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--message", "-m", help="Single MMS message text")
    input_group.add_argument("--file", "-f", help="File with one message per line")

    parser.add_argument(
        "--model", default="ax",
        help="LLM model alias: ax, gpt, gen, cld, gem, opus (default: ax)"
    )
    parser.add_argument(
        "--format", dest="fmt", choices=["text", "json"], default="text",
        help="Output format (default: text)"
    )
    parser.add_argument(
        "--output", "-o", help="Output file path (default: stdout)"
    )
    parser.add_argument(
        "--passes", type=int, default=1,
        help="Number of extraction passes (default: 1)"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(name)s %(levelname)s: %(message)s")
    else:
        logging.basicConfig(level=logging.WARNING)

    outputs = []

    if args.message:
        output = process_single(args.message, args.model, args.fmt, args.passes)
        outputs.append(output)
    elif args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"Error: file not found: {args.file}", file=sys.stderr)
            sys.exit(1)
        messages = [line.strip() for line in file_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        print(f"Processing {len(messages)} messages with model={args.model}...", file=sys.stderr)
        for i, msg in enumerate(messages, 1):
            print(f"  [{i}/{len(messages)}] ...", file=sys.stderr, end="\r")
            output = process_single(msg, args.model, args.fmt, args.passes)
            outputs.append(output)
        print(file=sys.stderr)

    # Write output
    separator = "\n" if args.fmt == "json" else "\n---\n"
    final_output = separator.join(outputs) + "\n"

    if args.output:
        Path(args.output).write_text(final_output, encoding="utf-8")
        print(f"Wrote {len(outputs)} result(s) to {args.output}", file=sys.stderr)
    else:
        print(final_output)


if __name__ == "__main__":
    main()

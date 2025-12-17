#!/usr/bin/env python3
"""
Documentation Generator for MMS Extractor

This script automatically updates the EXECUTION_FLOW.md file to use absolute paths
with line numbers, making all function links clickable and navigable in VS Code.

Usage:
    python3 generate_docs.py
"""

import re
from pathlib import Path


def get_project_root():
    """Get the absolute path to the project root directory."""
    return Path(__file__).parent.absolute()


def update_execution_flow_doc():
    """
    Update EXECUTION_FLOW.md to use absolute file:// URIs with line numbers.
    """
    project_root = get_project_root()
    doc_path = project_root / "docs" / "EXECUTION_FLOW.md"
    
    if not doc_path.exists():
        print(f"❌ Error: {doc_path} not found!")
        return False
    
    print(f"📄 Reading {doc_path}...")
    content = doc_path.read_text(encoding='utf-8')
    
    # Convert relative paths to absolute file:// URIs
    def replace_relative_path(match):
        prefix = match.group(1)
        rel_path = match.group(2)
        
        clean_path = rel_path.replace('../', '')
        abs_path = project_root / clean_path
        file_uri = abs_path.as_uri()
        
        return f"{prefix}{file_uri}"
    
    # Replace patterns like (../apps/cli.py) and [text](../apps/cli.py)
    pattern = r'(\[.*?\]\(|[\(])(\.\./[^\)]+\.py)'
    updated_content = re.sub(pattern, replace_relative_path, content)
    
    # Line numbers mapping (start line only)
    line_numbers = {
        'apps/cli.py': {'main()': 'L52'},
        'core/mms_extractor.py': {
            'MMSExtractor.__init__()': 'L347',
            '_set_default_config()': 'L472',
            '_initialize_device()': 'L496',
            '_initialize_llm()': 'L507',
            '_initialize_embedding_model()': 'L546',
            '_initialize_kiwi()': 'L583',
            '_load_data()': 'L602',
            '_load_item_data()': 'L663',
            '_load_stopwords()': 'L698',
            '_register_items_in_kiwi()': 'L707',
            '_load_program_data()': 'L746',
            '_load_organization_data()': 'L786',
            'process_message_with_dag()': 'L1170',
            'MMSExtractor.process_message()': 'L1000',
        },
        'services/item_data_loader.py': {
            'ItemDataLoader.load_and_prepare_items()': 'L529',
            'load_raw_data()': 'L169',
            'normalize_columns()': 'L216',
            'filter_by_domain()': 'L247',
            'load_alias_rules()': 'L269',
            'expand_build_aliases()': 'L297',
            'create_bidirectional_aliases()': 'L334',
            'apply_cascading_alias_rules()': 'L353',
            'add_user_defined_entities()': 'L460',
            'add_domain_name_column()': 'L486',
            'filter_test_items()': 'L511',
        },
        'core/mms_workflow_steps.py': {
            'InputValidationStep.execute()': 'L206',
            'EntityExtractionStep.execute()': 'L266',
            'ProgramClassificationStep.execute()': 'L347',
            'ContextPreparationStep.execute()': 'L362',
            'LLMExtractionStep.execute()': 'L447',
            'ResponseParsingStep.execute()': 'L545',
            'ResultConstructionStep.execute()': 'L608',
            'ValidationStep.execute()': 'L680',
            'DAGExtractionStep.execute()': 'L715',
        },
        'apps/api.py': {
            'initialize_global_extractor()': 'L286',
            'extract_message()': 'L472',
            'get_configured_extractor()': 'L368',
        },
    }
    
    # Add line numbers to function links
    for file_path, functions in line_numbers.items():
        file_uri_base = (project_root / file_path).as_uri()
        
        for func_name, line_range in functions.items():
            # Simple string replacement for function links
            old_link = f"`{func_name}`]({file_uri_base})"
            new_link = f"`{func_name}`]({file_uri_base}#{line_range})"
            updated_content = updated_content.replace(old_link, new_link)
    
    # Write updated content
    print(f"✍️  Writing updated content to {doc_path}...")
    doc_path.write_text(updated_content, encoding='utf-8')
    
    print("✅ Successfully updated EXECUTION_FLOW.md!")
    print(f"📍 All links now use absolute paths: file://{project_root}/...")
    print("🔗 Function links include line numbers for precise navigation")
    
    return True


def main():
    """Main entry point."""
    print("=" * 60)
    print("MMS Extractor Documentation Generator")
    print("=" * 60)
    print()
    
    success = update_execution_flow_doc()
    
    print()
    if success:
        print("🎉 Documentation generation complete!")
        print()
        print("💡 Tip: Run this script whenever you move the project")
        print("   to a different directory to update all paths.")
    else:
        print("❌ Documentation generation failed!")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

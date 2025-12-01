#!/usr/bin/env python3
"""
Script to refactor mms_extractor.py by removing methods that have been extracted to Mixin classes
"""

import re

# Read the original file
with open('/Users/yongwook/workspace/AgenticWorkflow/mms_extractor_exp/mms_extractor.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Methods to remove (these are now in MMSExtractorDataMixin)
data_methods = [
    '_set_default_config',
    '_initialize_device',
    '_initialize_llm',
    '_initialize_embedding_model',
    '_initialize_multiple_llm_models',
    '_initialize_kiwi',
    '_load_data',
    '_load_and_prepare_item_data',
    '_get_database_connection',
    '_database_connection',
    '_load_program_from_database',
    '_load_stop_words',
    '_register_items_to_kiwi',
    '_load_program_data',
    '_load_organization_data',
    '_load_org_from_database',
]

# Methods to remove (these are now in MMSExtractorEntityMixin)
entity_methods = [
    'extract_entities_from_kiwi',
    'extract_entities_by_logic',
    '_calculate_combined_similarity',
    '_parse_entity_response',
    '_calculate_optimal_batch_size',
    'extract_entities_by_llm',
    '_match_entities_with_products',
    '_map_products_with_similarity',
    '_create_action_mapping',
]

all_methods_to_remove = data_methods + entity_methods

print(f"Total methods to remove: {len(all_methods_to_remove)}")
print(f"Data methods: {len(data_methods)}")
print(f"Entity methods: {len(entity_methods)}")

# Find the line numbers for each method
lines = content.split('\n')
method_ranges = {}

for i, line in enumerate(lines, 1):
    for method in all_methods_to_remove:
        # Look for method definitions
        if re.match(rf'^\s+def {method}\(', line) or re.match(rf'^\s+@\w+\s*$', line):
            # Check if next line is the method we're looking for
            if i < len(lines):
                next_line = lines[i] if i < len(lines) else ""
                if re.match(rf'^\s+def {method}\(', next_line):
                    method_ranges[method] = i - 1  # Include decorator
                    continue
            if re.match(rf'^\s+def {method}\(', line):
                method_ranges[method] = i

print(f"\nFound {len(method_ranges)} methods in the file")
for method, line_num in sorted(method_ranges.items(), key=lambda x: x[1]):
    print(f"  {method}: line {line_num}")

print(f"\nMethods not found: {set(all_methods_to_remove) - set(method_ranges.keys())}")

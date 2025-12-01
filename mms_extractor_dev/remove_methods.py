#!/usr/bin/env python3
"""
Script to remove extracted methods from mms_extractor.py
"""

import re

# Read the file
with open('/Users/yongwook/workspace/AgenticWorkflow/mms_extractor_exp/mms_extractor.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Methods to remove (now in Mixins)
methods_to_remove = [
    # Data methods
    '_set_default_config', '_initialize_device', '_initialize_llm',
    '_initialize_embedding_model', '_initialize_multiple_llm_models',
    '_initialize_kiwi', '_load_data', '_load_and_prepare_item_data',
    '_get_database_connection', '_database_connection',
    '_load_program_from_database', '_load_stop_words',
    '_register_items_to_kiwi', '_load_program_data',
    '_load_organization_data', '_load_org_from_database',
    # Entity methods
    'extract_entities_from_kiwi', 'extract_entities_by_logic',
    '_calculate_combined_similarity', '_parse_entity_response',
    '_calculate_optimal_batch_size', 'extract_entities_by_llm',
    '_match_entities_with_products', '_map_products_with_similarity',
    '_create_action_mapping',
]

def find_method_end(lines, start_idx):
    """Find the end of a method definition"""
    indent_level = None
    
    for i in range(start_idx, len(lines)):
        line = lines[i]
        stripped = line.lstrip()
        
        # Skip empty lines and comments at the start
        if not stripped or stripped.startswith('#'):
            continue
            
        # Get the indent level of the method
        if indent_level is None:
            if stripped.startswith('def ') or stripped.startswith('@'):
                indent_level = len(line) - len(stripped)
                continue
        
        # Check if we've reached the next method or class
        current_indent = len(line) - len(stripped)
        if indent_level is not None and current_indent <= indent_level and stripped and not stripped.startswith('#'):
            # Check if it's a new method/class definition or end of class
            if (stripped.startswith('def ') or 
                stripped.startswith('class ') or
                stripped.startswith('@') or
                (current_indent < indent_level)):
                return i
    
    return len(lines)

# Find and mark methods for removal
methods_found = {}
i = 0
while i < len(lines):
    line = lines[i]
    
    # Check if this line starts a method we want to remove
    for method in methods_to_remove:
        # Look for method definition or decorator before it
        if re.match(rf'^\s+def {method}\(', line):
            # Check if previous line is a decorator
            start_idx = i
            if i > 0 and lines[i-1].strip().startswith('@'):
                start_idx = i - 1
            
            end_idx = find_method_end(lines, i + 1)
            methods_found[method] = (start_idx, end_idx)
            print(f"Found {method}: lines {start_idx+1}-{end_idx}")
            break
    
    i += 1

print(f"\nTotal methods found: {len(methods_found)}")
print(f"Methods to remove: {len(methods_to_remove)}")
print(f"Missing: {set(methods_to_remove) - set(methods_found.keys())}")

# Create new file without the removed methods
new_lines = []
skip_until = -1

for i, line in enumerate(lines):
    # Check if we should skip this line
    if i < skip_until:
        continue
    
    # Check if this line starts a method to remove
    should_skip = False
    for method, (start, end) in methods_found.items():
        if i == start:
            skip_until = end
            should_skip = True
            print(f"Removing {method} (lines {start+1}-{end})")
            break
    
    if not should_skip:
        new_lines.append(line)

# Write the new file
with open('/Users/yongwook/workspace/AgenticWorkflow/mms_extractor_exp/mms_extractor.py', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print(f"\nOriginal lines: {len(lines)}")
print(f"New lines: {len(new_lines)}")
print(f"Removed lines: {len(lines) - len(new_lines)}")
print("\nRefactoring complete!")

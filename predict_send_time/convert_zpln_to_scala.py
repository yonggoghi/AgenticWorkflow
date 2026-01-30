#!/usr/bin/env python3
"""
Zeppelin Notebook to Scala File Converter

Usage:
    python convert_zpln_to_scala.py <input.zpln> <output.scala>
"""

import json
import sys
import re
from pathlib import Path


def clean_code(code):
    """Clean and format code from Zeppelin notebook."""
    if not code:
        return ""
    
    # Remove Windows line endings
    code = code.replace('\r\n', '\n').replace('\r', '\n')
    
    # Remove trailing whitespace from each line
    lines = [line.rstrip() for line in code.split('\n')]
    
    # Remove leading/trailing empty lines
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    
    return '\n'.join(lines)


def extract_title(paragraph):
    """Extract title from paragraph."""
    title = paragraph.get('title', '')
    if not title:
        # Try to extract from first line if it's a comment
        text = paragraph.get('text', '')
        if text:
            first_line = text.split('\n')[0].strip()
            if first_line.startswith('//'):
                title = first_line[2:].strip()
    return title


def convert_zpln_to_scala(zpln_path, output_path):
    """Convert Zeppelin notebook (.zpln) to Scala file (.scala)."""
    
    print(f"Reading Zeppelin notebook: {zpln_path}")
    
    # Read Zeppelin notebook (JSON format)
    # Use utf-8-sig to handle BOM if present
    with open(zpln_path, 'r', encoding='utf-8-sig') as f:
        notebook = json.load(f)
    
    paragraphs = notebook.get('paragraphs', [])
    
    if not paragraphs:
        print("Warning: No paragraphs found in notebook")
        return
    
    print(f"Found {len(paragraphs)} paragraphs")
    
    # Collect Scala code from each paragraph
    scala_code_blocks = []
    
    for idx, paragraph in enumerate(paragraphs, 1):
        title = extract_title(paragraph)
        text = paragraph.get('text', '')
        
        if not text or not text.strip():
            continue
        
        # Skip if it's a markdown paragraph (starts with %md)
        if text.strip().startswith('%md'):
            continue
        
        # Skip if it's a shell command paragraph
        if text.strip().startswith('%sh'):
            continue
        
        # Skip if it's Python paragraph
        if text.strip().startswith('%pyspark') or text.strip().startswith('%python'):
            continue
        
        # Clean the code
        code = clean_code(text)
        
        if not code:
            continue
        
        # Create a section with title
        section = []
        section.append('')
        section.append('// ' + '=' * 77)
        
        if title:
            section.append(f'// Paragraph {idx}: {title}')
        else:
            section.append(f'// Paragraph {idx}')
        
        section.append('// ' + '=' * 77)
        section.append('')
        section.append(code)
        
        scala_code_blocks.append('\n'.join(section))
    
    # Combine all code blocks
    output_code = '\n\n'.join(scala_code_blocks)
    
    # Add file header
    header = f"""// =============================================================================
// Converted from Zeppelin Notebook
// =============================================================================
// Original file: {Path(zpln_path).name}
// Converted on: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
// Total paragraphs: {len(scala_code_blocks)}
// =============================================================================

"""
    
    final_code = header + output_code
    
    # Write to output file
    print(f"Writing Scala code to: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(final_code)
    
    print(f"✅ Conversion complete!")
    print(f"   - Input paragraphs: {len(paragraphs)}")
    print(f"   - Output code blocks: {len(scala_code_blocks)}")
    print(f"   - Output file: {output_path}")


def main():
    if len(sys.argv) != 3:
        print("Usage: python convert_zpln_to_scala.py <input.zpln> <output.scala>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    if not Path(input_file).exists():
        print(f"Error: Input file not found: {input_file}")
        sys.exit(1)
    
    if not input_file.endswith('.zpln'):
        print(f"Warning: Input file does not have .zpln extension: {input_file}")
    
    if not output_file.endswith('.scala'):
        print(f"Warning: Output file does not have .scala extension: {output_file}")
    
    try:
        convert_zpln_to_scala(input_file, output_file)
    except Exception as e:
        print(f"❌ Error during conversion: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

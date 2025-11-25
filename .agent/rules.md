# Project-Specific AI Rules

## Jupyter Notebook (.ipynb) Handling

### Default Behavior
When working with `.ipynb` files:
- **DO NOT** directly edit the notebook files
- **ALWAYS** provide code snippets that the user can copy and paste
- Only show the modified code, not the entire file
- Format code snippets with proper syntax highlighting

### Rationale
- Jupyter Notebook files are JSON-formatted and complex to edit programmatically
- Direct editing risks corrupting the notebook structure
- Users prefer to manually paste code into specific cells they're working on

### Example Response Format
When asked to modify a function in a notebook:

```python
# Replace the function in your notebook with this:

def your_function(args):
    # modified code here
    pass
```

### Exceptions
Direct editing is acceptable only when:
- User explicitly requests automated editing with clear confirmation
- Working with notebook metadata (tags, cell types) via programmatic tools

## Cell Reference Methods

### Preferred: Cell Tags
Use Jupyter cell tags for programmatic reference:
- Easy to search with `grep`
- Structured metadata
- No code pollution

### Alternative: Markdown Headers + Tags
For human readability, combine:
1. Markdown cell with descriptive header above the code cell
2. Tag on the code cell for programmatic access

Example:
```markdown
# 📌 Function Name
Brief description of what this cell does
```
Then add tag `function_name` to the following code cell.

## Python Execution Environment

### Virtual Environment Usage
This project uses a Python virtual environment located at `/Users/yongwook/workspace/AgenticWorkflow/venv`.

**ALWAYS** activate the virtual environment before running Python commands:

```bash
cd /Users/yongwook/workspace/AgenticWorkflow && source venv/bin/activate && cd mms_extractor_exp && python your_script.py
```

### Command Execution Pattern
When running Python scripts or commands:
1. Change to project root directory
2. Activate virtual environment with `source venv/bin/activate`
3. Change to appropriate subdirectory if needed
4. Execute the Python command

### Example Commands

**Running a Python script:**
```bash
cd /Users/yongwook/workspace/AgenticWorkflow && source venv/bin/activate && cd mms_extractor_exp && python mms_extractor.py
```

**Running pytest:**
```bash
cd /Users/yongwook/workspace/AgenticWorkflow && source venv/bin/activate && pytest
```

**Installing packages:**
```bash
cd /Users/yongwook/workspace/AgenticWorkflow && source venv/bin/activate && pip install package_name
```

### Why This Matters
- Ensures correct Python dependencies are available
- Prevents "ModuleNotFoundError" issues
- Maintains consistent execution environment
- Avoids conflicts with system Python packages


# tool-schema-validator

Validate LLM tool schemas for Anthropic, OpenAI, and Gemini providers.

## Install

```
pip install tool-schema-validator
```

## Usage

```python
from tool_schema_validator import SchemaValidator

v = SchemaValidator()
result = v.validate_anthropic({
    "name": "search",
    "description": "Search the web",
    "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}}
})
print(result.valid, result.errors)

# Validate against all providers at once
results = v.validate_all(schema)
print(v.all_valid(schema))
```

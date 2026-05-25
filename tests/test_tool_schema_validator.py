"""Tests for tool-schema-validator."""
import pytest
from tool_schema_validator import SchemaValidator, ValidationResult, ValidationIssue

VALID_ANTHROPIC = {
    "name": "search_web",
    "description": "Search the web",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
        },
        "required": ["query"],
    },
}

VALID_OPENAI = {
    "type": "function",
    "function": {
        "name": "search_web",
        "description": "Search the web",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
}

VALID_GEMINI = {
    "name": "search_web",
    "description": "Search the web",
    "parameters": {
        "type": "object",
        "properties": {"query": {"type": "string"}},
    },
}


def test_validate_anthropic_valid():
    v = SchemaValidator()
    result = v.validate_anthropic(VALID_ANTHROPIC)
    assert result.valid is True
    assert result.provider == "anthropic"


def test_validate_anthropic_missing_name():
    v = SchemaValidator()
    schema = {**VALID_ANTHROPIC}
    del schema["name"]
    result = v.validate_anthropic(schema)
    assert result.valid is False
    assert any("name" in i.field for i in result.errors)


def test_validate_anthropic_missing_input_schema():
    v = SchemaValidator()
    schema = {"name": "search", "description": "test"}
    result = v.validate_anthropic(schema)
    assert result.valid is False


def test_validate_anthropic_bad_input_schema_type():
    v = SchemaValidator()
    schema = {**VALID_ANTHROPIC, "input_schema": {"type": "array"}}
    result = v.validate_anthropic(schema)
    assert result.valid is False


def test_validate_anthropic_missing_description_warning():
    v = SchemaValidator()
    schema = {k: v for k, v in VALID_ANTHROPIC.items() if k != "description"}
    result = v.validate_anthropic(schema)
    assert any(i.severity == "warning" for i in result.issues)


def test_validate_openai_valid():
    v = SchemaValidator()
    result = v.validate_openai(VALID_OPENAI)
    assert result.valid is True
    assert result.provider == "openai"


def test_validate_openai_wrong_type():
    v = SchemaValidator()
    schema = {**VALID_OPENAI, "type": "tool"}
    result = v.validate_openai(schema)
    assert result.valid is False


def test_validate_openai_missing_function_name():
    v = SchemaValidator()
    schema = {"type": "function", "function": {"description": "test"}}
    result = v.validate_openai(schema)
    assert result.valid is False


def test_validate_gemini_valid():
    v = SchemaValidator()
    result = v.validate_gemini(VALID_GEMINI)
    assert result.valid is True
    assert result.provider == "gemini"


def test_validate_gemini_missing_name():
    v = SchemaValidator()
    schema = {k: v for k, v in VALID_GEMINI.items() if k != "name"}
    result = v.validate_gemini(schema)
    assert result.valid is False


def test_validate_all_anthropic():
    v = SchemaValidator()
    results = v.validate_all([VALID_ANTHROPIC, VALID_ANTHROPIC], provider="anthropic")
    assert len(results) == 2
    assert all(r.valid for r in results)


def test_all_valid_true():
    v = SchemaValidator()
    assert v.all_valid([VALID_ANTHROPIC], provider="anthropic") is True


def test_all_valid_false():
    v = SchemaValidator()
    bad = {"name": "", "input_schema": {"type": "object"}}
    assert v.all_valid([bad], provider="anthropic") is False


def test_validation_result_bool_true():
    r = ValidationResult(schema={}, issues=[], provider="test")
    assert bool(r) is True


def test_validation_result_bool_false():
    r = ValidationResult(schema={}, issues=[ValidationIssue("x", "bad")], provider="test")
    assert bool(r) is False


def test_validation_issue_str():
    i = ValidationIssue("name", "missing field")
    assert "name" in str(i)
    assert "missing" in str(i)


def test_errors_and_warnings_split():
    issues = [
        ValidationIssue("a", "bad", severity="error"),
        ValidationIssue("b", "warn", severity="warning"),
    ]
    r = ValidationResult(schema={}, issues=issues)
    assert len(r.errors) == 1
    assert len(r.warnings) == 1


def test_validate_anthropic_empty_name():
    v = SchemaValidator()
    schema = {**VALID_ANTHROPIC, "name": ""}
    result = v.validate_anthropic(schema)
    assert result.valid is False

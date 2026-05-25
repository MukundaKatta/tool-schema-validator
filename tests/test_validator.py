"""Tests for tool_schema_validator."""

import pytest

from tool_schema_validator import (
    SchemaValidator,
    ValidationResult,
    validate_anthropic,
    validate_gemini,
    validate_openai,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VALID_ANTHROPIC = {
    "name": "get_weather",
    "description": "Get current weather for a location",
    "input_schema": {
        "type": "object",
        "properties": {
            "location": {"type": "string"},
        },
    },
}

VALID_OPENAI = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string"},
            },
        },
    },
}

VALID_GEMINI = {
    "name": "get_weather",
    "description": "Get current weather for a location",
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "location": {"type": "STRING"},
        },
    },
}


# ---------------------------------------------------------------------------
# validate_anthropic
# ---------------------------------------------------------------------------


class TestValidateAnthropic:
    def test_valid_schema_passes(self):
        result = validate_anthropic(VALID_ANTHROPIC)
        assert result.valid is True
        assert result.errors == []

    def test_missing_name_error(self):
        schema = {**VALID_ANTHROPIC}
        del schema["name"]
        result = validate_anthropic(schema)
        assert result.valid is False
        assert any("name" in e for e in result.errors)

    def test_name_too_long_error(self):
        schema = {**VALID_ANTHROPIC, "name": "a" * 65}
        result = validate_anthropic(schema)
        assert result.valid is False
        assert any("64" in e for e in result.errors)

    def test_name_with_spaces_error(self):
        schema = {**VALID_ANTHROPIC, "name": "get weather"}
        result = validate_anthropic(schema)
        assert result.valid is False
        assert any("space" in e.lower() for e in result.errors)

    def test_missing_description_is_warning_not_error(self):
        schema = {**VALID_ANTHROPIC}
        del schema["description"]
        result = validate_anthropic(schema)
        # Warnings, not errors
        assert result.valid is True
        assert any("description" in w for w in result.warnings)
        assert not any("description" in e for e in result.errors)

    def test_empty_description_is_warning(self):
        schema = {**VALID_ANTHROPIC, "description": ""}
        result = validate_anthropic(schema)
        assert result.valid is True
        assert any("description" in w for w in result.warnings)

    def test_missing_input_schema_error(self):
        schema = {k: v for k, v in VALID_ANTHROPIC.items() if k != "input_schema"}
        result = validate_anthropic(schema)
        assert result.valid is False
        assert any("input_schema" in e for e in result.errors)

    def test_input_schema_wrong_type_error(self):
        schema = {
            **VALID_ANTHROPIC,
            "input_schema": {"type": "string", "properties": {}},
        }
        result = validate_anthropic(schema)
        assert result.valid is False
        assert any("object" in e for e in result.errors)

    def test_anyof_in_input_schema_error(self):
        schema = {
            **VALID_ANTHROPIC,
            "input_schema": {
                "type": "object",
                "properties": {},
                "anyOf": [{"type": "string"}],
            },
        }
        result = validate_anthropic(schema)
        assert result.valid is False
        assert any("anyOf" in e for e in result.errors)

    def test_oneof_in_input_schema_error(self):
        schema = {
            **VALID_ANTHROPIC,
            "input_schema": {
                "type": "object",
                "properties": {},
                "oneOf": [{"type": "string"}],
            },
        }
        result = validate_anthropic(schema)
        assert result.valid is False
        assert any("oneOf" in e for e in result.errors)

    def test_missing_properties_is_warning(self):
        schema = {
            **VALID_ANTHROPIC,
            "input_schema": {"type": "object"},
        }
        result = validate_anthropic(schema)
        assert result.valid is True
        assert any("properties" in w for w in result.warnings)

    def test_name_exactly_64_chars_passes(self):
        schema = {**VALID_ANTHROPIC, "name": "a" * 64}
        result = validate_anthropic(schema)
        assert result.valid is True

    def test_not_a_dict_error(self):
        result = validate_anthropic("not a dict")  # type: ignore[arg-type]
        assert result.valid is False
        assert any("dict" in e for e in result.errors)


# ---------------------------------------------------------------------------
# validate_openai
# ---------------------------------------------------------------------------


class TestValidateOpenAI:
    def test_valid_schema_passes(self):
        result = validate_openai(VALID_OPENAI)
        assert result.valid is True
        assert result.errors == []

    def test_missing_type_function_error(self):
        schema = {**VALID_OPENAI}
        del schema["type"]
        result = validate_openai(schema)
        assert result.valid is False
        assert any("type" in e for e in result.errors)

    def test_wrong_type_value_error(self):
        schema = {**VALID_OPENAI, "type": "tool"}
        result = validate_openai(schema)
        assert result.valid is False
        assert any("function" in e for e in result.errors)

    def test_missing_function_name_error(self):
        schema = {
            **VALID_OPENAI,
            "function": {k: v for k, v in VALID_OPENAI["function"].items() if k != "name"},
        }
        result = validate_openai(schema)
        assert result.valid is False
        assert any("name" in e for e in result.errors)

    def test_missing_function_parameters_error(self):
        fn = {k: v for k, v in VALID_OPENAI["function"].items() if k != "parameters"}
        schema = {**VALID_OPENAI, "function": fn}
        result = validate_openai(schema)
        assert result.valid is False
        assert any("parameters" in e for e in result.errors)

    def test_parameters_missing_type_object_error(self):
        schema = {
            **VALID_OPENAI,
            "function": {
                **VALID_OPENAI["function"],
                "parameters": {"type": "string", "properties": {}},
            },
        }
        result = validate_openai(schema)
        assert result.valid is False
        assert any("object" in e for e in result.errors)

    def test_missing_description_is_warning(self):
        fn = {k: v for k, v in VALID_OPENAI["function"].items() if k != "description"}
        schema = {**VALID_OPENAI, "function": fn}
        result = validate_openai(schema)
        assert result.valid is True
        assert any("description" in w for w in result.warnings)

    def test_missing_properties_is_warning(self):
        schema = {
            **VALID_OPENAI,
            "function": {
                **VALID_OPENAI["function"],
                "parameters": {"type": "object"},
            },
        }
        result = validate_openai(schema)
        assert result.valid is True
        assert any("properties" in w for w in result.warnings)

    def test_not_a_dict_error(self):
        result = validate_openai(42)  # type: ignore[arg-type]
        assert result.valid is False


# ---------------------------------------------------------------------------
# validate_gemini
# ---------------------------------------------------------------------------


class TestValidateGemini:
    def test_valid_schema_passes(self):
        result = validate_gemini(VALID_GEMINI)
        assert result.valid is True
        assert result.errors == []

    def test_missing_name_error(self):
        schema = {k: v for k, v in VALID_GEMINI.items() if k != "name"}
        result = validate_gemini(schema)
        assert result.valid is False
        assert any("name" in e for e in result.errors)

    def test_uppercase_object_passes(self):
        # Explicit check: "OBJECT" is the correct Gemini format
        result = validate_gemini(VALID_GEMINI)
        assert result.valid is True

    def test_lowercase_object_is_error(self):
        # Gemini requires uppercase "OBJECT", not lowercase "object"
        schema = {
            **VALID_GEMINI,
            "parameters": {**VALID_GEMINI["parameters"], "type": "object"},
        }
        result = validate_gemini(schema)
        assert result.valid is False
        assert any("OBJECT" in e or "uppercase" in e.lower() for e in result.errors)

    def test_missing_description_is_warning(self):
        schema = {k: v for k, v in VALID_GEMINI.items() if k != "description"}
        result = validate_gemini(schema)
        assert result.valid is True
        assert any("description" in w for w in result.warnings)

    def test_missing_parameters_error(self):
        schema = {k: v for k, v in VALID_GEMINI.items() if k != "parameters"}
        result = validate_gemini(schema)
        assert result.valid is False
        assert any("parameters" in e for e in result.errors)

    def test_missing_properties_is_warning(self):
        schema = {
            **VALID_GEMINI,
            "parameters": {"type": "OBJECT"},
        }
        result = validate_gemini(schema)
        assert result.valid is True
        assert any("properties" in w for w in result.warnings)

    def test_not_a_dict_error(self):
        result = validate_gemini(None)  # type: ignore[arg-type]
        assert result.valid is False


# ---------------------------------------------------------------------------
# SchemaValidator class
# ---------------------------------------------------------------------------


class TestSchemaValidator:
    def setup_method(self):
        self.v = SchemaValidator()

    def test_validate_dispatches_to_anthropic(self):
        result = self.v.validate(VALID_ANTHROPIC, "anthropic")
        assert result.valid is True

    def test_validate_dispatches_to_openai(self):
        result = self.v.validate(VALID_OPENAI, "openai")
        assert result.valid is True

    def test_validate_dispatches_to_gemini(self):
        result = self.v.validate(VALID_GEMINI, "gemini")
        assert result.valid is True

    def test_validate_raises_on_unknown_provider(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            self.v.validate({}, "fakeai")

    def test_validate_all_runs_all_providers(self):
        results = self.v.validate_all(VALID_ANTHROPIC)
        assert set(results.keys()) == {"anthropic", "openai", "gemini"}
        assert all(isinstance(r, ValidationResult) for r in results.values())

    def test_register_custom_provider_works(self):
        def always_ok(schema: dict) -> ValidationResult:
            return ValidationResult(valid=True)

        self.v.register_provider("myai", always_ok)
        result = self.v.validate({}, "myai")
        assert result.valid is True

    def test_register_provider_appears_in_validate_all(self):
        self.v.register_provider("myai", lambda s: ValidationResult(valid=True))
        results = self.v.validate_all({})
        assert "myai" in results

    def test_providers_returns_sorted_list(self):
        names = self.v.providers()
        assert names == sorted(names)
        assert "anthropic" in names
        assert "openai" in names
        assert "gemini" in names


# ---------------------------------------------------------------------------
# ValidationResult semantics
# ---------------------------------------------------------------------------


class TestValidationResult:
    def test_valid_false_when_errors_present(self):
        r = ValidationResult(valid=False, errors=["bad thing"])
        assert r.valid is False

    def test_valid_true_when_only_warnings(self):
        r = ValidationResult(valid=True, warnings=["consider adding description"])
        assert r.valid is True
        assert r.errors == []

    def test_defaults_empty_lists(self):
        r = ValidationResult(valid=True)
        assert r.errors == []
        assert r.warnings == []

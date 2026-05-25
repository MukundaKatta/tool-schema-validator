"""
tool-schema-validator: Validate tool schemas against Anthropic / OpenAI / Gemini specs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ValidationIssue:
    field: str
    reason: str
    severity: str = "error"  # "error" | "warning"

    def __str__(self) -> str:
        return f"[{self.severity}] {self.field}: {self.reason}"


@dataclass
class ValidationResult:
    schema: dict[str, Any]
    issues: list[ValidationIssue] = field(default_factory=list)
    provider: str = ""

    @property
    def valid(self) -> bool:
        return not any(i.severity == "error" for i in self.issues)

    @property
    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "warning"]

    def __bool__(self) -> bool:
        return self.valid


def _check_common(schema: dict[str, Any]) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    if "name" not in schema:
        issues.append(ValidationIssue("name", "Missing required field 'name'"))
    elif not isinstance(schema["name"], str) or not schema["name"]:
        issues.append(ValidationIssue("name", "Field 'name' must be a non-empty string"))
    if "description" not in schema:
        issues.append(ValidationIssue("description", "Missing 'description'", severity="warning"))
    return issues


def _check_input_schema(input_schema: Any, prefix: str = "input_schema") -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    if not isinstance(input_schema, dict):
        issues.append(ValidationIssue(prefix, "Must be a JSON object (dict)"))
        return issues
    if input_schema.get("type") != "object":
        issues.append(ValidationIssue(f"{prefix}.type", "Must be 'object'"))
    if "properties" not in input_schema:
        issues.append(ValidationIssue(f"{prefix}.properties", "Missing 'properties' field"))
    elif not isinstance(input_schema["properties"], dict):
        issues.append(ValidationIssue(f"{prefix}.properties", "Must be a dict"))
    else:
        for param_name, param_schema in input_schema["properties"].items():
            if not isinstance(param_schema, dict):
                issues.append(ValidationIssue(f"{prefix}.properties.{param_name}", "Parameter schema must be a dict"))
                continue
            if "type" not in param_schema:
                issues.append(ValidationIssue(f"{prefix}.properties.{param_name}.type",
                                               "Missing 'type' field", severity="warning"))
    return issues


class SchemaValidator:
    """
    Validate tool schemas against provider specs.

    Usage::

        validator = SchemaValidator()
        result = validator.validate_anthropic({"name": "search", "description": "...",
                                               "input_schema": {"type": "object", "properties": {...}}})
        assert result.valid
        results = validator.validate_all([schema1, schema2])
    """

    def validate_anthropic(self, schema: dict[str, Any]) -> ValidationResult:
        issues = _check_common(schema)
        if "input_schema" not in schema:
            issues.append(ValidationIssue("input_schema", "Missing required field 'input_schema'"))
        else:
            issues.extend(_check_input_schema(schema["input_schema"]))
        return ValidationResult(schema=schema, issues=issues, provider="anthropic")

    def validate_openai(self, schema: dict[str, Any]) -> ValidationResult:
        issues: list[ValidationIssue] = []
        if schema.get("type") != "function":
            issues.append(ValidationIssue("type", "Must be 'function' for OpenAI format"))
        fn = schema.get("function", {})
        if not isinstance(fn, dict):
            issues.append(ValidationIssue("function", "Must be a dict"))
        else:
            issues.extend(_check_common(fn))
            if "parameters" in fn:
                issues.extend(_check_input_schema(fn["parameters"], prefix="function.parameters"))
        return ValidationResult(schema=schema, issues=issues, provider="openai")

    def validate_gemini(self, schema: dict[str, Any]) -> ValidationResult:
        issues = _check_common(schema)
        if "parameters" in schema:
            issues.extend(_check_input_schema(schema["parameters"]))
        return ValidationResult(schema=schema, issues=issues, provider="gemini")

    def validate_all(self, schemas: list[dict[str, Any]], provider: str = "anthropic") -> list[ValidationResult]:
        dispatch = {
            "anthropic": self.validate_anthropic,
            "openai": self.validate_openai,
            "gemini": self.validate_gemini,
        }
        fn = dispatch.get(provider, self.validate_anthropic)
        return [fn(s) for s in schemas]

    def all_valid(self, schemas: list[dict[str, Any]], provider: str = "anthropic") -> bool:
        return all(r.valid for r in self.validate_all(schemas, provider=provider))


__all__ = ["SchemaValidator", "ValidationResult", "ValidationIssue"]

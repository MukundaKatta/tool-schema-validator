"""tool-schema-validator: validate tool JSON schemas before sending to LLMs."""

from dataclasses import dataclass, field


@dataclass
class ValidationResult:
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def validate_anthropic(schema: dict) -> ValidationResult:
    """
    Validate an Anthropic tool schema.

    Rules:
    - Must be a dict
    - Must have "name": str, non-empty, max 64 chars, no spaces
    - Must have "description": str, non-empty (warning if missing/empty)
    - Must have "input_schema": dict with "type": "object"
    - input_schema should have "properties": dict (warning if missing)
    - input_schema must NOT have anyOf/oneOf/allOf at top level
    """
    errors: list[str] = []
    warnings: list[str] = []

    if not isinstance(schema, dict):
        return ValidationResult(valid=False, errors=["schema must be a dict"])

    # name checks
    name = schema.get("name")
    if not isinstance(name, str) or not name:
        errors.append("'name' is required and must be a non-empty string")
    else:
        if " " in name:
            errors.append("'name' must not contain spaces")
        if len(name) > 64:
            errors.append("'name' must be 64 characters or fewer")

    # description checks
    desc = schema.get("description")
    if not isinstance(desc, str) or not desc:
        warnings.append("'description' is recommended and should be a non-empty string")

    # input_schema checks
    input_schema = schema.get("input_schema")
    if not isinstance(input_schema, dict):
        errors.append("'input_schema' is required and must be a dict")
    else:
        if input_schema.get("type") != "object":
            errors.append("'input_schema.type' must be \"object\"")
        for banned in ("anyOf", "oneOf", "allOf"):
            if banned in input_schema:
                errors.append(f"'input_schema' must not contain '{banned}'")
        if not isinstance(input_schema.get("properties"), dict):
            warnings.append("'input_schema.properties' is recommended and should be a dict")

    valid = len(errors) == 0
    return ValidationResult(valid=valid, errors=errors, warnings=warnings)


def validate_openai(schema: dict) -> ValidationResult:
    """
    Validate an OpenAI tool schema.

    Rules:
    - Must be a dict
    - Must have "type": "function"
    - Must have "function": dict
    - function must have "name": str, non-empty
    - function must have "description": str (warning if missing)
    - function must have "parameters": dict with "type": "object"
    - parameters should have "properties": dict (warning if missing)
    """
    errors: list[str] = []
    warnings: list[str] = []

    if not isinstance(schema, dict):
        return ValidationResult(valid=False, errors=["schema must be a dict"])

    # type check
    if schema.get("type") != "function":
        errors.append("'type' must be \"function\"")

    # function block
    fn = schema.get("function")
    if not isinstance(fn, dict):
        errors.append("'function' is required and must be a dict")
    else:
        name = fn.get("name")
        if not isinstance(name, str) or not name:
            errors.append("'function.name' is required and must be a non-empty string")

        desc = fn.get("description")
        if not isinstance(desc, str) or not desc:
            warnings.append(
                "'function.description' is recommended and should be a non-empty string"
            )

        params = fn.get("parameters")
        if not isinstance(params, dict):
            errors.append("'function.parameters' is required and must be a dict")
        else:
            if params.get("type") != "object":
                errors.append("'function.parameters.type' must be \"object\"")
            if not isinstance(params.get("properties"), dict):
                warnings.append(
                    "'function.parameters.properties' is recommended and should be a dict"
                )

    valid = len(errors) == 0
    return ValidationResult(valid=valid, errors=errors, warnings=warnings)


def validate_gemini(schema: dict) -> ValidationResult:
    """
    Validate a Gemini function declaration schema.

    Rules:
    - Must be a dict
    - Must have "name": str, non-empty
    - Must have "description": str (warning if missing)
    - Must have "parameters": dict with "type": "OBJECT" (Gemini uses uppercase)
    - parameters should have "properties": dict (warning if missing)
    """
    errors: list[str] = []
    warnings: list[str] = []

    if not isinstance(schema, dict):
        return ValidationResult(valid=False, errors=["schema must be a dict"])

    # name check
    name = schema.get("name")
    if not isinstance(name, str) or not name:
        errors.append("'name' is required and must be a non-empty string")

    # description check
    desc = schema.get("description")
    if not isinstance(desc, str) or not desc:
        warnings.append("'description' is recommended and should be a non-empty string")

    # parameters check
    params = schema.get("parameters")
    if not isinstance(params, dict):
        errors.append("'parameters' is required and must be a dict")
    else:
        # Gemini requires uppercase "OBJECT", not lowercase "object"
        if params.get("type") != "OBJECT":
            errors.append("'parameters.type' must be \"OBJECT\" (Gemini uses uppercase)")
        if not isinstance(params.get("properties"), dict):
            warnings.append("'parameters.properties' is recommended and should be a dict")

    valid = len(errors) == 0
    return ValidationResult(valid=valid, errors=errors, warnings=warnings)


class SchemaValidator:
    """Multi-provider tool schema validator."""

    def __init__(self) -> None:
        self._providers: dict[str, object] = {
            "anthropic": validate_anthropic,
            "openai": validate_openai,
            "gemini": validate_gemini,
        }

    def register_provider(self, name: str, validate_fn) -> None:
        """Register a custom provider. validate_fn(schema: dict) -> ValidationResult."""
        self._providers[name] = validate_fn

    def validate(self, schema: dict, provider: str) -> ValidationResult:
        """Validate schema against a named provider. Raises ValueError if provider unknown."""
        if provider not in self._providers:
            registered = sorted(self._providers)
            raise ValueError(
                f"Unknown provider {provider!r}. Registered providers: {registered}"
            )
        fn = self._providers[provider]
        return fn(schema)  # type: ignore[operator]

    def validate_all(self, schema: dict) -> dict[str, ValidationResult]:
        """Run all registered providers and return {provider_name: ValidationResult}."""
        return {name: fn(schema) for name, fn in self._providers.items()}  # type: ignore[operator]

    def providers(self) -> list[str]:
        """Return sorted list of registered provider names."""
        return sorted(self._providers)

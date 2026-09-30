from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


REQUIRED_TOP_LEVEL = {
    "specification",
    "physics",
    "normalization",
    "geometry",
    "initial_states",
    "observations",
    "uncertainty",
    "policies",
    "solutions",
    "assessment",
    "provenance",
}


def load_schema(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_minimal_record(record: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    missing = sorted(REQUIRED_TOP_LEVEL - set(record))
    if missing:
        errors.append("missing_top_level:" + ",".join(missing))
    if record.get("specification", {}).get("id") != "RMO-R6A-SPEC-1.0.0":
        errors.append("wrong_specification_id")
    physics = record.get("physics", {})
    if physics.get("model") != "ideal_mhd_1d":
        errors.append("wrong_physics_model")
    if physics.get("admissibility_policy") not in {"REGULAR_EVOLUTIONARY_1.0", "ENUMERATE_NONREGULAR_1.0"}:
        errors.append("invalid_policy")
    return errors


def validate_record(record: dict[str, Any], schema: dict[str, Any]) -> list[str]:
    """Validate the frozen RMO schema subset without a third-party runtime.

    The frozen schema uses only local ``$ref``, object/array/scalar types,
    required, properties, additionalProperties, enum/const and numeric/string
    bounds. Unsupported keywords fail closed instead of being ignored.
    """

    errors: list[str] = []
    supported = {
        "$schema", "$id", "$ref", "$defs", "title", "type", "required",
        "properties", "additionalProperties", "items", "minItems", "maxItems",
        "uniqueItems", "minimum", "exclusiveMinimum", "enum", "const",
    }

    def resolve(ref: str) -> dict[str, Any]:
        if not ref.startswith("#/"):
            raise ValueError(f"external_ref_not_supported:{ref}")
        node: Any = schema
        for part in ref[2:].split("/"):
            node = node[part.replace("~1", "/").replace("~0", "~")]
        return node

    def matches_type(value: Any, kind: str) -> bool:
        if kind == "object":
            return isinstance(value, dict)
        if kind == "array":
            return isinstance(value, list)
        if kind == "string":
            return isinstance(value, str)
        if kind == "number":
            return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))
        if kind == "integer":
            return isinstance(value, int) and not isinstance(value, bool)
        if kind == "null":
            return value is None
        return False

    def visit(value: Any, node: dict[str, Any], path: str) -> None:
        unknown = set(node) - supported
        if unknown:
            errors.append(f"{path}:unsupported_schema_keyword:{','.join(sorted(unknown))}")
            return
        if "$ref" in node:
            visit(value, resolve(node["$ref"]), path)
            return
        if "const" in node and value != node["const"]:
            errors.append(f"{path}:const")
        if "enum" in node and value not in node["enum"]:
            errors.append(f"{path}:enum")
        declared = node.get("type")
        if declared is not None:
            kinds = [declared] if isinstance(declared, str) else declared
            if not any(matches_type(value, kind) for kind in kinds):
                errors.append(f"{path}:type")
                return
        if isinstance(value, dict):
            props = node.get("properties", {})
            for key in node.get("required", []):
                if key not in value:
                    errors.append(f"{path}.{key}:required")
            if node.get("additionalProperties") is False:
                for key in sorted(set(value) - set(props)):
                    errors.append(f"{path}.{key}:additional_property")
            for key, child in props.items():
                if key in value:
                    visit(value[key], child, f"{path}.{key}")
        elif isinstance(value, list):
            if len(value) < node.get("minItems", 0):
                errors.append(f"{path}:minItems")
            if "maxItems" in node and len(value) > node["maxItems"]:
                errors.append(f"{path}:maxItems")
            if node.get("uniqueItems"):
                encoded = [json.dumps(item, sort_keys=True) for item in value]
                if len(encoded) != len(set(encoded)):
                    errors.append(f"{path}:uniqueItems")
            if "items" in node:
                for index, item in enumerate(value):
                    visit(item, node["items"], f"{path}[{index}]")
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            if "minimum" in node and value < node["minimum"]:
                errors.append(f"{path}:minimum")
            if "exclusiveMinimum" in node and value <= node["exclusiveMinimum"]:
                errors.append(f"{path}:exclusiveMinimum")

    visit(record, schema, "$")
    return errors

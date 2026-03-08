#!/usr/bin/env python3
"""
Parse OpenAPI spec and generate FastAPI routers for each root path.
"""

import yaml
import re
from pathlib import Path
from collections import defaultdict


def load_openapi_spec(spec_path: str) -> dict:
    """Load OpenAPI YAML spec"""
    with open(spec_path, "r") as f:
        return yaml.safe_load(f)


def extract_root_path(path: str) -> str:
    """Extract root path segment. e.g., /chat/completions -> chat"""
    parts = path.strip("/").split("/")
    return parts[0] if parts else "root"


def pascal_case_to_snake(name: str) -> str:
    """Convert Pascal case to snake_case"""
    name = re.sub(r"([A-Z])", r"_\1", name).lower()
    return name.lstrip("_")


def get_request_model(operation: dict) -> tuple[str, str]:
    """Extract request schema from operation spec"""
    if "requestBody" not in operation:
        return None, None

    request_body = operation["requestBody"]
    content = request_body.get("content", {})
    json_content = content.get("application/json", {})
    schema_ref = json_content.get("schema", {}).get("$ref", "")

    if schema_ref:
        model_name = schema_ref.split("/")[-1]
        return model_name, pascal_case_to_snake(model_name)
    return None, None


def get_response_model(operation: dict) -> tuple[str, str]:
    """Extract response schema from operation spec"""
    responses = operation.get("responses", {})

    for status_code in ["200", "201", "202"]:
        if status_code in responses:
            response = responses[status_code]
            content = response.get("content", {})
            json_content = content.get("application/json", {})
            schema_ref = json_content.get("schema", {}).get("$ref", "")

            if schema_ref:
                model_name = schema_ref.split("/")[-1]
                return model_name, pascal_case_to_snake(model_name)

    return None, None


def generate_endpoint_code(
    path: str,
    method: str,
    operation_id: str,
    request_model: str | None,
    response_model: str | None,
    path_params: list[str],
) -> str:
    """Generate Python endpoint code"""

    method_lower = method.lower()

    # Build parameters
    params = []
    for param in path_params:
        params.append(f"{param}: str")

    if request_model:
        params.append(f"body: {request_model}")

    params_str = ", ".join(params) if params else ""

    # Build response type
    response_type = response_model if response_model else "dict"

    # Build function name from operation_id if available, else derive it
    func_name = (
        operation_id
        if operation_id
        else f"{method_lower}_{path.replace('/', '_').replace('{', '').replace('}', '')}"
    )

    code = f'''
@router.{method_lower}("{path}")
async def {func_name}({params_str}) -> {response_type}:
    """Operation ID: {operation_id}"""
    raise NotImplementedError("Endpoint not yet implemented")
'''
    return code


def generate_router_file(
    root_path: str,
    paths_info: dict,
) -> str:
    """Generate complete router file for a root path"""

    imports = """from fastapi import APIRouter
from typing import Optional
"""

    # Add model imports
    models_used = set()
    for path_info in paths_info.values():
        for method_info in path_info.values():
            if method_info.get("request_model"):
                models_used.add(method_info["request_model"])
            if method_info.get("response_model"):
                models_used.add(method_info["response_model"])

    if models_used:
        for model in sorted(models_used):
            snake_model = pascal_case_to_snake(model)
            imports += f"from models.openai.{snake_model} import {model}\n"

    imports += "\n"

    router_var = f"""
router = APIRouter(prefix="/{root_path}", tags=["{root_path.capitalize()}"])
"""

    endpoints = ""
    for path, methods in sorted(paths_info.items()):
        for method, info in sorted(methods.items()):
            # Extract path parameters
            path_params = re.findall(r"\{(\w+)\}", path)

            # Remove root prefix from path
            relative_path = path.replace(f"/{root_path}", "", 1) or "/"

            endpoint_code = generate_endpoint_code(
                relative_path,
                method,
                info["operation_id"],
                info["request_model"],
                info["response_model"],
                path_params,
            )
            endpoints += endpoint_code

    return imports + router_var + endpoints


def main():
    spec_path = "openai.documented.yml"
    spec = load_openapi_spec(spec_path)

    # Organize paths by root
    paths_by_root = defaultdict(lambda: defaultdict(dict))

    for path, path_item in spec.get("paths", {}).items():
        root_path = extract_root_path(path)

        for method, operation in path_item.items():
            if method.lower() not in ["get", "post", "put", "delete", "patch"]:
                continue

            operation_id = operation.get("operationId", f"{method}_{path}")
            request_model, _ = get_request_model(operation)
            response_model, _ = get_response_model(operation)

            paths_by_root[root_path][path][method.lower()] = {
                "operation_id": operation_id,
                "request_model": request_model,
                "response_model": response_model,
            }

    # Generate router files
    output_dir = Path("inference/server/routers/openai")
    output_dir.mkdir(parents=True, exist_ok=True)

    for root_path, paths_info in sorted(paths_by_root.items()):
        router_code = generate_router_file(root_path, paths_info)
        router_file = output_dir / f"{root_path}.py"

        with open(router_file, "w") as f:
            f.write(router_code)

        print(f"Generated {router_file}")

    # Generate __init__.py for routers
    init_content = '''"""OpenAI API routers"""
from .import_routers import *
'''

    (output_dir / "__init__.py").write_text(init_content)

    # Generate import_routers.py
    import_content = '''"""Auto-generated router imports"""
'''
    for root_path in sorted(paths_by_root.keys()):
        import_content += f"from .{root_path} import router as {root_path}_router\n"

    import_content += "\nROUTERS = [\n"
    for root_path in sorted(paths_by_root.keys()):
        import_content += f"    {root_path}_router,\n"
    import_content += "]\n"

    (output_dir / "import_routers.py").write_text(import_content)

    print(f"\nGenerated {output_dir / 'import_routers.py'}")
    print(f"Generated {output_dir / '__init__.py'}")
    print(f"\nTotal root paths: {len(paths_by_root)}")


if __name__ == "__main__":
    main()

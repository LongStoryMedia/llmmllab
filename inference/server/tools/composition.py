"""
Tool composition system for creating complex workflow tools from simple ones
"""
from typing import List

from .dynamic_tool import DynamicTool

class ToolComposer:
    """Compose multiple tools into complex workflows"""

    @staticmethod
    def create_workflow_tool(
        tools: List[DynamicTool], workflow_description: str
    ) -> DynamicTool:
        """Create a meta-tool that orchestrates multiple tools"""

        # Generate code that calls multiple tools in sequence
        tool_calls = []
        for i, tool in enumerate(tools):
            tool_calls.append(f"    result_{i} = {tool.function_name}(**kwargs)")

        # Join all tool codes
        tool_codes = "\n".join([tool.code for tool in tools])
        # Join all tool calls
        tools_call_str = "\n".join(tool_calls)
        
        # Create results list for formatted string
        result_list = []
        for i in range(len(tools)):
            result_list.append(f"result_{i}")
        results_str = ", ".join(result_list)

        workflow_code = f"""
def workflow_executor(**kwargs):
    \"\"\"
    {workflow_description}
    \"\"\"
    results = []
    
{tools_call_str}
    
    # Combine results (customize based on needs)
    combined_result = {{
        'workflow_description': '{workflow_description}',
        'individual_results': [{results_str}]
    }}
    
    return combined_result

# Include all component tool functions
{tool_codes}
"""

        return DynamicTool(
            name=f"workflow_{'_'.join([t.name[:5] for t in tools])}",  # Shortened name
            description=f"Workflow: {workflow_description}",
            code=workflow_code,
            function_name="workflow_executor",
        )

    @staticmethod
    def create_parallel_tool(
        tools: List[DynamicTool], workflow_description: str
    ) -> DynamicTool:
        """Create a tool that executes multiple tools in parallel"""
        
        # First, import necessary libraries for parallel execution
        import_code = """
import asyncio
import concurrent.futures
"""
        
        # Generate function stubs for parallel execution
        tool_functions = []
        for i, tool in enumerate(tools):
            tool_functions.append(f"""
def execute_tool_{i}(**kwargs):
    return {tool.function_name}(**kwargs)
""")
        
        # Join tool functions and codes
        tool_functions_str = "".join(tool_functions)
        tool_codes = "\n".join([tool.code for tool in tools])
        
        # Create futures append statements
        futures_append = ""
        for i in range(len(tools)):
            futures_append += f'futures.append(executor.submit(execute_tool_{i}, **kwargs))\n        '
        
        # Generate parallel execution code
        workflow_code = f"""{import_code}

def workflow_parallel_executor(**kwargs):
    \"\"\"
    {workflow_description} (Parallel execution)
    \"\"\"
    
    def run_in_executor(fn, **kw):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(fn, **kw)
            return future.result()
    
    # Execute tools in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers={len(tools)}) as executor:
        futures = []
        
        # Submit all tool executions
        {futures_append}
        
        # Collect results
        results = [future.result() for future in concurrent.futures.as_completed(futures)]
    
    # Return combined results
    combined_result = {{
        'workflow_description': '{workflow_description}',
        'parallel_results': results
    }}
    
    return combined_result

{tool_functions_str}

# Include all component tool functions
{tool_codes}
"""

        return DynamicTool(
            name=f"parallel_workflow_{hash(workflow_description) % 1000}",
            description=f"Parallel Workflow: {workflow_description}",
            code=workflow_code,
            function_name="workflow_parallel_executor",
        )

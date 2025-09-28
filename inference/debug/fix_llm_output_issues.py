#!/usr/bin/env python3
"""
Fix LLM output formatting issues:
1. Tool name showing as "unknown" instead of proper tool names
2. Web search results being truncated at 500 characters
"""

from pathlib import Path


def main():
    """Apply both fixes to the run.py file."""
    print("🔧 Fixing LLM output formatting issues...")

    run_py_path = Path(
        "/Users/lons7862/workspace/llmmllab/inference/runner/pipelines/run.py"
    )

    # Read current content
    with open(run_py_path, "r", encoding="utf-8") as f:
        content = f.read()

    print("1. Fixing tool name detection...")
    # Fix the _process_tool_start method
    old_tool_start = '''    def _process_tool_start(self, evt: StandardStreamEvent) -> Optional[ChatResponse]:
        """Process tool start events."""
        try:
            data = evt.get("data", {})
            tool_name = data.get("name", "unknown")
            tool_input = data.get("input", {})

            tool_txt = ""
            if isinstance(tool_input, dict):
                for key, value in tool_input.items():
                    str_value = str(value)
                    if len(str_value) > 100:
                        str_value = str_value[:100] + "..."
                    tool_txt += f"   - {key}: {str_value}\\n"

            if tool_txt:
                return create_streaming_chunk(
                    f"\\n\\n🔧 **Using {tool_name}**\\n{tool_txt}",
                    done=False,
                    role=MessageRole.OBSERVER,
                )
            else:
                return create_streaming_chunk(
                    f"\\n\\n🔧 **Using {tool_name}**\\n",
                    done=False,
                    role=MessageRole.OBSERVER,
                )

        except Exception as e:
            self.logger.error(f"Error processing tool start: {e}")
            return create_streaming_chunk("\\n\\n🔧 **Using tool**\\n")'''

    new_tool_start = '''    def _process_tool_start(self, evt: StandardStreamEvent) -> Optional[ChatResponse]:
        """Process tool start events with improved tool name detection."""
        try:
            data = evt.get("data", {})
            
            # Try multiple ways to extract tool name from LangGraph events
            tool_name = "unknown"
            
            # Method 1: Direct name field
            if "name" in data:
                tool_name = data["name"]
            # Method 2: Check if data has a 'tool' field with name
            elif "tool" in data and isinstance(data["tool"], dict) and "name" in data["tool"]:
                tool_name = data["tool"]["name"]
            # Method 3: Check for nested structure
            elif "input" in data and isinstance(data["input"], dict):
                if "tool" in data["input"] and isinstance(data["input"]["tool"], str):
                    tool_name = data["input"]["tool"]
                elif "name" in data["input"]:
                    tool_name = data["input"]["name"]
            # Method 4: Check event metadata
            elif "metadata" in evt and "name" in evt["metadata"]:
                tool_name = evt["metadata"]["name"]
            
            # Log for debugging
            self.logger.debug(f"Tool start event - extracted name: '{tool_name}', data keys: {list(data.keys())}")
            
            tool_input = data.get("input", {})

            tool_txt = ""
            if isinstance(tool_input, dict):
                for key, value in tool_input.items():
                    str_value = str(value)
                    if len(str_value) > 100:
                        str_value = str_value[:100] + "..."
                    tool_txt += f"   - {key}: {str_value}\\n"

            if tool_txt:
                return create_streaming_chunk(
                    f"\\n\\n🔧 **Using {tool_name}**\\n{tool_txt}",
                    done=False,
                    role=MessageRole.OBSERVER,
                )
            else:
                return create_streaming_chunk(
                    f"\\n\\n🔧 **Using {tool_name}**\\n",
                    done=False,
                    role=MessageRole.OBSERVER,
                )

        except Exception as e:
            self.logger.error(f"Error processing tool start: {e}")
            return create_streaming_chunk("\\n\\n🔧 **Using tool**\\n")'''

    content = content.replace(old_tool_start, new_tool_start)

    print("2. Fixing output truncation...")
    # Fix the _process_tool_end method
    old_tool_end = '''    def _process_tool_end(self, evt: StandardStreamEvent) -> Optional[ChatResponse]:
        """Process tool end events."""
        try:
            data = evt.get("data", {})
            tool_output = str(data.get("output", ""))

            if len(tool_output) > 500:  # Limit output length
                tool_output = tool_output[:500] + "..."

            return create_streaming_chunk(
                f"✅ **Tool completed**\\n{tool_output}\\n\\n",
                done=False,
                role=MessageRole.OBSERVER,
            )

        except Exception as e:
            self.logger.error(f"Error processing tool end: {e}")
            return create_streaming_chunk("✅ **Tool completed**\\n\\n")'''

    new_tool_end = '''    def _process_tool_end(self, evt: StandardStreamEvent) -> Optional[ChatResponse]:
        """Process tool end events with configurable output length."""
        try:
            data = evt.get("data", {})
            tool_output = str(data.get("output", ""))

            # Make truncation configurable and increase limit for better debugging
            max_output_length = getattr(self, 'max_tool_output_length', 2000)  # Default 2000 chars
            
            if len(tool_output) > max_output_length:
                # Show beginning and end for better context
                half_length = (max_output_length - 20) // 2  # Leave room for "..." separator
                tool_output = tool_output[:half_length] + "\\n... (truncated) ...\\n" + tool_output[-half_length:]

            return create_streaming_chunk(
                f"✅ **Tool completed**\\n{tool_output}\\n\\n",
                done=False,
                role=MessageRole.OBSERVER,
            )

        except Exception as e:
            self.logger.error(f"Error processing tool end: {e}")
            return create_streaming_chunk("✅ **Tool completed**\\n\\n")'''

    content = content.replace(old_tool_end, new_tool_end)

    # Write the updated content
    with open(run_py_path, "w", encoding="utf-8") as f:
        f.write(content)

    print("✅ Applied fixes to run.py")
    print("\nChanges made:")
    print("- Enhanced tool name extraction from LangGraph events")
    print("- Increased output length limit from 500 to 2000 characters")
    print("- Added beginning/end truncation for better context")
    print("- Added debug logging for tool name extraction")


if __name__ == "__main__":
    main()

# Tool Calling Fixes for GPT-OSS and Qwen Pipelines

## Issues Identified

### 1. Missing ToolMessage Support
**Problem**: The `from_lc_message` function in `utils/message.py` was missing support for LangChain's `ToolMessage` type, causing tool results to be incorrectly classified as "USER" messages.

**Fix**: Added proper handling for `ToolMessage` and "tool" message types, converting them to `MessageRole.SYSTEM` to preserve tool output context.

### 2. Qwen Pipeline Tool Calling
**Problem**: The Qwen3MoE pipeline wasn't extracting tool calls from generated content, and had missing methods causing streaming errors.

**Fix**: 
- Added `_parse_qwen_tool_calls()` method to extract JSON-formatted tool calls
- Added `_clean_tool_calls_from_content()` to remove tool JSON from user-visible content
- Added missing `_create_thinking_response()` and `_create_streaming_response()` methods
- Enhanced error handling in `finalize_streaming()`
- Updated system prompt to include tool calling instructions

### 3. GPT-OSS Tool Result Integration
**Problem**: While the GPT-OSS pipeline was successfully parsing tool calls, tool results weren't being properly integrated for subsequent iterations.

**Fix**: The ToolMessage handling fix addresses this by ensuring tool results are properly converted to system messages.

## Files Modified

### `/inference/utils/message.py`
- Added `ToolMessage` import from `langchain_core.messages`
- Added handling for `ToolMessage` instances in `from_lc_message()`
- Added handling for "tool" type in `LangChainMessage` processing

### `/inference/runner/pipelines/txt2txt/qwen3moe.py`
- Enhanced `_create_system_prompt()` with detailed tool calling instructions
- Added `_parse_qwen_tool_calls()` method to extract JSON tool calls
- Added `_clean_tool_calls_from_content()` method to clean user-visible content
- Updated `_agent_node()` to handle tool call extraction and formatting
- Added missing `_create_thinking_response()` and `_create_streaming_response()` methods
- Enhanced error handling in `finalize_streaming()`

## Expected Improvements

### GPT-OSS Pipeline
- Tool results will now be properly classified as system messages instead of user messages
- Subsequent iterations will have access to tool output context
- Multi-turn tool conversations should work correctly

### Qwen Pipeline
- Will now extract and execute tool calls in JSON format
- Streaming errors ("object of type 'NoneType' has no len()") should be resolved
- Tool calls will be properly formatted for LangGraph execution

## Testing

Run a simple test to verify ToolMessage handling:
```python
from langchain_core.messages import ToolMessage
from utils.message import from_lc_message

tool_msg = ToolMessage(content="Search results: ...", tool_call_id="call_1")
internal_msg = from_lc_message(tool_msg)
assert internal_msg.role == MessageRole.SYSTEM
```

## Usage Examples

### Qwen Tool Calling Format
The Qwen pipeline now expects tool calls in this JSON format:
```json
{
    "tool_calls": [
        {
            "name": "web_search",
            "arguments": {
                "query": "search term",
                "limit": 5
            }
        }
    ]
}
```

### GPT-OSS Harmony Format
The GPT-OSS pipeline continues to use the harmony format:
```
<|channel|>commentary to=functions <|constrain|>json<|message|>{"name":"web_search","arguments":{"query":"...","limit":5}}
```

## Next Steps

1. Test both pipelines with tool-requiring queries
2. Monitor logs for proper tool call parsing and execution
3. Verify that multi-turn tool conversations work correctly
4. Consider adding more robust tool call validation and error handling
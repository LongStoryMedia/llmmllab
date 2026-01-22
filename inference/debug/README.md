# Debug and Testing Scripts

This directory contains the most valuable debugging and testing scripts focused on core functionality: tool creation, tool calling, web extraction, and end-to-end pipeline testing.

## Current Scripts

### End-to-End Testing

- **`test_real_end_to_end_pipeline.py`** - Enhanced comprehensive e2e test with LLM output capture
  - Tests the complete QwenMoE pipeline with dynamic tool generation
  - Includes comprehensive LLM response logging and analysis
  - Command line options: `--no-capture`, `--print-output`, `--help`

### Tool Calling & Execution

- **`test_qwen_tool_calling.py`** - Qwen model tool calling validation
- **`test_end_to_end_tool_calling.py`** - Comprehensive tool calling pipeline test
- **`test_full_tool_execution.py`** - Full tool execution validation
- **`test_simplified_tool_integration.py`** - Streamlined tool integration testing
- **`test_tool_execution_pipeline.py`** - Tool execution pipeline validation

### Web Extraction

- **`test_web_extraction_dedicated.py`** - Dedicated web extraction testing
- **`test_simple_web_extractor.py`** - Simple web extraction validation

## Purpose

These scripts focus on the most critical functionality:

- **Tool Creation**: Dynamic tool generation and registration
- **Tool Calling**: LLM-to-tool communication and execution
- **Web Extraction**: Scrapy-based content extraction and processing
- **E2E Integration**: Complete pipeline validation with real-world scenarios

## Usage

All scripts can be run directly with Python:

```bash
python -m debug.test_real_end_to_end_pipeline [model_name] [--options]
python -m debug.test_qwen_tool_calling
python -m debug.test_web_extraction_dedicated
```

The main e2e test supports several command-line options for different testing modes and output control.

## Note

These are development and validation tools, not formal unit tests (`/test/`) or evaluation benchmarks (`/evaluation/`). They provide hands-on testing and debugging capabilities for the core platform features.

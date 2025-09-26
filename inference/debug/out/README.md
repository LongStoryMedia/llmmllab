# Debug Output Files

This directory contains output files generated during debug and test runs.

## File Types

- **LLM Output Files** (`llm_output_*.txt`): Complete transcripts of LLM responses during tests
- **Test Results** (`real_pipeline_test_*.json`): Detailed test execution results and metrics

## Sync Integration

These files are automatically synced from the remote server using:

```bash
# Pull only output files (benchmark data + debug output)
./sync-code.sh --pull-output

# Or as part of a full sync
./sync-code.sh
```

The sync script will:

1. Pull benchmark data from `benchmark_data/`
2. Pull debug output files from `debug/out/`
3. Push local code changes to server

## File Management

Files are excluded from version control via `.gitignore` but are preserved locally for analysis and debugging.

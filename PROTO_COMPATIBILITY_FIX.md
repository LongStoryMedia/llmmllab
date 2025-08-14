# Protobuf Compatibility Fix

This directory contains scripts to fix compatibility issues with generated protobuf files.

## Problem

When using a newer version of `protoc` to generate Python code, but running with an older version of the `protobuf` Python package, you may encounter errors like:

```
ImportError: cannot import name 'runtime_version' from 'google.protobuf'
```

This happens because newer versions of `protoc` (6.31.0+) include validation code that is not compatible with older versions of the `protobuf` Python package.

## Solution

Two scripts are provided to fix this issue:

1. `fix_proto_files.py` - A Python script that removes the problematic imports and validation calls
2. `fix_proto_files.sh` - A shell script version that can be used in Docker builds

## Usage

### Local Development

```bash
# Run the Python script to fix protobuf files in the local workspace
python fix_proto_files.py
```

### In Docker

The `Dockerfile.Inference.GRPC` has been modified to automatically run the fix script during the build process.

## Alternative Solutions

If you prefer to avoid fixing the files, you could also:

1. Install a newer version of the `protobuf` package that includes the `runtime_version` module
2. Use an older version of `protoc` to generate the files

However, fixing the files is often the simplest approach when you don't have control over the environment.

# Debug and Review Test Files

This directory contains debugging and review test files that are used for development, testing, and validation purposes but are not formal unit tests or evaluation benchmarks.

## Organization

- **Root level (`debug/`)**: Test files that were originally in the project root directory
  - Various debugging scripts and JSON configuration files
  - Integration testing and tool validation files
  - Performance and feature testing scripts

- **Inference level (`debug/inference/`)**: Test files that were originally in the `inference/` directory
  - End-to-end pipeline testing scripts
  - Tool execution and validation tests
  - Search functionality and embedding tests
  - Wrapper scripts for complex test scenarios

## Purpose

These files serve various purposes:

- **Debugging**: Scripts to help identify and fix issues during development
- **Validation**: Tests to ensure features work as expected in different scenarios  
- **Performance**: Scripts to test and validate performance improvements
- **Integration**: End-to-end tests that validate entire workflows
- **Prototyping**: Experimental code and proof-of-concept implementations

## Usage

Most of these files can be run directly with Python or executed as shell scripts (for wrapper files). Many include command-line arguments for different testing modes and configurations.

## Note

These are not part of the formal test suite (`test/` directory) or evaluation benchmarks (`inference/evaluation/` directory). They are development and debugging tools that help maintain and improve the codebase.

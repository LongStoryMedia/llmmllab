# Pipeline Tests

This directory contains unit tests for the pipeline factory and individual pipeline implementations in the inference service.

## Test Files

- `test_pipeline_factory.py` - Tests for the `PipelineFactory` class
- `test_pipelines.py` - Tests for the individual pipeline implementations (SD3Pipe, SDXLPipe, FluxPipe)
- `test_helpers.py` - Tests for the helper functions

## Running the Tests

You can run the tests using the provided script:

```bash
./run_tests.sh
```

Or you can run them directly using pytest:

```bash
python -m pytest test/ -v
```

## Test Coverage

The tests cover:

1. **Pipeline Factory**
   - Creating pipelines for different model types
   - Handling errors when models are not found
   - Loading LoRA weights

2. **Individual Pipelines**
   - SD3 Pipeline loading
   - SDXL Pipeline loading
   - Flux Pipeline loading
   - Handling different quantization levels

3. **Helper Functions**
   - Converting dtype strings to torch datatypes
   - Converting dtype strings to precision strings

## Test Data

Test models are loaded directly from `config/models.json`, ensuring that tests run with real configuration data. This approach:

1. Makes the tests more robust by using real-world model configurations
2. Automatically adapts when new models are added
3. Reduces maintenance by avoiding hardcoded test data

## Adding New Tests

To add new tests:

1. Create a new test file in the `test/pipelines` directory
2. Import the necessary modules
3. Create test classes and test methods
4. Use the fixtures defined in `conftest.py` for common objects

## Test Dependencies

- pytest
- pytest-mock

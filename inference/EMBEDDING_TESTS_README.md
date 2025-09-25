# 🧪 Embedding Pipeline Test Suite

This directory contains comprehensive tests for the Nomic embedding pipeline to diagnose and monitor embedding functionality on your Kubernetes pod.

## 📁 Test Files

### 1. `embedding_smoke_test.py` 
**Lightweight focused tests for quick diagnostics**

- ✅ **Basic embedding generation** - Can the pipeline create embeddings?
- ✅ **llama_decode error resilience** - How does batching handle decode failures?
- ✅ **Text processing** - Edge cases, Unicode, long texts, empty inputs

**Use this for**: Quick health checks, first-line diagnostics

### 2. `test_embedding_pipeline.py`
**Comprehensive test suite for deep analysis**

- ✅ **Smoke tests** - Basic functionality verification
- ✅ **Cosine similarity** - Semantic understanding validation
- ✅ **Text splitting** - Automatic chunking for long documents
- ✅ **Batch processing** - Different batch sizes and retry logic
- ✅ **Error handling** - Graceful degradation with problematic inputs

**Use this for**: Full pipeline validation, performance analysis

### 3. `run_embedding_tests_k8s.sh`
**Kubernetes runner script with environment setup**

- 🔧 Automatically sets up pod environment
- 📊 Collects environment diagnostics  
- 🚀 Runs tests using `/app/v.sh runner` 
- 📝 Provides detailed failure diagnostics

## 🚀 Running Tests on Kubernetes Pod

### Quick Smoke Test (Recommended)
```bash
# Connect to your pod
POD_NAME=$(k get pods -n ollama -o jsonpath='{.items[0].metadata.name}')
k exec -it -n ollama $POD_NAME -- bash

# Inside the pod, run the smoke test
cd /app
./embedding_smoke_test.py
```

### Full Test Suite
```bash
# Inside the pod
cd /app
./run_embedding_tests_k8s.sh
```

### Manual Test Execution
```bash
# Inside the pod - using the runner environment
/app/v.sh runner python /app/test_embedding_pipeline.py
```

## 📊 Understanding Test Results

### Exit Codes
- **0**: All tests passed ✅
- **1**: Degraded performance (some failures) ⚠️
- **2**: Critical issues (major failures) 🚨
- **3**: Test execution crashed 💥

### Test Status Indicators
- **✅ PASSED**: Test completed successfully
- **❌ FAILED**: Test failed but executed
- **💥 CRASHED**: Test crashed during execution

### Key Metrics to Watch

#### Basic Embedding Generation
- **Success**: Pipeline can create 768-dimensional embeddings
- **Failure**: `llama_decode returned -1`, model file issues, memory problems

#### llama_decode Error Resilience  
- **Success**: At least one batch size configuration works
- **Failure**: All batch sizes fail with decode errors
- **Solution**: Reduce `EMBEDDING_BATCH_SIZE` environment variable

#### Text Splitting
- **Success**: Long texts are automatically chunked and aggregated
- **Failure**: Token estimation or splitting logic issues
- **Solution**: Review text splitter configuration in `nom2.py`

#### Similarity Testing
- **Success**: Semantically similar texts have higher cosine similarity
- **Failure**: Poor model quality or embedding corruption
- **Solution**: Verify model file integrity

## 🔧 Troubleshooting Common Issues

### Issue: `llama_decode returned -1`
**Symptoms**: Embedding generation fails with decode errors
**Solutions**:
```bash
# Reduce batch size (inside pod)
export EMBEDDING_BATCH_SIZE=4
export EMBEDDING_MAX_RETRIES=5

# Re-run tests
./embedding_smoke_test.py
```

### Issue: Model file not found
**Symptoms**: `FileNotFoundError` or `GGUF file not found`
**Solutions**:
```bash
# Check model files
ls -la /app/models/
ls -la /models/

# Verify model configuration
cat /app/.models.json
```

### Issue: CUDA out of memory
**Symptoms**: GPU memory errors during embedding generation
**Solutions**:
```bash
# Check GPU memory
nvidia-smi

# Reduce GPU layers (edit model config)
# Or use CPU-only mode
export CUDA_VISIBLE_DEVICES=""
```

### Issue: Import errors
**Symptoms**: `ModuleNotFoundError` for pipeline components
**Solutions**:
```bash
# Verify environment setup
/app/v.sh runner python -c "import runner; print('OK')"

# Check Python path
/app/v.sh runner python -c "import sys; print(sys.path)"
```

## 📈 Performance Monitoring

### Recommended Test Schedule
- **Daily**: `embedding_smoke_test.py` (quick health check)
- **Weekly**: Full test suite for comprehensive validation
- **After changes**: Always run smoke tests before deployment

### Monitoring Metrics
```bash
# Inside pod - continuous monitoring
while true; do
  echo "=== $(date) ==="
  timeout 60 ./embedding_smoke_test.py
  echo "Exit code: $?"
  sleep 300  # 5 minutes
done
```

### Performance Baselines
- **Basic embedding**: < 2 seconds for single text
- **Batch processing**: < 5 seconds for 5 documents  
- **Long text**: < 10 seconds for 10K character document
- **Success rate**: > 90% for all test categories

## 🧰 Environment Configuration

### Key Environment Variables
```bash
# Embedding pipeline configuration
export EMBEDDING_BATCH_SIZE=8        # Batch size for processing
export EMBEDDING_MAX_RETRIES=3       # Retry attempts for decode errors
export EMBEDDING_ENABLE_BATCHING=true # Enable batching (vs single processing)

# Model configuration
export LOG_LEVEL=INFO                # Logging verbosity
export CUDA_VISIBLE_DEVICES=0        # GPU device selection
```

### Pod Resource Requirements
- **Memory**: Minimum 4GB, recommended 8GB+
- **GPU**: Optional but recommended for performance
- **Storage**: Model files ~500MB-2GB depending on configuration

## 📝 Test Output Files

Tests generate detailed JSON reports:
- `embedding_smoke_test_YYYYMMDD_HHMMSS.json` - Smoke test results
- `embedding_test_results_YYYYMMDD_HHMMSS.json` - Full test suite results

**Sample smoke test output**:
```json
{
  "timestamp": "2024-09-24T21:30:00.000Z",
  "results": {
    "Basic Embedding": true,
    "Decode Resilience": true,
    "Text Processing": false
  },
  "summary": {
    "passed": 2,
    "total": 3,
    "success_rate": 0.67
  }
}
```

## 🎯 Integration with Monitoring

### Kubernetes Health Checks
```yaml
# Add to your deployment
livenessProbe:
  exec:
    command: ["/app/embedding_smoke_test.py"]
  initialDelaySeconds: 60
  periodSeconds: 300
  timeoutSeconds: 120
```

### Alerting Rules
```bash
# Alert if success rate < 80%
if [[ $(./embedding_smoke_test.py; echo $?) -ne 0 ]]; then
  echo "ALERT: Embedding pipeline degraded" | mail -s "K8s Embedding Alert" admin@company.com
fi
```

---

These tests provide comprehensive coverage of your embedding pipeline functionality and should help identify the specific causes of the `llama_decode returned -1` errors and other embedding issues you've been experiencing.
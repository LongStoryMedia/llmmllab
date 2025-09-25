# 🧪 Embedding Test Suite - Quick Start Guide

## 🎯 **Purpose**
Test the Nomic embedding pipeline to diagnose `llama_decode returned -1` errors and validate embedding functionality on your Kubernetes pod.

## 🚀 **Quick Usage**

### 1. Deploy Tests to Pod
```bash
# Sync the test files to your cluster
cd /Users/lons7862/workspace/llmmllab/inference
./sync-code.sh
```

### 2. Run Smoke Tests (Recommended First Step)
```bash
# Connect to pod
POD_NAME=$(k get pods -n ollama -o jsonpath='{.items[0].metadata.name}')
k exec -it -n ollama $POD_NAME -- bash

# Inside pod - quick health check
cd /app
./embedding_smoke_test.py
```

### 3. Run Full Test Suite (If smoke tests reveal issues)
```bash
# Inside pod - comprehensive analysis
./run_embedding_tests_k8s.sh
```

## 📊 **What the Tests Check**

### Smoke Test (Fast - 1-2 minutes)
- ✅ **Basic embedding generation**: Can create 768D vectors?
- ✅ **Batch resilience**: Handles different batch sizes and decode errors?
- ✅ **Text processing**: Edge cases, Unicode, long texts?

### Full Test Suite (Comprehensive - 5-10 minutes)  
- ✅ **Cosine similarity**: Semantic understanding working?
- ✅ **Text splitting**: Long documents chunked properly?
- ✅ **Error handling**: Graceful degradation with bad inputs?
- ✅ **Performance**: Timing and resource usage analysis

## 🔍 **Interpreting Results**

### ✅ All Tests Pass
```
📊 Tests: 3/3 passed
🎯 Success Rate: 100.0%
🔍 Overall Status: HEALTHY
💡 Recommendations: ✨ All systems operational!
```
**Action**: Embedding pipeline is working correctly

### ⚠️ Partial Failures  
```
📊 Tests: 2/3 passed
🎯 Success Rate: 66.7%
🔍 Overall Status: DEGRADED
💡 Recommendations: ⚠️ Pipeline may struggle with llama_decode errors - reduce batch sizes
```
**Action**: Tune `EMBEDDING_BATCH_SIZE` environment variable

### 🚨 Critical Issues
```
📊 Tests: 0/3 passed
🎯 Success Rate: 0.0%
🔍 Overall Status: CRITICAL
💡 Recommendations: 🚨 Basic embedding is failing - check model configuration and files
```
**Action**: Check model files, memory, GPU configuration

## 🔧 **Common Fixes**

### Fix 1: Reduce Batch Size for llama_decode Errors
```bash
# Inside pod
export EMBEDDING_BATCH_SIZE=4
export EMBEDDING_MAX_RETRIES=5
./embedding_smoke_test.py
```

### Fix 2: Check Model Files
```bash
# Inside pod
ls -la /app/models/
ls -la /models/
cat /app/.models.json  # Check model configuration
```

### Fix 3: Memory Issues
```bash
# Inside pod
free -h                # Check available memory
nvidia-smi            # Check GPU memory
# Restart pod if needed
```

## 📝 **Test Files Created**

1. **`embedding_smoke_test.py`** - Quick health check (use this first)
2. **`test_embedding_pipeline.py`** - Comprehensive test suite  
3. **`run_embedding_tests_k8s.sh`** - Kubernetes runner with diagnostics
4. **`EMBEDDING_TESTS_README.md`** - Detailed documentation

## 🎯 **Expected Outcomes**

These tests will help you:

1. **Identify the root cause** of `llama_decode returned -1` errors
2. **Validate embedding quality** through similarity testing  
3. **Optimize batch sizes** for your specific hardware configuration
4. **Monitor pipeline health** over time
5. **Verify fixes** after making configuration changes

## 📈 **Next Steps After Running Tests**

### If Tests Pass
- ✅ Embedding pipeline is healthy
- ✅ Tool execution issues are likely in other components (web extraction, synthesis)
- ✅ Focus on the search pipeline improvements we made earlier

### If Tests Fail  
- 🔧 Use test results to tune `EMBEDDING_BATCH_SIZE` 
- 🔧 Check model file integrity and paths
- 🔧 Verify GPU/memory resources
- 🔧 Re-run tests after each fix to validate improvements

The combination of these embedding tests + the search pipeline improvements should resolve your tool execution issues!
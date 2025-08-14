# Debugging Container Issues

If the container is entering a crash loop or failing to start services properly, follow these debugging steps:

## Steps to Debug

1. Connect to the container:
```bash
kubectl exec -it -n ollama $(kubectl get pods -n ollama -o jsonpath='{.items[0].metadata.name}') -- /bin/bash
```

2. Run the debugging script:
```bash
cd /app
./debug_run.sh
```

This script will:
- Show system information (memory, disk space, processes)
- Display directory structure and available files
- Try to start the server directly for debugging

3. If the debug script shows issues, check these common problems:
   - Missing or incorrect file paths
   - Permission issues
   - Environment variables not set properly
   - Missing dependencies

## Restoring Normal Operation

After debugging, if you need to start all services normally:

```bash
cd /app
./run.sh
```

## Using Advanced Debugging Techniques

To run specific parts of the server with detailed logging:

```bash
# Start just the FastAPI server
cd /app
v server python -m server.main --host 0.0.0.0 --port 8000 --log-level debug

# Or start with uvicorn directly
cd /app
v server python -m uvicorn server.app:app --host 0.0.0.0 --port 8000 --log-level debug
```

Remember to check logs in `/var/log` directory for additional error information.

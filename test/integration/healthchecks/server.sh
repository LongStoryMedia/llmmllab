#!/bin/bash
# Server health check script for Docker

# Check if server is responding
if curl -sf http://localhost:8000/health > /dev/null; then
    echo "Server is healthy"
    exit 0
else
    echo "Server is not responding"
    exit 1
fi
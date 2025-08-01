#!/bin/bash
# startup.sh - Master script to start all services with the correct virtual environments
set -e

# Color codes for better visibility
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to log messages with timestamp
log() {
    local level=$1
    local message=$2
    local color=$3
    local timestamp
    timestamp=$(date +"%Y-%m-%d %H:%M:%S")
    echo -e "${color}[${timestamp}] [${level}] ${message}${NC}"
}

# Function to run commands with the appropriate virtual environment
run_with_env() {
    local env_name=$1
    local venv_path=$2
    local dir_path=$3
    shift 3
    log "INFO" "Running in ${env_name} environment: $*" "${BLUE}"
    cd "$dir_path" || { 
        log "ERROR" "Failed to change to $dir_path" "${RED}"
        return 1
    }
    source "$venv_path/bin/activate" || { 
        log "ERROR" "Failed to activate $venv_path" "${RED}"
        return 1
    }
    "$@" || {
        local exit_code=$?
        log "ERROR" "Command failed with exit code $exit_code: $*" "${RED}"
        deactivate
        cd /app
        return $exit_code
    }
    deactivate
    cd /app || { 
        log "ERROR" "Failed to return to /app" "${RED}"
        return 1
    }
    return 0
}

# Function to wait for a service to become available
wait_for_service() {
    local service_name=$1
    local host=$2
    local port=$3
    local max_attempts=$4
    local wait_seconds=$5
    log "INFO" "Waiting for $service_name to become available at http://$host:$port..." "${BLUE}"
    for (( i=1; i<=$max_attempts; i++ )); do
        # Use curl to check for a successful HTTP response.
        if curl --fail --silent --show-error --output /dev/null "http://$host:$port"; then
            log "INFO" "$service_name is available at $host:$port" "${GREEN}"
            return 0
        fi
        log "INFO" "Attempt $i/$max_attempts: $service_name not available yet, waiting ${wait_seconds}s..." "${YELLOW}"
        sleep $wait_seconds
    done
    log "ERROR" "Timed out waiting for $service_name to become available" "${RED}"
    return 1
}

log "INFO" "-----------------------------------------------------------" "${GREEN}"
log "INFO" "Starting LLM ML Lab with separate environments" "${GREEN}"
log "INFO" "-----------------------------------------------------------" "${GREEN}"

# Create a file to track running services
SERVICE_STATUS_FILE="/tmp/service_status"
echo "# Service status file - $(date)" > $SERVICE_STATUS_FILE

# Start Ollama if not already running
if curl --silent --output /dev/null "http://localhost:11434"; then
    log "INFO" "Ollama is already running and available. Skipping startup." "${GREEN}"
    OLLAMA_PID=$(pgrep -o ollama || echo "0")
    echo "ollama:running:${OLLAMA_PID}" >> $SERVICE_STATUS_FILE
else
    if command -v ollama &> /dev/null; then
        log "INFO" "Starting Ollama service..." "${BLUE}"
        ollama serve &> /var/log/ollama.log &
        OLLAMA_PID=$!
        if [ $? -eq 0 ]; then
            log "INFO" "Ollama started with PID $OLLAMA_PID" "${GREEN}"
            echo "ollama:running:$OLLAMA_PID" >> $SERVICE_STATUS_FILE
            # Wait for Ollama to become available
            wait_for_service "Ollama" "localhost" 11434 12 5
        else
            log "ERROR" "Failed to start Ollama service" "${RED}"
            echo "ollama:failed:0" >> $SERVICE_STATUS_FILE
        fi
    else
        log "WARNING" "Ollama not installed, skipping Ollama service startup" "${YELLOW}"
        echo "ollama:missing:0" >> $SERVICE_STATUS_FILE
    fi
fi

# Start server if app.py exists
if [ -f /app/server/app.py ]; then
    log "INFO" "Starting REST API server..." "${BLUE}"
    # This check can be improved to see if the port is already in use
    /app/run_with_env.sh "server" \
        'python -m uvicorn app:app --host 0.0.0.0 --port '"${PORT:-8000}"' --reload' &> /var/log/server_api.log &
    SERVER_PID=$!
    if [ $? -eq 0 ]; then
        log "INFO" "REST API server started on port ${PORT:-8000} with PID $SERVER_PID" "${GREEN}"
        echo "rest_api:running:$SERVER_PID" >> $SERVICE_STATUS_FILE
        # Wait for the REST API server to become available
        wait_for_service "REST API" "localhost" "${PORT:-8000}" 12 5
    else
        log "ERROR" "Failed to start REST API server" "${RED}"
        echo "rest_api:failed:0" >> $SERVICE_STATUS_FILE
    fi
else
    log "WARNING" "app.py not found in /app/server directory" "${YELLOW}"
    ls -la /app/server
    echo "rest_api:missing:0" >> $SERVICE_STATUS_FILE
fi

# Start gRPC server if grpc_server.py exists
if [ -f /app/server/grpc_server.py ]; then
    log "INFO" "Starting gRPC server..." "${BLUE}"
    # This check can be improved to see if the port is already in use
    run_with_env "server" "${SERVER_VENV}" "/app/server" \
        python grpc_server.py --port "${GRPC_PORT:-50051}" --max_workers "${GRPC_MAX_WORKERS:-10}" &> /var/log/grpc_server.log &
    GRPC_PID=$!
    if [ $? -eq 0 ]; then
        log "INFO" "gRPC server started on port ${GRPC_PORT:-50051} with PID $GRPC_PID" "${GREEN}"
        echo "grpc_server:running:$GRPC_PID" >> $SERVICE_STATUS_FILE
        # Wait for the gRPC server to become available
        # Note: This wait_for_service uses HTTP check, which will fail for a gRPC service.
        # A proper gRPC health check would be needed for this.
        log "INFO" "Skipping gRPC health check as it requires a gRPC-specific client." "${YELLOW}"
    else
        log "ERROR" "Failed to start gRPC server" "${RED}"
        echo "grpc_server:failed:0" >> $SERVICE_STATUS_FILE
    fi
else
    log "WARNING" "grpc_server.py not found in /app/server directory" "${YELLOW}"
    echo "grpc_server:missing:0" >> $SERVICE_STATUS_FILE
fi

log "INFO" "-----------------------------------------------------------" "${GREEN}"
log "INFO" "All services started. Container is now running." "${GREEN}"
log "INFO" "Service logs are available in /var/log/" "${BLUE}"
log "INFO" "Use 'docker exec' to run commands in this container." "${BLUE}"
log "INFO" "Example: docker exec -it <container_id> /app/run_with_env.sh server python -m your_command" "${BLUE}"
log "INFO" "-----------------------------------------------------------" "${GREEN}"

# Monitor running services
monitor_services() {
    while true; do
        log "INFO" "Checking service status..." "${BLUE}"
        while IFS=: read -r service status pid; do
            if [ "$status" = "running" ] && [ "$pid" -ne "0" ]; then
                if ! ps -p "$pid" &> /dev/null; then
                    log "ERROR" "$service (PID $pid) has died" "${RED}"
                    # Here you could implement restart logic if needed
                else
                    log "INFO" "$service (PID $pid) is running" "${GREEN}"
                fi
            fi
        done < "$SERVICE_STATUS_FILE"
        sleep 60  # Check every minute
    done
}

# Start the service monitoring in the background
monitor_services &
MONITOR_PID=$!

# Handle signals gracefully
cleanup() {
    log "INFO" "Shutting down services..." "${YELLOW}"
    # Kill all processes started by this script
    if [ -f "$SERVICE_STATUS_FILE" ]; then
        while IFS=: read -r service status pid; do
            if [ "$status" = "running" ] && [ "$pid" -ne 0 ]; then
                log "INFO" "Stopping $service (PID $pid)..." "${BLUE}"
                kill -TERM "$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null
            fi
        done < "$SERVICE_STATUS_FILE"
    fi
    # Kill the monitoring process
    if [ -n "$MONITOR_PID" ]; then
        kill "$MONITOR_PID" 2>/dev/null
    fi
    # Kill any other background jobs of this script
    jobs -p | xargs kill 2>/dev/null
    log "INFO" "All services stopped" "${GREEN}"
    exit 0
}

trap cleanup INT TERM EXIT

# Keep container running
tail -f /dev/null
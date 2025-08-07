..." "${YELLOW}"
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
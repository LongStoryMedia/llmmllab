#!/bin/bash
# Test script for RabbitMQ integration
# Usage: ./test_rabbitmq.sh

# Function to check if the RabbitMQ server is accessible
check_rabbitmq() {
    echo "Checking RabbitMQ connection..."
    if curl -s -u lsm:$RABBITMQ_PASSWORD http://192.168.0.122:15672/api/overview >/dev/null; then
        echo "RabbitMQ is accessible."
        return 0
    else
        echo "Error: Cannot connect to RabbitMQ. Check that RabbitMQ is running and credentials are correct."
        return 1
    fi
}

# Main test function
main() {
    echo "==== RabbitMQ Integration Test ===="

    # Check if RABBITMQ_PASSWORD is set
    if [ -z "$RABBITMQ_PASSWORD" ]; then
        echo "Error: RABBITMQ_PASSWORD environment variable is not set."
        echo "Please set it using: export RABBITMQ_PASSWORD=your_password"
        exit 1
    fi

    # Check RabbitMQ connection
    check_rabbitmq || exit 1

    # Start maistro with RabbitMQ enabled
    echo "Starting maistro with RabbitMQ enabled..."
    echo "Press Ctrl+C to stop the test when done."

    # # Enable RabbitMQ in the config
    # sed -i.bak 's/enabled: false/enabled: true/g' .config.yaml 2>/dev/null || true
    # go build . -o maistro

    # # Run maistro
    # echo "Running maistro..."
    # $(dirname "$0")/maistro
}

# Run the main function
main

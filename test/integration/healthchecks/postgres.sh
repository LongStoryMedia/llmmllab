#!/bin/bash
# PostgreSQL health check script for Docker

# Wait for PostgreSQL to be ready
until pg_isready -U postgres -d llmmll_test; do
    echo "Waiting for PostgreSQL..."
    sleep 1
done

echo "PostgreSQL is ready"
exit 0
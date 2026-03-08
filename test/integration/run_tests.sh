#!/bin/bash
# Integration test runner script

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Integration Test Runner${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker is not installed or not in PATH${NC}"
    exit 1
fi

# Check if docker-compose is available
if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}Error: docker-compose is not installed or not in PATH${NC}"
    exit 1
fi

# Change to test directory
cd "$SCRIPT_DIR"

# Parse arguments
TEST_PATTERN=""
RUN_MODE="test"

while [[ $# -gt 0 ]]; do
    case $1 in
        --build)
            RUN_MODE="build"
            shift
            ;;
        --logs)
            RUN_MODE="logs"
            shift
            ;;
        --clean)
            RUN_MODE="clean"
            shift
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --build     Build images before running tests"
            echo "  --logs      Show logs instead of running tests"
            echo "  --clean     Clean up containers and volumes"
            echo "  --help      Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                    # Run all tests"
            echo "  $0 --build            # Build and run tests"
            echo "  $0 --logs             # View logs"
            echo ""
            exit 0
            ;;
        *)
            TEST_PATTERN="$1"
            shift
            ;;
    esac
done

# Function to start services
start_services() {
    echo -e "${YELLOW}Starting integration test services...${NC}"
    docker-compose up -d
    echo -e "${GREEN}Services started.${NC}"
    echo ""
}

# Function to wait for services
wait_for_services() {
    echo -e "${YELLOW}Waiting for services to be ready...${NC}"

    # Wait for PostgreSQL
    echo -n "Waiting for PostgreSQL... "
    for i in {1..30}; do
        if docker-compose exec -T db pg_isready -U postgres -d llmmll_test &> /dev/null; then
            echo -e "${GREEN}ready${NC}"
            break
        fi
        if [ $i -eq 30 ]; then
            echo -e "${RED}timed out${NC}"
            exit 1
        fi
        sleep 2
    done

    # Wait for Redis
    echo -n "Waiting for Redis... "
    for i in {1..15}; do
        if docker-compose exec -T redis redis-cli ping &> /dev/null; then
            echo -e "${GREEN}ready${NC}"
            break
        fi
        if [ $i -eq 15 ]; then
            echo -e "${RED}timed out${NC}"
            exit 1
        fi
        sleep 2
    done

    # Wait for Server
    echo -n "Waiting for Server... "
    for i in {1..60}; do
        if curl -sf http://localhost:8000/health &> /dev/null; then
            echo -e "${GREEN}ready${NC}"
            break
        fi
        if [ $i -eq 60 ]; then
            echo -e "${YELLOW}server not responding (may need more time)${NC}"
            break
        fi
        sleep 2
    done

    echo ""
}

# Function to run tests
run_tests() {
    echo -e "${YELLOW}Running integration tests...${NC}"
    echo ""

    if [ -n "$TEST_PATTERN" ]; then
        docker-compose run --rm test-runner pytest test/integration/$TEST_PATTERN -v
    else
        docker-compose run --rm test-runner pytest test/integration -v
    fi

    echo ""
    echo -e "${GREEN}Tests completed.${NC}"
}

# Function to show logs
show_logs() {
    docker-compose logs -f
}

# Function to clean up
clean_up() {
    echo -e "${YELLOW}Cleaning up...${NC}"
    docker-compose down -v --remove-orphans
    echo -e "${GREEN}Cleanup complete.${NC}"
}

# Function to build images
build_images() {
    echo -e "${YELLOW}Building images...${NC}"
    docker-compose build --no-cache
    echo -e "${GREEN}Build complete.${NC}"
}

# Main execution
case $RUN_MODE in
    build)
        build_images
        start_services
        wait_for_services
        run_tests
        ;;
    logs)
        show_logs
        ;;
    clean)
        clean_up
        ;;
    test)
        start_services
        wait_for_services
        run_tests
        ;;
esac

# Always clean up after tests
echo -e "${YELLOW}Cleaning up...${NC}"
docker-compose down -v --remove-orphans
echo -e "${GREEN}Done.${NC}"
# Makefile for Runner Integration Testing

.PHONY: help start stop test clean logs rebuild

help:
	@echo "Runner Integration Testing Makefile"
	@echo ""
	@echo "Usage:"
	@echo "  make start      - Start all services in background"
	@echo "  make stop       - Stop all services"
	@echo "  make restart    - Restart all services"
	@echo "  make test       - Run integration tests"
	@echo "  make logs       - View logs (follow mode)"
	@echo "  make clean      - Remove containers and volumes"
	@echo "  make rebuild    - Rebuild images and restart"

start:
	@echo "Starting runner integration test services..."
	docker-compose up -d
	@echo ""
	@echo "Services started. Waiting for runner to be ready..."
	@sleep 30
	@echo "Checking runner health..."
	@grpcurl -plaintext localhost:50052 list || echo "Runner not ready yet"

stop:
	@echo "Stopping runner integration test services..."
	docker-compose down

restart: stop start

test:
	@echo "Running runner integration tests..."
	docker-compose run --rm test-runner pytest test/integration -v

test-runner:
	@echo "Running runner-specific tests..."
	docker-compose run --rm test-runner pytest test/integration/test_runner.py -v

logs:
	@echo "Viewing logs (Ctrl+C to exit)..."
	docker-compose logs -f

logs-runner:
	docker-compose logs -f runner

logs-db:
	docker-compose logs -f db

logs-redis:
	docker-compose logs -f redis

logs-test:
	docker-compose logs -f test-runner

clean:
	@echo "Cleaning up containers and volumes..."
	docker-compose down -v --remove-orphans
	@echo "Cleanup complete"

rebuild:
	@echo "Rebuilding images..."
	docker-compose build --no-cache
	@echo "Restarting services..."
	docker-compose up -d

status:
	@echo "Service status:"
	docker-compose ps

db-shell:
	@echo "Connecting to PostgreSQL..."
	docker-compose exec db psql -U postgres -d llmmll_test
# Makefile for Server Integration Testing

.PHONY: help start stop test clean logs rebuild

help:
	@echo "Server Integration Testing Makefile"
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
	@echo "Starting server integration test services..."
	docker-compose up -d
	@echo ""
	@echo "Services started. Waiting for health checks..."
	@sleep 30
	@echo "Checking server health..."
	@curl -f http://localhost:8000/health || echo "Server not ready yet"

stop:
	@echo "Stopping server integration test services..."
	docker-compose down

restart: stop start

test:
	@echo "Running server integration tests..."
	docker-compose run --rm test-runner pytest test/integration -v

test-health:
	@echo "Running health check tests..."
	docker-compose run --rm test-runner pytest test/integration/test_health.py -v

logs:
	@echo "Viewing logs (Ctrl+C to exit)..."
	docker-compose logs -f

logs-server:
	docker-compose logs -f server

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
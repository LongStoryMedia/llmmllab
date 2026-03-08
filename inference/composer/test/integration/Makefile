# Makefile for Composer Integration Testing

.PHONY: help start stop test clean logs rebuild

help:
	@echo "Composer Integration Testing Makefile"
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
	@echo "Starting composer integration test services..."
	docker-compose up -d
	@echo ""
	@echo "Services started. Waiting for composer to be ready..."
	@sleep 30
	@echo "Checking composer health..."
	@grpcurl -plaintext localhost:50051 list || echo "Composer not ready yet"

stop:
	@echo "Stopping composer integration test services..."
	docker-compose down

restart: stop start

test:
	@echo "Running composer integration tests..."
	docker-compose run --rm test-runner pytest test/integration -v

test-composer:
	@echo "Running composer-specific tests..."
	docker-compose run --rm test-runner pytest test/integration/test_composer.py -v

logs:
	@echo "Viewing logs (Ctrl+C to exit)..."
	docker-compose logs -f

logs-composer:
	docker-compose logs -f composer

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
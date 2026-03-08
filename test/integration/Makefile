# Makefile for Integration Testing

.PHONY: help start stop test clean logs rebuild

help:
	@echo "Integration Testing Makefile"
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
	@echo "Starting integration test services..."
	docker-compose up -d
	@echo ""
	@echo "Services started. Waiting for health checks..."
	@sleep 30
	@echo "Checking server health..."
	@curl -f http://localhost:8000/health || echo "Server not ready yet"

stop:
	@echo "Stopping integration test services..."
	docker-compose down

restart: stop start

test:
	@echo "Running integration tests..."
	docker-compose run --rm test-runner pytest test/integration -v

test-db:
	@echo "Running database tests..."
	docker-compose run --rm test-runner pytest test/integration/test_database.py -v

test-server:
	@echo "Running server tests..."
	docker-compose run --rm test-runner pytest test/integration/test_server.py -v

test-composer:
	@echo "Running composer tests..."
	docker-compose run --rm test-runner pytest test/integration/test_composer.py -v

test-runner:
	@echo "Running runner tests..."
	docker-compose run --rm test-runner pytest test/integration/test_runner.py -v

test-e2e:
	@echo "Running end-to-end tests..."
	docker-compose run --rm test-runner pytest test/integration/test_e2e_flow.py -v

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

db-migrate:
	@echo "Running database migrations..."
	docker-compose exec db psql -U postgres -d llmmll_test -c "SELECT set_config('search_path', 'public', false);"

coverage:
	@echo "Generating coverage report..."
	docker-compose run --rm test-runner pytest test/integration -v \
		--cov=server --cov=composer --cov=runner \
		--cov-report=html:test-output/coverage \
		--cov-report=xml:test-output/coverage.xml

junit:
	@echo "Generating JUnit report..."
	docker-compose run --rm test-runner pytest test/integration -v \
		--junitxml=test-output/junit.xml
# Integration Testing with Docker Compose

This directory contains the integration test setup for the llmmllab project.

## Architecture

The integration tests cover the full stack:

```
┌─────────────────────────────────────────────────────────────┐
│                        Test Runner                          │
│                  (pytest + httpx)                           │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                      Server (FastAPI)                       │
│                   Port: 8000 (HTTP)                         │
│  - Routers: OpenAI, Anthropic, Common                       │
│  - Middleware: Auth, DB Init, Validation                    │
└─────────────────────────┬───────────────────────────────────┘
                          │
          ┌───────────────┼───────────────┐
          │               │               │
          ▼               ▼               ▼
    ┌──────────┐    ┌──────────┐    ┌───────────┐
    │ Composer │    │  Runner  │    │  Database │
    │ LangGraph│    │ Pipelines│    │ PostgreSQL│
    └──────────┘    └──────────┘    └───────────┘
          │               │               │
          └───────────────┼───────────────┘
                          │
                  ┌───────▼───────┐
                  │    Redis      │
                  │   (Cache)     │
                  └───────────────┘
```

## Quick Start

```bash
# Start all services and run tests
cd test/integration
docker-compose up --build

# Or run services in background and then run tests
docker-compose up -d

# Run tests
docker-compose run --rm test-runner

# View logs
docker-compose logs -f server
```

## Services

### db (PostgreSQL + TimescaleDB + pgvector)
- **Image**: `timescale/timescaledb:latest-pg15`
- **Port**: 5433 (host) → 5432 (container)
- **Database**: `llmmll_test`
- **User**: `postgres`
- **Password**: `postgres`

### redis
- **Image**: `redis:7-alpine`
- **Port**: 6380 (host) → 6379 (container)

### server
- **Build**: From root Dockerfile.server
- **Port**: 8000
- **Environment**: Configured for testing
- **Health Check**: `/health` endpoint

### test-runner
- **Build**: From Dockerfile.test
- **Command**: Runs pytest integration tests
- **Volumes**: Mounts repository for test access

## Running Tests

### Run All Tests
```bash
docker-compose run --rm test-runner
```

### Run Specific Test File
```bash
docker-compose run --rm test-runner pytest test/integration/test_database.py -v
```

### Run with Coverage
```bash
docker-compose run --rm test-runner \
    pytest test/integration -v \
    --cov=server --cov=composer --cov=runner \
    --cov-report=xml:test-output/coverage.xml
```

### Run with Verbose Output
```bash
docker-compose run --rm test-runner -vv
```

## Test Organization

### Component-Specific Tests

Tests are organized per component with unit and integration tests:

```
server/test/
├── unit/              # Server unit tests
│   ├── __init__.py
│   ├── test_app.py    # App initialization tests
│   ├── middleware/    # Middleware tests
│   └── routers/       # Router tests
└── integration/       # Server integration tests
    ├── __init__.py
    ├── conftest.py    # Shared fixtures
    ├── test_database.py          # Database tests
    ├── test_integration_setup.py # Environment setup
    └── test_server.py            # Server API tests

composer/test/
├── unit/              # Composer unit tests
└── integration/       # Composer integration tests
    ├── __init__.py
    ├── conftest.py    # Shared fixtures
    ├── test_e2e_flow.py  # End-to-end tests (Composer to Runner)

runner/test/
├── unit/              # Runner unit tests
└── integration/       # Runner integration tests
    ├── __init__.py
    └── test_runner.py # Pipeline execution tests
```

### Integration Tests (Legacy)

The `test/integration/` directory contains legacy tests that span multiple components:

```
test/integration/
├── docker-compose.yml          # Service definitions
├── Dockerfile.server           # Server build definition
├── Dockerfile.test            # Test runner build definition
├── conftest.py                # Pytest fixtures
├── pytest.ini                 # Pytest configuration
├── server/                    # Server integration tests (legacy)
│   ├── __init__.py
│   └── test_health.py         # Health check tests
├── composer/                  # Composer integration tests (legacy)
│   └── __init__.py
├── runner/                    # Runner integration tests (legacy)
│   └── __init__.py
├── e2e/                       # End-to-end tests
│   ├── __init__.py
│   └── test_full_flow.py      # Full stack flow tests
├── test_database.py           # Database tests (legacy)
├── test_server.py             # Server endpoint tests (legacy)
├── test_composer.py           # Composer tests (legacy)
├── test_runner.py             # Runner tests (legacy)
└── test_e2e_flow.py           # End-to-end flow tests (legacy)
```

## Fixtures

### db_pool
Session-scoped database connection pool.

### db_connection
Function-scoped database connection with transaction rollback.

### clean_database
Function-scoped database cleanup before/after tests.

### server_client
Async HTTP client for server testing.

### test_user_id
Test user ID from environment or default.

### api_key
API key for testing.

## Environment Variables

### Database
- `DB_CONNECTION_STRING`: PostgreSQL connection string
- `DB_HOST`, `DB_PORT`, `DB_USER`, `DB_PASSWORD`, `DB_NAME`

### Server
- `SERVER_URL`: Server base URL (default: http://localhost:8000)
- `SERVER_API_KEY`: API key for testing

### Test Configuration
- `TEST_TIMEOUT`: Test timeout in seconds (default: 300)
- `TEST_PARALLEL`: Number of parallel workers
- `LOG_LEVEL`: Logging level

## CI/CD Integration

```yaml
# Example GitHub Actions workflow
name: Integration Tests

on:
  pull_request:
    branches: [main]

jobs:
  integration-tests:
    runs-on: ubuntu-latest
    services:
      db:
        image: timescale/timescaledb:latest-pg15
        env:
          POSTGRES_PASSWORD: postgres
        ports:
          - 5433:5432
      redis:
        image: redis:7-alpine
        ports:
          - 6380:6379
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.12'

      - name: Install dependencies
        run: |
          pip install -r server/requirements.txt
          pip install -r composer/requirements.txt
          pip install -r runner/requirements.txt
          pip install pytest pytest-asyncio pytest-cov

      - name: Run integration tests
        run: |
          pytest test/integration -v \
            --cov=server --cov=composer --cov=runner \
            --cov-report=xml
```

## Troubleshooting

### Database Connection Issues
```bash
# Check if database is healthy
docker-compose ps db

# Check database logs
docker-compose logs db

# Connect to database
docker-compose exec db psql -U postgres -d llmmll_test
```

### Server Not Starting
```bash
# Check server logs
docker-compose logs server

# Check server health
curl http://localhost:8000/health
```

### Test Timeout
```bash
# Increase timeout
docker-compose run --rm test-runner \
    pytest test/integration -v --timeout=600
```

## Cleanup

```bash
# Stop and remove containers
docker-compose down

# Remove volumes (database data)
docker-compose down -v

# Remove all unused images
docker image prune -a
```
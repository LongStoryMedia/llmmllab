# Proto repository Makefile
# Manages Protocol Buffer compilation for all services

.PHONY: all generate clean check

all: generate

# Generate gRPC code for all services
generate: generate-server generate-composer generate-runner

# Generate Python gRPC code
generate-python: generate-python-server generate-python-composer generate-python-runner

# Generate TypeScript gRPC code
generate-typescript: generate-typescript-server generate-typescript-composer generate-typescript-runner

# =============================================================================
# Server Service
# =============================================================================

generate-server:
	@echo "Generating server gRPC code..."
	@mkdir -p ../server/gen/python
	python -m grpc_tools.protoc \
		-I. \
		--python_out=../server/gen/python \
		--grpc_python_out=../server/gen/python \
		server/v1/server.proto \
		composer/v1/composer.proto \
		runner/v1/runner.proto \
		common/timestamp.proto \
		common/version.proto
	@echo "Server gRPC code generated"

generate-python-server:
	@echo "Generating Python server gRPC code..."
	@mkdir -p ../server/gen/python
	python -m grpc_tools.protoc \
		-I. \
		--python_out=../server/gen/python \
		--grpc_python_out=../server/gen/python \
		server/v1/server.proto \
		common/timestamp.proto \
		common/version.proto
	@echo "Python server gRPC code generated"

generate-typescript-server:
	@echo "Generating TypeScript server gRPC code..."
	@mkdir -p ../server/gen/typescript
	npx -y grpc_tools_node_protoc \
		--js_out=import_style=commonjs,binary:../server/gen/typescript \
		--grpc_out=../server/gen/typescript \
		-I. \
		server/v1/server.proto
	@echo "TypeScript server gRPC code generated"

# =============================================================================
# Composer Service
# =============================================================================

generate-composer:
	@echo "Generating composer gRPC code..."
	@mkdir -p ../composer/gen/python
	python -m grpc_tools.protoc \
		-I. \
		--python_out=../composer/gen/python \
		--grpc_python_out=../composer/gen/python \
		composer/v1/composer.proto \
		server/v1/server.proto \
		runner/v1/runner.proto \
		common/timestamp.proto \
		common/version.proto
	@echo "Composer gRPC code generated"

generate-python-composer:
	@echo "Generating Python composer gRPC code..."
	@mkdir -p ../composer/gen/python
	python -m grpc_tools.protoc \
		-I. \
		--python_out=../composer/gen/python \
		--grpc_python_out=../composer/gen/python \
		composer/v1/composer.proto \
		common/timestamp.proto \
		common/version.proto
	@echo "Python composer gRPC code generated"

generate-typescript-composer:
	@echo "Generating TypeScript composer gRPC code..."
	@mkdir -p ../composer/gen/typescript
	npx -y grpc_tools_node_protoc \
		--js_out=import_style=commonjs,binary:../composer/gen/typescript \
		--grpc_out=../composer/gen/typescript \
		-I. \
		composer/v1/composer.proto
	@echo "TypeScript composer gRPC code generated"

# =============================================================================
# Runner Service
# =============================================================================

generate-runner:
	@echo "Generating runner gRPC code..."
	@mkdir -p ../runner/gen/python
	python -m grpc_tools.protoc \
		-I. \
		--python_out=../runner/gen/python \
		--grpc_python_out=../runner/gen/python \
		runner/v1/runner.proto \
		server/v1/server.proto \
		composer/v1/composer.proto \
		common/timestamp.proto \
		common/version.proto
	@echo "Runner gRPC code generated"

generate-python-runner:
	@echo "Generating Python runner gRPC code..."
	@mkdir -p ../runner/gen/python
	python -m grpc_tools.protoc \
		-I. \
		--python_out=../runner/gen/python \
		--grpc_python_out=../runner/gen/python \
		runner/v1/runner.proto \
		common/timestamp.proto \
		common/version.proto
	@echo "Python runner gRPC code generated"

generate-typescript-runner:
	@echo "Generating TypeScript runner gRPC code..."
	@mkdir -p ../runner/gen/typescript
	npx -y grpc_tools_node_protoc \
		--js_out=import_style=commonjs,binary:../runner/gen/typescript \
		--grpc_out=../runner/gen/typescript \
		-I. \
		runner/v1/runner.proto
	@echo "TypeScript runner gRPC code generated"

# =============================================================================
# Validation
# =============================================================================

check:
	@echo "Checking proto files..."
	@for proto in $$(find . -name "*.proto"); do \
		echo "  Checking $$proto..."; \
		protoc --proto_path=. --cpp_out=/tmp "$$proto" 2>&1 || exit 1; \
	done
	@echo "All proto files are valid"

# =============================================================================
# Cleanup
# =============================================================================

clean:
	@echo "Cleaning generated code..."
	rm -rf ../server/gen/python/*
	rm -rf ../composer/gen/python/*
	rm -rf ../runner/gen/python/*
	@echo "Generated code cleaned"

# =============================================================================
# Help
# =============================================================================

help:
	@echo "Proto Repository Makefile"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  generate              - Generate gRPC code for all services"
	@echo "  generate-python       - Generate Python gRPC code"
	@echo "  generate-typescript   - Generate TypeScript gRPC code"
	@echo ""
	@echo "  generate-server       - Generate server gRPC code"
	@echo "  generate-composer     - Generate composer gRPC code"
	@echo "  generate-runner       - Generate runner gRPC code"
	@echo ""
	@echo "  generate-python-server    - Generate Python server gRPC code"
	@echo "  generate-python-composer  - Generate Python composer gRPC code"
	@echo "  generate-python-runner    - Generate Python runner gRPC code"
	@echo ""
	@echo "  check                 - Validate proto files"
	@echo "  clean                 - Remove generated code"
	@echo "  help                  - Show this help message"
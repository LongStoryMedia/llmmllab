# Changelog

All notable changes to LLM ML Lab will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.0.1] - 2025-09-28

### Added
- Multi-environment Python architecture (evaluation, server, runner)
- Schema-driven development with automatic code generation
- Kubernetes deployment automation
- Comprehensive benchmarking and evaluation suite
- Web content extraction and synthesis capabilities

### Changed
- **MAJOR**: Simplified web extraction service architecture
- Eliminated complex Scrapy-based extraction system
- Removed separate `simple_web_extractor.py` and `web_extraction_patch.py` files
- Integrated HTTP extraction directly into `WebExtractionService` using requests + BeautifulSoup
- Made `user_config` parameter optional with proper null handling

### Technical Improvements
- Direct `requests.Session` with retry strategies and browser-like headers
- `BeautifulSoup` parsing with intelligent content selectors
- Clean synchronous extraction without Scrapy complexity
- Reduced code complexity by ~200 lines while preserving functionality

### Memory Management
- GPU memory optimization with sophisticated tracking and automatic resource management
- On-demand model loading and unloading strategies
- Multiple memory optimization approaches based on available VRAM
- Enhanced hardware manager with improved GPU detection and allocation

### LangGraph Integration
- Enhanced base LangGraph pipeline architecture for complex workflows
- Improved state management and node coordination
- Better context preservation between pipeline nodes
- Robust error recovery mechanisms in graph-based pipelines

### Tool Usage & Execution
- Improved dynamic tool loading and execution capabilities
- Enhanced security validation for dynamic tool execution
- Better error handling and recovery in tool execution pipelines
- Fixed integration issues between tools and model pipelines

### Infrastructure
- PostgreSQL for persistent storage
- GPU resources for model inference
- Multi-service container orchestration

## [Unreleased]

### Planned
- Additional model pipeline integrations
- Enhanced context extension system
- Performance optimizations
- Additional benchmarking suites

---

**Legend:**
- `Added` for new features
- `Changed` for changes in existing functionality  
- `Deprecated` for soon-to-be removed features
- `Removed` for now removed features
- `Fixed` for any bug fixes
- `Security` for vulnerability fixes
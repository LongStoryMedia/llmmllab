# Release Notes

This directory contains release notes for LLM ML Lab versions.

## Structure

- `v<major>.<minor>.<patch>.md` - Individual release notes for each version
- `CHANGELOG.md` - Consolidated changelog across all versions

## Release Process

1. Create release notes file: `docs/releases/v<version>.md`
2. Update `CHANGELOG.md` with new version entry
3. Create git tag: `git tag v<version>`
4. Push tag: `git push origin v<version>`
5. Create GitHub release using the release notes

## Format

Each release notes file should include:

- **Major Improvements** - Key features and changes
- **Technical Improvements** - Implementation details
- **Architecture Benefits** - System-level improvements
- **Infrastructure** - Deployment and operational changes
- **Requirements** - System and dependency requirements
- **Breaking Changes** - Any backwards incompatible changes (if applicable)

## Versions

- [v0.0.1](v0.0.1.md) - Web Extraction Service Simplification
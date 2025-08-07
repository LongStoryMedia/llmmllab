# API Versioning Environment Configuration

The API versioning system can be configured using environment variables. This document explains how to use them.

## Environment Variables

| Variable    | Description                | Default Value | Example     |
|-------------|----------------------------|--------------|-------------|
| API_VERSION | Current API version string | "v1"         | "v2", "v3"  |

## Usage in Kubernetes

The API version is set in the Kubernetes deployment configuration in `deployment.yaml`:

```yaml
- name: API_VERSION
  value: "v1"
```

To change the API version for a deployment, update this value.

## Local Development

For local development, you can set the environment variable before starting the server:

```bash
# Linux/macOS
export API_VERSION=v1
python -m uvicorn server.app:app --reload

# Windows
set API_VERSION=v1
python -m uvicorn server.app:app --reload
```

## Notes

- Changing the API version will affect the URL structure for all API endpoints
- The UI client must be configured to use the same API version
- Backward compatibility should be maintained when introducing new API versions

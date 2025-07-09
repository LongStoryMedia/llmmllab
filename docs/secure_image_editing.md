# Secure Image Editing Workflow

This document describes the secure image editing workflow implemented in the system, which enables:

1. The UI to request maistro to edit an image
2. maistro to send the edit request (with image reference) to inference via RabbitMQ
3. inference to perform the edit after fetching the image from maistro via a secure endpoint

## Architecture

The image editing workflow follows these steps:

1. **User Request**: The user makes a request to edit an image via the UI, providing the image ID and editing parameters.

2. **Maistro Processing**:
   - Maistro receives the edit request with the image ID
   - It looks up the image in storage to verify it exists and the user has permission
   - It generates an internal URL for inference to use to fetch the image
   - It sends an image editing request to RabbitMQ, including the internal URL

3. **Inference Processing**:
   - Inference receives the edit request from RabbitMQ
   - It fetches the image from maistro using the internal URL and API key authentication
   - It performs the image editing operation
   - It sends the edited image result back to maistro via RabbitMQ

4. **Result Handling**:
   - Maistro receives the edited image result
   - It saves the image and updates metadata
   - It makes the result available to the UI

## Security Features

- **Internal API Key Authentication**: All internal service-to-service communication is secured with API keys.
- **Optional IP Allowlisting**: Can restrict internal endpoints to specific server IPs.
- **No Image Storage on Inference**: Images are fetched on demand and not stored on the inference service.
- **No Direct User Access**: Users cannot directly access the inference service or the internal endpoints.
- **Authorization Verification**: Maistro verifies user ownership of images before generating internal URLs.

## Configuration

### Maistro Configuration

Add to your `config.yaml`:

```yaml
internal:
  api_key: "your-secure-api-key-here"  # Required for internal service auth
  allowed_ips: "172.17.0.1,10.0.0.5"   # Optional comma-separated list of allowed IPs
```

### Inference Configuration

Add these environment variables to your inference service:

```
MAISTRO_INTERNAL_API_KEY=your-secure-api-key-here
MAISTRO_BASE_URL=http://maistro:8080  # Internal service URL
```

## API Endpoints

### Maistro Endpoints

- `POST /api/images/edit`: Edit an image (requires user authentication)
  - Request body includes image ID, prompt, and other parameters
  
- `GET /internal/images/:userID/:filename`: Internal endpoint for inference to fetch images
  - Requires internal API key authentication
  - Not accessible to end users

### RabbitMQ Messages

- **Image Editing Request**: 
  - Task type: `image_editing`
  - Contains prompt, parameters, and internal image URL

- **Image Editing Result**:
  - Contains the edited image data and metadata

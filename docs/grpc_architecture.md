```
                                          gRPC API
+---------------------+            +------------------------+
|                     |            |                        |
|   Go Maistro        | Request    |   Python Inference     |
|   Service           +----------->+   Service              |
|   (gRPC Client)     |            |   (gRPC Server)        |
|                     |            |                        |
+----------+----------+            +-----------+------------+
           |                                   |
           |                                   |
           v                                   v
+----------+-------------------+  +-----------+------------+
|                              |  |                        |
|   Web UI / API Clients       |  |   ML Models (PyTorch)  |
|                              |  |                        |
+------------------------------+  +------------------------+

Communication Flow:
1. Client sends request to Go Maistro Service
2. Maistro Service forwards request to Python Inference Service via gRPC
3. Inference Service processes request and streams response back to Maistro
4. Maistro streams response back to client

Key Features:
- Streaming for chat and text generation
- Bidirectional streaming for VRAM management
- Non-blocking image generation and editing
- Secure communication with TLS and API key authentication
```

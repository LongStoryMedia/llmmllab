"""
API Versioning Test Script

This script verifies that the API versioning works as expected by testing
both non-versioned and versioned endpoints.
"""

import requests
import sys

# Get the base URL from command line or use default
base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"

print(f"Testing API versioning against {base_url}...")

# Test endpoints
endpoints = [
    "/",  # Root endpoint
    "/chat/conversations",  # Non-versioned chat endpoint
    "/v1/chat/conversations",  # Versioned chat endpoint
    "/models",  # Non-versioned models endpoint
    "/v1/models",  # Versioned models endpoint
    "/config",  # Non-versioned config endpoint
    "/v1/config",  # Versioned config endpoint
    "/resources",  # Non-versioned resources endpoint
    "/v1/resources",  # Versioned resources endpoint
    "/internal",  # Internal endpoint (should be non-versioned only)
    "/v1/internal",  # Versioned internal endpoint (should not work)
]

# Headers for auth (if needed)
headers = {}

# Run tests
success_count = 0
failure_count = 0

for endpoint in endpoints:
    url = f"{base_url}{endpoint}"
    print(f"\nTesting endpoint: {url}")

    try:
        response = requests.get(url, headers=headers, timeout=10)
        print(f"  Status code: {response.status_code}")
        if response.status_code < 400:
            print(f"  SUCCESS: Endpoint {endpoint} is available")
            success_count += 1
        else:
            # For internal endpoints, 403 or 404 is expected
            if endpoint.endswith("internal") and response.status_code in [403, 404]:
                if endpoint == "/internal":
                    print(
                        "  SUCCESS: Internal endpoint requires authentication (403) or is not exposed (404)"
                    )
                    success_count += 1
                elif endpoint == "/v1/internal":
                    print(
                        "  SUCCESS: Versioned internal endpoint is not available (expected behavior)"
                    )
                    success_count += 1
            else:
                print(f"  FAILURE: Endpoint {endpoint} returned {response.status_code}")
                failure_count += 1
    except requests.RequestException as e:
        print(f"  ERROR: {str(e)}")
        failure_count += 1

print("\n=== Test Results ===")
print(f"Total endpoints tested: {len(endpoints)}")
print(f"Successful: {success_count}")
print(f"Failed: {failure_count}")

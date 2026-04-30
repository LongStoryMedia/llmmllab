import os

from dotenv import load_dotenv

# Load .env file for local development; in k8s env vars are injected directly.
load_dotenv()

from utils.logging import llmmllogger

logger = llmmllogger.logger.bind(component="Server")

# ── Logging ──────────────────────────────────────────────────────────
LOG_LEVEL = os.environ.get("LOG_LEVEL", "WARNING")

# ── Authentication ───────────────────────────────────────────────────
AUTH_ISSUER = os.environ.get("AUTH_ISSUER", "https://auth.longstorymedia.com")
AUTH_AUDIENCE = os.environ.get("AUTH_AUDIENCE", "lsm-client")
AUTH_CLIENT_ID = os.environ.get("AUTH_CLIENT_ID", "lsm-client")
AUTH_CLIENT_SECRET = os.environ.get("AUTH_CLIENT_SECRET", "")
AUTH_JWKS_URI = os.environ.get("AUTH_JWKS_URI", "https://auth.longstorymedia.com/keys")
DISABLE_AUTH = os.environ.get("DISABLE_AUTH", "").lower() == "true"
TEST_USER_ID = os.environ.get("TEST_USER_ID", "test-user-auth-disabled")

# ── Database ─────────────────────────────────────────────────────────
DB_HOST = os.environ.get("DB_HOST", "localhost")
DB_PORT = int(os.environ.get("DB_PORT", "5432"))
DB_USER = os.environ.get("DB_USER", "postgres")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "")
DB_NAME = os.environ.get("DB_NAME", "llmmll")
DB_SSLMODE = os.environ.get("DB_SSLMODE", "disable")
DB_CONNECTION_STRING = os.environ.get(
    "DB_CONNECTION_STRING",
    f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}?sslmode={DB_SSLMODE}",
)

if DB_CONNECTION_STRING:
    parts = DB_CONNECTION_STRING.split("@")
    if len(parts) > 1:
        masked_conn = f"***@{parts[1]}"
        logger.debug(f"Database connection string available: {masked_conn}")
else:
    logger.warning("No database connection string available")

# ── Redis ────────────────────────────────────────────────────────────
REDIS_ENABLED = os.environ.get("REDIS_ENABLED", "true").lower() == "true"
REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("REDIS_PORT", "6379"))
REDIS_PASSWORD = os.environ.get("REDIS_PASSWORD", "")
REDIS_DB = int(os.environ.get("REDIS_DB", "0"))
REDIS_CONVERSATION_TTL = int(os.environ.get("REDIS_CONVERSATION_TTL", "360"))
REDIS_MESSAGE_TTL = int(os.environ.get("REDIS_MESSAGE_TTL", "180"))
REDIS_SUMMARY_TTL = int(os.environ.get("REDIS_SUMMARY_TTL", "720"))
REDIS_POOL_SIZE = int(os.environ.get("REDIS_POOL_SIZE", "10"))
REDIS_MIN_IDLE_CONNECTIONS = int(os.environ.get("REDIS_MIN_IDLE_CONNECTIONS", "2"))
REDIS_CONNECT_TIMEOUT = int(os.environ.get("REDIS_CONNECT_TIMEOUT", "5"))

# ── Storage / Paths ──────────────────────────────────────────────────
IMAGE_DIR = os.environ.get("IMAGE_DIR", "/root/images")
IMAGE_RETENTION_HOURS = int(os.environ.get("IMAGE_RETENTION_HOURS", "24"))
CONFIG_DIR = os.environ.get("CONFIG_DIR", "/app/config")
HF_HOME = os.environ.get("HF_HOME", "/root/.cache/huggingface")

# ── Runner / llama.cpp ───────────────────────────────────────────────
LLAMA_SERVER_EXECUTABLE = os.environ.get(
    "LLAMA_SERVER_EXECUTABLE", "/llama.cpp/build/bin/llama-server"
)
GPU_POWER_CAP_PCT = int(os.environ.get("GPU_POWER_CAP_PCT", "85"))
PIPELINE_CACHE_TIMEOUT_MIN = int(os.environ.get("PIPELINE_CACHE_TIMEOUT_MIN", "30"))
PIPELINE_EVICTION_TIMEOUT_MIN = int(os.environ.get("PIPELINE_EVICTION_TIMEOUT_MIN", "60"))

# ── External API keys ───────────────────────────────────────────────
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
SEARX_HOST = os.environ.get("SEARX_HOST", "")

# ── Internal security ───────────────────────────────────────────────
INTERNAL_API_KEY = os.environ.get("INTERNAL_API_KEY", "")
INTERNAL_ALLOWED_IPS = os.environ.get(
    "INTERNAL_ALLOWED_IPS", "192.168.0.0/24,10.43.0.0/16"
)

# ── Feature flags ───────────────────────────────────────────────────
ENABLE_TOOL_CONTINUATION = (
    os.environ.get("ENABLE_TOOL_CONTINUATION", "true").lower() == "true"
)

# ── PyTorch ──────────────────────────────────────────────────────────
os.environ.setdefault("PYTORCH_NO_CUDA_MEMORY_CACHING", "1")

"""Auto-generated router imports"""

from .messages import router as messages_router

# models and files routers moved to common
# from .models import router as models_router
# from .files import router as files_router
from .completions import router as completions_router

ROUTERS = [
    messages_router,
    completions_router,
]

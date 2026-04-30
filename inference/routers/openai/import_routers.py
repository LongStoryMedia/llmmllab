"""Auto-generated router imports"""

from .assistants import router as assistants_router
from .audio import router as audio_router
from .batches import router as batches_router
from .chat import router as chat_router
from .chatkit import router as chatkit_router
from .completions import router as completions_router
from .containers import router as containers_router
from .conversations import router as conversations_router
from .embeddings import router as embeddings_router
from .evals import router as evals_router

# files router moved to common
# from .files import router as files_router
from .fine_tuning import router as fine_tuning_router
from .images import router as images_router

# models router moved to common
# from .models import router as models_router
from .moderations import router as moderations_router
from .organization import router as organization_router
from .projects import router as projects_router
from .projects import router as projects_router
from .realtime import router as realtime_router
from .responses import router as responses_router
from .threads import router as threads_router
from .uploads import router as uploads_router
from .vector_stores import router as vector_stores_router
from .videos import router as videos_router

ROUTERS = [
    assistants_router,
    audio_router,
    batches_router,
    chat_router,
    chatkit_router,
    completions_router,
    containers_router,
    conversations_router,
    embeddings_router,
    evals_router,
    fine_tuning_router,
    images_router,
    moderations_router,
    organization_router,
    projects_router,
    realtime_router,
    responses_router,
    threads_router,
    uploads_router,
    vector_stores_router,
    videos_router,
]

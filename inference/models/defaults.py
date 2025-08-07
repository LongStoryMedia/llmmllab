from .model import Model
from .model_details import ModelDetails
from .dev_stats import DevStats


def new_dev_stats() -> DevStats:
    """
    Create a new instance of DevStats with default values.

    Returns:
        DevStats: A new instance of DevStats with default values.
    """
    return DevStats(
        id="-1",
        uuid="",
        gpu_util=0.0,
        mem_util=0.0,
        mem_total=0,
        mem_used=0,
        mem_free=0,
        driver="",
        name="",
        serial="",
        display_mode="",
        display_active="",
        temperature=0.0,
    )


def new_model_details() -> ModelDetails:
    """
    Create a new instance of ModelDetails with default values.

    Returns:
        ModelDetails: A new instance of ModelDetails with default values.
    """
    return ModelDetails(
        description="",
        format="",
        family="",
        families=[],
        parameter_size="0",
        quantization_level="",
    )


def new_model() -> Model:
    """
    Create a new instance of Model with default values.

    Returns:
        Model: A new instance of Model with default values.
    """
    from models.model_details import ModelDetails

    return Model(
        name="",
        model="",
        modified_at="1970-01-01T00:00:00Z",
        size=0,
        digest="",
        details=new_model_details(),
    )

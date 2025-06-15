import os
import json
from typing import Dict, List, Optional, Union

from config import LORAS_CONFIG_PATH, DEFAULT_LORA_WEIGHT
from models.model_details import ModelDetails
from models.model import Model


class LoraService:
    """Service for managing LoRA weights."""

    def __init__(self):
        """Initialize the LoRA service."""
        # Dictionary to store LoRA configurations
        self.loras: Dict[str, Model] = {}
        self.active_loras: List[str] = []

        # Initialize from config or create default
        self._load_loras_config()

    def _load_loras_config(self):
        """Load LoRA configuration from file or create default."""
        try:
            if os.path.exists(LORAS_CONFIG_PATH):
                with open(LORAS_CONFIG_PATH, 'r') as f:
                    config = json.load(f)
                    loras_dict = config.get('loras', {})

                    # Convert dictionary back to Model objects
                    self.loras = {}
                    for lora_id, lora_data in loras_dict.items():
                        # Convert model_details dict to ModelDetails object first
                        if 'details' in lora_data and isinstance(lora_data['details'], dict):
                            lora_data['details'] = ModelDetails(
                                **lora_data['details'])
                        # Create Model object from dictionary
                        self.loras[lora_id] = Model(**lora_data)

                    self.active_loras = config.get('active_loras', [])
            else:
                # Create empty configuration
                self._save_loras_config()
        except Exception as e:
            print(f"Error loading LoRAs config: {e}")
            # Create empty configuration
            self._save_loras_config()

    def _save_loras_config(self):
        """Save LoRA configuration to file."""
        # Convert Model objects to dictionaries for JSON serialization
        serializable_loras = {}
        for lora_id, lora in self.loras.items():
            # Use model_dump() for newer Pydantic or dict() for older versions
            try:
                lora_dict = lora.model_dump()  # Pydantic v2
            except AttributeError:
                lora_dict = lora.dict()  # Pydantic v1
            serializable_loras[lora_id] = lora_dict

        config = {
            'loras': serializable_loras,
            'active_loras': self.active_loras
        }
        os.makedirs(os.path.dirname(LORAS_CONFIG_PATH), exist_ok=True)
        with open(LORAS_CONFIG_PATH, 'w') as f:
            json.dump(config, f, indent=2)

    def get_loras(self) -> List[Model]:
        """Get list of all available LoRAs."""
        return [lora for lora in self.loras.values()]

    def get_lora(self, lora_id: str) -> Optional[Model]:
        """Get details of a specific LoRA."""
        return self.loras.get(lora_id)

    def add_lora(self, name: str, source: str, description: Optional[str] = None, weight: float = DEFAULT_LORA_WEIGHT) -> Model:
        """Add a new LoRA to the configuration."""
        # Generate a unique Id for the LoRA

        # Add the LoRA to the configuration
        self.loras[source] = Model(
            name=name,
            model=source,  # Store the actual source/repository path, not the ID
            modified_at='',
            size=0,  # Size can be updated later if needed
            digest='',  # Digest can be updated later if needed
            details=ModelDetails(
                parent_model=None,
                format='lora',
                family='lora',
                families=['lora'],
                parameter_size='',
                quantization_level='',
                specialization='LoRA',
                description=description,
                weight=weight,
            )
        )

        # Save the updated configuration
        self._save_loras_config()

        return self.loras[source]

    def remove_lora(self, lora_id: str) -> bool:
        """Remove a LoRA from the configuration."""
        if lora_id in self.loras:
            # If removing an active LoRA, deactivate it first
            if lora_id in self.active_loras:
                self.active_loras.remove(lora_id)

            # Remove the LoRA from configuration
            del self.loras[lora_id]
            self._save_loras_config()
            return True

        return False

    def activate_lora(self, lora_id: str) -> bool:
        """Activate a LoRA for use."""
        if lora_id in self.loras and lora_id not in self.active_loras:
            # Add to active LoRAs
            self.active_loras.append(lora_id)
            self._save_loras_config()
            return True
        return False

    def deactivate_lora(self, lora_id: str) -> bool:
        """Deactivate a LoRA."""
        if lora_id in self.active_loras:
            self.active_loras.remove(lora_id)
            self._save_loras_config()
            return True
        return False

    def get_active_loras(self) -> List[Model]:
        """Get list of all active LoRAs."""
        return [self.loras[lora_id] for lora_id in self.active_loras if lora_id in self.loras]

    def set_lora_weight(self, lora_id: str, weight: float) -> bool:
        """Set the weight for a LoRA."""
        if lora_id in self.loras:
            self.loras[lora_id].details.weight = weight
            self._save_loras_config()
            return True
        return False


# Create a singleton instance
lora_service = LoraService()

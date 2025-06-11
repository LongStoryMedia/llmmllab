import os
import json
from typing import Dict, List, Optional, Union
from uuid import uuid4

from config import LORAS_CONFIG_PATH, DEFAULT_LORA_WEIGHT


class LoraService:
    """Service for managing LoRA weights."""

    def __init__(self):
        """Initialize the LoRA service."""
        # Dictionary to store LoRA configurations
        self.loras: Dict[str, dict] = {}
        self.active_loras: List[str] = []

        # Initialize from config or create default
        self._load_loras_config()

    def _load_loras_config(self):
        """Load LoRA configuration from file or create default."""
        try:
            if os.path.exists(LORAS_CONFIG_PATH):
                with open(LORAS_CONFIG_PATH, 'r') as f:
                    config = json.load(f)
                    self.loras = config.get('loras', {})
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
        config = {
            'loras': self.loras,
            'active_loras': self.active_loras
        }
        os.makedirs(os.path.dirname(LORAS_CONFIG_PATH), exist_ok=True)
        with open(LORAS_CONFIG_PATH, 'w') as f:
            json.dump(config, f, indent=2)

    def get_loras(self) -> List[dict]:
        """Get list of all available LoRAs."""
        return [lora for lora in self.loras.values()]

    def get_lora(self, lora_id: str) -> Optional[dict]:
        """Get details of a specific LoRA."""
        return self.loras.get(lora_id)

    def add_lora(self, name: str, source: str, description: Optional[str] = None, weight: float = DEFAULT_LORA_WEIGHT) -> dict:
        """Add a new LoRA to the configuration."""
        # Generate a unique Id for the LoRA
        lora_id = str(uuid4())

        # Add the LoRA to the configuration
        self.loras[lora_id] = {
            'id': lora_id,
            'name': name,
            'source': source,
            'description': description,
            'weight': weight,
            'is_active': False
        }

        # Save the updated configuration
        self._save_loras_config()

        return self.loras[lora_id]

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
            # Update is_active flag
            self.loras[lora_id]['is_active'] = True
            self._save_loras_config()
            return True
        return False

    def deactivate_lora(self, lora_id: str) -> bool:
        """Deactivate a LoRA."""
        if lora_id in self.active_loras:
            self.active_loras.remove(lora_id)
            # Update is_active flag
            if lora_id in self.loras:
                self.loras[lora_id]['is_active'] = False
            self._save_loras_config()
            return True
        return False

    def get_active_loras(self) -> List[dict]:
        """Get list of all active LoRAs."""
        return [self.loras[lora_id] for lora_id in self.active_loras if lora_id in self.loras]

    def set_lora_weight(self, lora_id: str, weight: float) -> bool:
        """Set the weight for a LoRA."""
        if lora_id in self.loras:
            self.loras[lora_id]['weight'] = weight
            self._save_loras_config()
            return True
        return False


# Create a singleton instance
lora_service = LoraService()

import nvsmi
from typing import Dict, List


class HardwareManager:
    """Minimal GPU memory manager. Queries nvsmi for VRAM stats."""

    def __init__(self):
        self._has_gpu = False
        self._gpu_count = 0
        self._gpus: List[nvsmi.GPU] = []
        try:
            self._gpus = list(nvsmi.get_gpus())
            self._has_gpu = len(self._gpus) > 0
            self._gpu_count = len(self._gpus)
        except Exception:
            pass

    @property
    def has_gpu(self) -> bool:
        return self._has_gpu

    @property
    def gpu_count(self) -> int:
        return self._gpu_count

    def available_vram_bytes(self) -> float:
        """Total free VRAM across all GPUs, in bytes."""
        if not self._has_gpu:
            return 0.0
        try:
            return sum(g.mem_free for g in self._gpus) * 1024 * 1024
        except Exception:
            return 0.0

    def gpu_stats(self) -> Dict[str, Dict]:
        """Per-GPU stats: name, total_mb, used_mb, free_mb, util_percent."""
        stats: Dict[str, Dict] = {}
        if not self._has_gpu:
            return stats
        try:
            for g in self._gpus:
                stats[str(g.id)] = {
                    "name": g.name,
                    "total_mb": g.mem_total,
                    "used_mb": g.mem_used,
                    "free_mb": g.mem_free,
                    "util_percent": g.mem_util,
                }
        except Exception:
            pass
        return stats


hardware_manager = HardwareManager()

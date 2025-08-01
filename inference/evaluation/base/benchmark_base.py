import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union

from datasets import (
    load_dataset,
    Split,
    NamedSplit,
    DatasetDict,
    Dataset,
    IterableDatasetDict,
    IterableDataset,
)
from .result_types import BenchmarkResult


class BenchmarkBase(ABC):
    """Abstract base class for all benchmarks."""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.logger = logging.getLogger(f"benchmark.{name}")
        self.huggingface_dataset = None  # Will be loaded when needed

    @abstractmethod
    def run(
        self, model_id: str, num_samples: int = 50, dataset_path: Optional[str] = None
    ) -> BenchmarkResult:
        """
        Run the benchmark on the specified model.

        Args:
            model_id: The ID of the model to evaluate
            num_samples: Number of questions to evaluate
            dataset_path: Optional path to a HuggingFace dataset in the format "space/dataset-name"
        """
        pass

    @abstractmethod
    def get_sample_questions(self) -> List[Dict[str, Any]]:
        """Get sample questions for the benchmark."""
        pass

    def validate_sample_count(self, num_samples: int, max_samples: int = -1) -> int:
        """Validate and adjust sample count."""
        if max_samples and num_samples > max_samples and max_samples > 0:
            self.logger.warning(
                f"Requested {num_samples} samples, but max is {max_samples}"
            )
            return max_samples
        return max(1, num_samples)

    def load_dataset_from_huggingface(
        self,
        dataset_path: str,
        split: Union[str, NamedSplit] = "test",
        config_name: Optional[str] = None,
        num_samples: Optional[int] = None,
    ) -> Dataset:
        """
        Loads a dataset from HuggingFace, handling splits and configs.
        Args:
            dataset_path: The name of the dataset (e.g., 'cais/mmlu').
            split: The dataset split to load (e.g., 'test', 'validation').
            config_name: The dataset configuration name (e.g., 'abstract_algebra').
            num_samples: Optional number of samples to take from the dataset.
        Returns:
            The loaded HuggingFace Dataset object.
        """
        self.logger.info(
            f"Loading dataset from HuggingFace: {dataset_path} with config: {config_name}"
        )
        try:
            dataset = load_dataset(dataset_path, config_name, split=split)  # type: ignore

            # If the dataset is a DatasetDict, we need to select the specific split
            if isinstance(dataset, DatasetDict) and split in dataset:
                dataset = dataset[split]
            elif isinstance(dataset, DatasetDict):
                # Fallback if the split is not a direct key
                self.logger.error(
                    f"Split '{split}' not found in dataset. Available splits: {list(dataset.keys())}"
                )
                raise ValueError(f"Split '{split}' not found in the dataset.")

            if num_samples is not None:
                self.logger.info(f"Sampling {num_samples} examples from the dataset.")
                dataset = dataset.select(range(min(num_samples, len(dataset))))  # type: ignore

            return dataset  # type: ignore
        except Exception as e:
            self.logger.error(f"Error loading HuggingFace dataset: {e}")
            raise RuntimeError(f"Failed to load dataset from HuggingFace: {e}")

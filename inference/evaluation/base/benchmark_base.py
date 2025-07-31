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
        split: Union[NamedSplit, Split, str, List[str], list[Split]] = "test",
        filters: Optional[Dict] = None,
    ) -> Optional[Union[IterableDatasetDict, IterableDataset, DatasetDict, Dataset]]:
        """
        Load a dataset from HuggingFace.

        Args:
            dataset_path: Path to the dataset in format "space/dataset-name"
            split: Dataset split to use (e.g., "dev", "test", "validation")
            filters: Optional dict of filters to apply to the dataset

        Returns:
            Dataset object or None if loading fails
        """
        if not dataset_path:
            self.logger.warning("No dataset path provided")
            return None

        try:
            # Load the specified dataset
            self.logger.info(f"Loading dataset from HuggingFace: {dataset_path}")

            # Special handling for known problematic datasets like cais/mmlu
            if "mmlu" in dataset_path.lower():
                # For MMLU datasets, try specific approaches
                splits_to_try = ["test", "dev", "val", "validation"]

                for split_name in splits_to_try:
                    try:
                        self.logger.info(f"Trying MMLU split: {split_name}")
                        # Try loading with streaming=False first for MMLU
                        dataset = load_dataset(
                            dataset_path, split=split_name, streaming=False
                        )
                        self.logger.info(
                            f"Successfully loaded MMLU split: {split_name}"
                        )
                        return dataset
                    except Exception as split_error:
                        self.logger.debug(
                            f"Failed to load MMLU split '{split_name}': {str(split_error)}"
                        )
                        continue

                # If individual splits fail, try loading the full dataset without streaming
                try:
                    self.logger.info("Trying to load full MMLU dataset without split")
                    dataset = load_dataset(dataset_path, streaming=False)

                    if isinstance(dataset, DatasetDict):
                        available_splits = list(dataset.keys())
                        self.logger.info(f"Available MMLU splits: {available_splits}")

                        # Prefer test, then dev, then val
                        for preferred_split in ["test", "dev", "val", "validation"]:
                            if preferred_split in available_splits:
                                self.logger.info(f"Using MMLU split: {preferred_split}")
                                return dataset[preferred_split]

                        # Use first available split
                        first_split = available_splits[0]
                        self.logger.info(
                            f"Using first available MMLU split: {first_split}"
                        )
                        return dataset[first_split]

                    return dataset

                except Exception as e:
                    self.logger.error(
                        f"Failed to load MMLU dataset without streaming: {str(e)}"
                    )

            # General approach for other datasets
            splits_to_try = [split, "test", "dev", "validation", "train"]

            dataset = None
            for split_name in splits_to_try:
                try:
                    self.logger.info(f"Trying split: {split_name}")
                    dataset = load_dataset(
                        dataset_path, split=split_name, streaming=True
                    )
                    self.logger.info(f"Successfully loaded split: {split_name}")
                    break
                except Exception as split_error:
                    self.logger.debug(
                        f"Failed to load split '{split_name}': {str(split_error)}"
                    )
                    continue

            if dataset is None:
                # Try loading without specifying a split
                self.logger.info("Trying to load dataset without specifying split")
                dataset = load_dataset(dataset_path, streaming=True)

                # If it's a DatasetDict, try to get a reasonable split
                if isinstance(dataset, (DatasetDict, IterableDatasetDict)):
                    available_splits = list(dataset.keys())
                    self.logger.info(f"Available splits: {available_splits}")

                    # Prefer test, then dev, then validation, then train
                    preferred_splits = ["test", "dev", "validation", "train"]
                    for preferred_split in preferred_splits:
                        if preferred_split in available_splits:
                            dataset = dataset[preferred_split]
                            self.logger.info(f"Using split: {preferred_split}")
                            break
                    else:
                        # Use the first available split
                        first_split = available_splits[0]
                        dataset = dataset[first_split]
                        self.logger.info(f"Using first available split: {first_split}")

            return dataset

        except Exception as e:
            self.logger.error(f"Failed to load dataset from HuggingFace: {str(e)}")
            self.logger.debug(f"Dataset path: {dataset_path}, Split: {split}")
            return None

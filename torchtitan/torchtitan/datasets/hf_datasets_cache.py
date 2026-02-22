# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

import torch

from datasets import Dataset, load_dataset
from datasets.distributed import split_dataset_by_node
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import IterableDataset

from torchtitan.components.dataloader import ParallelAwareDataloader
from torchtitan.components.tokenizer import Tokenizer
from torchtitan.config_manager import JobConfig
from torchtitan.tools.logging import logger


def _load_c4_dataset(dataset_path: str, split: str = "train"):
    """Load C4 dataset with default configuration and better timeout handling."""
    # Set environment variables for better timeout handling
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "1200"  # 20 minutes
    os.environ["REQUESTS_TIMEOUT"] = "1200"

    try:
        return load_dataset(
            dataset_path,
            name="en",
            split=split,
            streaming=True,
            trust_remote_code=True,
            download_mode="reuse_cache_if_exists",
        )
    except Exception as e:
        logger.warning(f"Failed to load c4 dataset: {e}")
        logger.info("Retrying with force_download=False...")
        return load_dataset(
            dataset_path,
            name="en",
            split=split,
            streaming=True,
            trust_remote_code=True,
            download_mode="force_redownload",
        )


def _load_fineweb_dataset(dataset_path: str, split: str = "train"):
    """Load fineweb dataset with default configuration and better timeout handling."""
    # Set environment variables for better timeout handling
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "1200"  # 20 minutes
    os.environ["REQUESTS_TIMEOUT"] = "1200"

    try:
        return load_dataset(
            dataset_path,
            name="default",
            split=split,
            streaming=True,
            trust_remote_code=True,
            download_mode="reuse_cache_if_exists",
        )
    except Exception as e:
        logger.warning(f"Failed to load fineweb dataset: {e}")
        logger.info("Retrying with force_download=False...")
        return load_dataset(
            dataset_path,
            name="default",
            split=split,
            streaming=True,
            trust_remote_code=True,
            download_mode="force_redownload",
        )


def _load_longalign_dataset(dataset_path: str, split: str = "train"):
    """Load LongAlign dataset with default configuration."""
    # Set environment variables for better timeout handling
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "1200"  # 20 minutes
    os.environ["REQUESTS_TIMEOUT"] = "1200"

    try:
        return load_dataset(
            dataset_path,
            split=split,
            streaming=True,
            trust_remote_code=True,
            download_mode="reuse_cache_if_exists",
        )
    except Exception as e:
        logger.warning(f"Failed to load longalign dataset: {e}")
        logger.info("Retrying with force_download=False...")
        return load_dataset(
            dataset_path,
            split=split,
            streaming=True,
            trust_remote_code=True,
            download_mode="force_redownload",
        )


def _process_c4_text(sample: dict[str, Any]) -> str:
    """Process C4 dataset sample text."""
    return sample["text"]


def _process_longalign_text(sample: dict[str, Any]) -> dict[str, Any]:
    """Process LongAlign dataset sample text."""
    # Return the full sample since LongAlignDataset needs the conversation structure
    return sample


@dataclass
class DatasetConfig:
    path: str
    loader: Callable
    text_processor: Callable


# Add your dataset here here - more information at docs/datasets.md
DATASETS = {
    "c4": DatasetConfig(
        path="allenai/c4",
        loader=_load_c4_dataset,
        text_processor=_process_c4_text,
    ),
    "c4_test": DatasetConfig(
        path="tests/assets/c4_test",
        loader=lambda path: load_dataset(path, split="train"),
        text_processor=_process_c4_text,
    ),
    "fineweb": DatasetConfig(
        path="HuggingFaceFW/fineweb",
        loader=_load_fineweb_dataset,
        text_processor=_process_c4_text,
    ),
    "longalign": DatasetConfig(
        path="THUDM/LongAlign-10k",
        loader=_load_longalign_dataset,
        text_processor=_process_longalign_text,
    ),
}


def _validate_dataset(
    dataset_name: str, dataset_path: str | None = None
) -> tuple[str, Callable, Callable]:
    """Validate dataset name and path."""
    if dataset_name not in DATASETS:
        raise ValueError(
            f"Dataset {dataset_name} is not supported. "
            f"Supported datasets are: {list(DATASETS.keys())}"
        )

    config = DATASETS[dataset_name]
    path = dataset_path or config.path
    logger.info(f"Preparing {dataset_name} dataset from {path}")
    return path, config.loader, config.text_processor


class HuggingFaceDataset(IterableDataset, Stateful):
    def __init__(
        self,
        dataset_name: str,
        dataset_path: str | None,
        tokenizer: Tokenizer,
        seq_len: int = 2048,
        dp_rank: int = 0,
        dp_world_size: int = 1,
        infinite: bool = False,
        split: str = "train",
    ) -> None:
        # Force lowercase for consistent comparison
        dataset_name = dataset_name.lower()

        path, dataset_loader, text_processor = _validate_dataset(
            dataset_name, dataset_path
        )
        if dataset_name == "c4_test":
            ds = dataset_loader(path)
        else:
            # if split == "validation":
            #     # FineWeb does not have a validation split, so we use the c4_test for validation
            #     assert dataset_name == "fineweb" or dataset_name == "longalign", "FineWeb and LongAlign do not have a validation split, so we use the c4_test for validation. For other datasets, please use a different split."
            #     ds = dataset_loader("tests/assets/c4_test")
            if split == "validation":
                # FineWeb does not have a validation split, so we use the c4_test for validation
                assert (
                    dataset_name == "fineweb" or dataset_name == "longalign"
                ), "FineWeb and LongAlign do not have a validation split, so we use the 100 samples from train itself."
                ds = dataset_loader(path, split="train")
                ds = ds.take(100)  # 100 samples for validation
                # if dataset_name == "longalign":
                #     ds = ds.sort("length")
            else:
                if dataset_name == "fineweb" or dataset_name == "longalign":
                    ds = dataset_loader(path, split="train")
                    ds = ds.skip(100)  # Skip first 100 samples for training
                    # if dataset_name == "longalign":
                    #     ds = ds.sort("length")
                else:
                    ds = dataset_loader(path, split=split)

        self.dataset_name = dataset_name
        self._data = split_dataset_by_node(ds, dp_rank, dp_world_size)
        self._tokenizer = tokenizer
        self.seq_len = seq_len
        self.infinite = infinite
        self._text_processor = text_processor

        # Variables for checkpointing
        self._sample_idx = 0
        self._token_buffer: list[int] = []

    def _get_data_iter(self):
        # For map-style datasets, resume by skipping to the correct index
        # For iterable-style datasets, the underlying iterator already points to the correct index
        if isinstance(self._data, Dataset):
            if self._sample_idx == len(self._data):
                return iter([])
            else:
                return iter(self._data.skip(self._sample_idx))

        return iter(self._data)

    def __iter__(self):
        max_buffer_token_len = 1 + self.seq_len

        while True:
            for sample in self._get_data_iter():
                # Use the dataset-specific text processor
                sample_text = self._text_processor(sample)
                sample_tokens = self._tokenizer.encode(sample_text, bos=True, eos=True)
                self._token_buffer.extend(sample_tokens)
                self._sample_idx += 1

                while len(self._token_buffer) >= max_buffer_token_len:
                    x = torch.LongTensor(self._token_buffer[:max_buffer_token_len])
                    # update tokens to the remaining tokens
                    self._token_buffer = self._token_buffer[max_buffer_token_len:]
                    input = x[:-1]
                    label = x[1:]
                    yield {"input": input}, label

            if not self.infinite:
                logger.warning(f"Dataset {self.dataset_name} has run out of data")
                break
            else:
                # Reset offset for the next iteration
                self._sample_idx = 0
                logger.warning(f"Dataset {self.dataset_name} is being re-looped")
                # Ensures re-looping a dataset loaded from a checkpoint works correctly
                if not isinstance(self._data, Dataset):
                    if hasattr(self._data, "set_epoch") and hasattr(
                        self._data, "epoch"
                    ):
                        self._data.set_epoch(self._data.epoch + 1)

    def load_state_dict(self, state_dict):
        self._token_buffer = state_dict["token_buffer"]

        if isinstance(self._data, Dataset):
            self._sample_idx = state_dict["sample_idx"]
        else:
            assert "data" in state_dict
            self._data.load_state_dict(state_dict["data"])

    def state_dict(self):
        _state_dict = {"token_buffer": self._token_buffer}

        if isinstance(self._data, Dataset):
            _state_dict["sample_idx"] = self._sample_idx
        else:
            # Save the iterable dataset's state to later efficiently resume from it
            # https://huggingface.co/docs/datasets/v3.5.0/en/stream#save-a-dataset-checkpoint-and-resume-iteration
            _state_dict["data"] = self._data.state_dict()

        return _state_dict


class LongAlignDataset(HuggingFaceDataset):
    """Dataset class for LongAlign that handles conversation-based training with masked labels.

    Supports offline caching of processed data to .npy files for efficient repeated loading.

    Features:
    - Automatic caching of processed conversations to avoid repeated tokenization
    - Distributed training support with per-rank caching
    - Fallback to streaming behavior if caching is disabled
    - Proper state management for checkpointing

    Args:
        model_type (str): Model type for conversation formatting ('llama' or 'chatglm')
        cache_dir (str): Directory to store cached processed data
        use_cache (bool): Whether to use caching functionality
        **kwargs: Additional arguments passed to parent HuggingFaceDataset
    """

    def __init__(
        self,
        model_type: str = "llama",
        cache_dir: str = "",
        use_cache: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_type = model_type.lower()
        self.cache_dir = cache_dir
        self.use_cache = use_cache
        self.split = kwargs.get("split", "train")

        # Special tokens from longalign_tokenizer.py
        # self.BEGIN_TOKEN = "<|begin_of_text|>" # for tiktokenizer
        # self.END_TOKEN = "<|end_of_text|>" # for tiktokenizer
        self.BEGIN_TOKEN, self.END_TOKEN = (
            "<|reserved_special_token_0|>",
            "<|reserved_special_token_1|>",
        )  # for tiktokenizer
        self.EOS_ID = self._tokenizer.eos_id
        # Use EOS token as pad token since tiktokenizer doesn't have a dedicated pad token
        self.PAD_ID = self.EOS_ID

        # # Add special tokens to tokenizer
        # special_tokens = {'cls_token': self.BEGIN_TOKEN, 'sep_token': self.END_TOKEN}
        # self._tokenizer.model.add_special_tokens(special_tokens)

        # Get token IDs for BEGIN and END tokens
        self.BEGIN_ID = self._tokenizer.special_tokens[self.BEGIN_TOKEN]
        self.END_ID = self._tokenizer.special_tokens[self.END_TOKEN]

        # Setup cache file paths
        if self.use_cache:
            os.makedirs(self.cache_dir, exist_ok=True)
            # Create unique cache key based on dataset config including distributed settings
            dp_rank = kwargs.get("dp_rank", 0)
            dp_world_size = kwargs.get("dp_world_size", 1)
            cache_key = f"{self.dataset_name}_{self.model_type}_{self.seq_len}_{self.split}_rank{dp_rank}_world{dp_world_size}"
            self.inputs_cache_path = os.path.join(
                self.cache_dir, f"{cache_key}_inputs.npy"
            )
            self.labels_cache_path = os.path.join(
                self.cache_dir, f"{cache_key}_labels.npy"
            )

            # Check if cached files exist and prepare data accordingly
            if os.path.exists(self.inputs_cache_path) and os.path.exists(
                self.labels_cache_path
            ):
                logger.info(f"Loading cached processed data from {self.cache_dir}")
                self._load_cached_data()
                self.use_cached_data = True
            else:
                logger.info(f"Processing and caching data to {self.cache_dir}")
                self._process_and_cache_data()
                self.use_cached_data = True
        else:
            self.use_cached_data = False

        # Variables for checkpointing - override parent's token buffer since we handle differently
        self._processed_samples: list[tuple[torch.Tensor, torch.Tensor]] = []

    def _build_longalign_input(
        self, conversations: list[dict[str, str]]
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Build input and labels for LongAlign conversation data."""
        # Filter zero-width characters from assistant responses
        zero_width_chars = ["\u200b", "\u200c", "\u200d", "\ufeff"]
        for conv in conversations:
            if conv["role"] == "assistant":
                for char in zero_width_chars:
                    conv["content"] = conv["content"].replace(char, "")

        # Ensure even number of messages (user-assistant pairs)
        if len(conversations) % 2 != 0:
            conversations = conversations[:-1]
        if len(conversations) == 0:
            return None

        # Build conversation text with special format
        formatted_conversations = []
        input_tokens = []
        for i in range(0, len(conversations), 2):
            local_rank = i // 2
            user_input = (
                conversations[i]["content"] if conversations[i]["content"] else ""
            )
            assistant_response = (
                conversations[i + 1]["content"]
                if conversations[i + 1]["content"]
                else ""
            )

            if user_input == "" or assistant_response == "":
                if i == len(conversations) - 2:
                    return None
                continue

            # Wrap assistant response with special tokens
            assistant_with_tokens = (
                self.BEGIN_TOKEN + assistant_response + self.END_TOKEN
            )

            if i == 0:
                curr_text = f"[INST]{user_input}[/INST]{assistant_with_tokens}"
                curr_tokens = self._tokenizer.encode(
                    curr_text, bos=False, eos=False, allowed_special="all"
                )
            else:  # llama format
                # formatted_conversations.append(
                #     f'[INST]{user_input}[/INST]{assistant_with_tokens}'
                # )
                curr_text = f"\n\n[INST]{user_input}[/INST]{assistant_with_tokens}"
                curr_tokens = self._tokenizer.encode(
                    user_input, bos=False, eos=False, allowed_special="all"
                )

        # Join all conversations
        full_text = "\n\n".join(formatted_conversations)

        # Tokenize
        input_tokens = self._tokenizer.encode(
            full_text, bos=False, eos=False, allowed_special="all"
        )
        inputs = torch.tensor(input_tokens, dtype=torch.int64)

        # Create labels - initially all -100 (ignore)
        labels = torch.full_like(inputs, -100)

        # Find BEGIN and END token positions
        begin_positions = (inputs == self.BEGIN_ID).nonzero(as_tuple=True)[0].tolist()
        end_positions = (inputs == self.END_ID).nonzero(as_tuple=True)[0].tolist()

        if len(begin_positions) != len(end_positions):
            logger.warning("Mismatch between BEGIN and END tokens, skipping sample")
            return None

        assert len(begin_positions) != 0, "No BEGIN token found, skipping sample"

        # Set labels for assistant response regions (between BEGIN and END tokens)
        for begin_pos, end_pos in zip(begin_positions, end_positions):
            labels[begin_pos : end_pos + 1] = inputs[begin_pos : end_pos + 1]
            # Set EOS token after END token for proper completion
            if end_pos + 1 < len(labels):
                labels[end_pos + 1] = self.EOS_ID

        # Remove BEGIN and END tokens from both inputs and labels
        mask = ~torch.isin(inputs, torch.tensor([self.BEGIN_ID, self.END_ID]))
        inputs = inputs[mask]
        labels = labels[mask]

        # Ensure last token is EOS for proper completion
        labels = torch.cat([labels, torch.tensor([self.EOS_ID], dtype=torch.int64)])

        # Mask out empty responses (consecutive -100 followed by EOS)
        for i in range(1, len(labels)):
            if labels[i - 1] == -100 and labels[i] == self.EOS_ID:
                labels[i] = -100

        # confirm that not all entries in labels are -100
        assert not all(
            labels == -100
        ), "All entries in labels are -100, skipping sample"

        # Handle sequence length - ensure all samples have exactly seq_len length
        if inputs.shape[0] < self.seq_len:
            # Pad inputs to seq_len, and labels to seq_len + 1 (so labels[1:] gives seq_len)
            input_pad_length = self.seq_len - inputs.shape[0]
            label_pad_length = self.seq_len + 1 - labels.shape[0]

            inputs = torch.cat(
                [
                    inputs,
                    torch.full((input_pad_length,), self.PAD_ID, dtype=torch.int64),
                ]
            )
            labels = torch.cat(
                [labels, torch.full((label_pad_length,), -100, dtype=torch.int64)]
            )

            # Apply causal shift: labels are inputs shifted by 1
            return inputs, labels[1:]
        else:
            # Truncate to seq_len (from right side)
            inputs = inputs[-self.seq_len :]
            labels = labels[1 : self.seq_len + 1]
            return inputs, labels

    def _process_and_cache_data(self):
        """Process all samples from the dataset and cache them to .npy files."""
        logger.info("Processing dataset samples for caching...")
        logger.info(f"Expected sequence length: {self.seq_len}")
        logger.info(f"Cache directory: {self.cache_dir}")

        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)

        all_inputs = []
        all_labels = []

        sample_count = 0
        processed_count = 0
        for sample in self._get_data_iter():
            sample_count += 1

            # Process sample to get conversations
            sample_data = self._text_processor(sample)

            # Extract conversations - assuming 'messages' field contains conversation data
            if "messages" not in sample_data:
                logger.warning("Sample missing 'messages' field, skipping")
                continue

            conversations = sample_data["messages"]

            # Build input and labels
            result = self._build_longalign_input(conversations)
            if result is None:
                continue

            inputs, labels = result

            # Convert tensors to numpy and ensure they have the expected shape
            input_np = inputs.numpy()
            label_np = labels.numpy()

            # Verify shapes are as expected
            if input_np.shape[0] != self.seq_len:
                logger.warning(
                    f"Input sample {sample_count} has unexpected shape {input_np.shape}, expected ({self.seq_len},)"
                )
                continue
            if label_np.shape[0] != self.seq_len:
                logger.warning(
                    f"Label sample {sample_count} has unexpected shape {label_np.shape}, expected ({self.seq_len},)"
                )
                continue

            all_inputs.append(input_np)
            all_labels.append(label_np)
            processed_count += 1

            if processed_count % 1000 == 0:
                logger.info(
                    f"Processed {processed_count} valid samples out of {sample_count} total samples..."
                )

        logger.info(
            f"Finished processing {processed_count} valid samples out of {sample_count} total samples. Saving to cache..."
        )

        if processed_count == 0:
            raise ValueError(
                "No valid samples found in the dataset. Cannot create cache."
            )

        # Convert to numpy arrays and save
        try:
            # Verify all samples have the same shape before converting to numpy array
            if all_inputs:
                expected_input_shape = all_inputs[0].shape
                expected_label_shape = all_labels[0].shape

                logger.info(
                    f"Verifying shape consistency for {len(all_inputs)} samples..."
                )
                logger.info(
                    f"Expected shapes: inputs {expected_input_shape}, labels {expected_label_shape}"
                )

                for i, (inp, lbl) in enumerate(zip(all_inputs, all_labels)):
                    if inp.shape != expected_input_shape:
                        raise ValueError(
                            f"Input sample {i} has shape {inp.shape}, expected {expected_input_shape}"
                        )
                    if lbl.shape != expected_label_shape:
                        raise ValueError(
                            f"Label sample {i} has shape {lbl.shape}, expected {expected_label_shape}"
                        )

                logger.info(f"✅ All {len(all_inputs)} samples have consistent shapes")

            logger.info("Converting to numpy arrays...")
            inputs_array = np.array(all_inputs, dtype=np.int64)
            labels_array = np.array(all_labels, dtype=np.int64)
            logger.info(
                f"Final array shapes: inputs {inputs_array.shape}, labels {labels_array.shape}"
            )

            np.save(self.inputs_cache_path, inputs_array)
            np.save(self.labels_cache_path, labels_array)

            logger.info(
                f"Successfully cached {len(inputs_array)} samples to {self.cache_dir}"
            )
            logger.info(
                f"Cache files: {os.path.basename(self.inputs_cache_path)}, {os.path.basename(self.labels_cache_path)}"
            )

            # Store in memory for immediate use
            self.cached_inputs = inputs_array
            self.cached_labels = labels_array
        except Exception as e:
            logger.error(f"Failed to save cached data: {e}")
            # Clean up partial files
            if os.path.exists(self.inputs_cache_path):
                os.remove(self.inputs_cache_path)
            if os.path.exists(self.labels_cache_path):
                os.remove(self.labels_cache_path)
            raise

    def _load_cached_data(self):
        """Load cached processed data from .npy files."""
        try:
            logger.info(f"Loading cached inputs from {self.inputs_cache_path}")
            self.cached_inputs = np.load(self.inputs_cache_path)

            logger.info(f"Loading cached labels from {self.labels_cache_path}")
            self.cached_labels = np.load(self.labels_cache_path)

            # Validate that inputs and labels have the same number of samples
            if len(self.cached_inputs) != len(self.cached_labels):
                raise ValueError(
                    f"Mismatch between cached inputs ({len(self.cached_inputs)}) and labels ({len(self.cached_labels)})"
                )

            logger.info(f"Loaded {len(self.cached_inputs)} cached samples")
        except Exception as e:
            logger.error(f"Failed to load cached data: {e}")
            logger.info("Will fall back to processing data from scratch...")
            # Remove corrupted cache files
            if os.path.exists(self.inputs_cache_path):
                os.remove(self.inputs_cache_path)
            if os.path.exists(self.labels_cache_path):
                os.remove(self.labels_cache_path)
            # Process and cache data
            self._process_and_cache_data()

    def __iter__(self):
        """Iterator that yields input/label pairs for conversation-based training."""
        if self.use_cached_data:
            # Use cached data
            return self._iter_cached_data()
        else:
            # Use original streaming behavior
            return self._iter_streaming_data()

    def _iter_cached_data(self):
        """Iterator for cached data."""
        while True:
            # Start from current sample index (for resuming from checkpoints)
            start_idx = self._sample_idx % len(self.cached_inputs)

            for i in range(start_idx, len(self.cached_inputs)):
                # Get cached sample
                inputs = torch.from_numpy(self.cached_inputs[i]).long()
                labels = torch.from_numpy(self.cached_labels[i]).long()

                if self._sample_idx == 0:
                    logger.info(f"Inputs: {self._tokenizer.decode(inputs)}")
                    logger.info(
                        f"Labels: {self._tokenizer.decode(labels[labels!=-100])}"
                    )

                self._sample_idx += 1
                yield {"input": inputs}, labels

            if not self.infinite:
                logger.warning(f"Dataset {self.dataset_name} has run out of data")
                return
            else:
                # Reset for re-looping
                logger.warning(f"Dataset {self.dataset_name} is being re-looped")
                # Continue the outer while loop with sample_idx reset to 0 by modulo operation

    def _iter_streaming_data(self):
        """Iterator for streaming data (original behavior)."""
        while True:
            for sample in self._get_data_iter():
                # Process sample to get conversations
                sample_data = self._text_processor(sample)

                # Extract conversations - assuming 'messages' field contains conversation data
                if "messages" not in sample_data:
                    logger.warning("Sample missing 'messages' field, skipping")
                    continue

                conversations = sample_data["messages"]

                # Build input and labels
                result = self._build_longalign_input(conversations)
                if result is None:
                    continue

                inputs, labels = result
                if self._sample_idx == 0:
                    logger.info(f"Inputs: {self._tokenizer.decode(inputs)}")
                    logger.info(
                        f"Labels: {self._tokenizer.decode(labels[labels!=-100])}"
                    )
                self._sample_idx += 1

                # Yield the processed sample
                # For LongAlign, we don't shift tokens - input and labels are already properly aligned
                yield {"input": inputs}, labels

            if not self.infinite:
                logger.warning(f"Dataset {self.dataset_name} has run out of data")
                break
            else:
                # Reset for re-looping
                self._sample_idx = 0
                logger.warning(f"Dataset {self.dataset_name} is being re-looped")
                if not isinstance(self._data, Dataset):
                    if hasattr(self._data, "set_epoch") and hasattr(
                        self._data, "epoch"
                    ):
                        self._data.set_epoch(self._data.epoch + 1)

    def load_state_dict(self, state_dict):
        """Load state for checkpointing."""
        self._sample_idx = state_dict.get("sample_idx", 0)

        # For cached data, we don't need to restore the underlying dataset state
        if not self.use_cached_data:
            if isinstance(self._data, Dataset):
                pass  # Map-style dataset state is handled by sample_idx
            else:
                if "data" in state_dict:
                    self._data.load_state_dict(state_dict["data"])

    def state_dict(self):
        """Save state for checkpointing."""
        _state_dict = {"sample_idx": self._sample_idx}

        # For cached data, we only need to save the sample index
        if not self.use_cached_data:
            if isinstance(self._data, Dataset):
                pass  # Map-style dataset state is handled by sample_idx
            else:
                _state_dict["data"] = self._data.state_dict()

        return _state_dict

    def __len__(self):
        """Return the number of samples in the cached dataset."""
        if self.use_cached_data and hasattr(self, "cached_inputs"):
            return len(self.cached_inputs)
        else:
            # For streaming datasets, length is not well-defined
            raise NotImplementedError("Length not available for streaming datasets")

    def get_cache_info(self):
        """Return information about the cached data."""
        if not self.use_cache:
            return {"caching": False}

        info = {
            "caching": True,
            "cache_dir": self.cache_dir,
            "inputs_cache_path": self.inputs_cache_path,
            "labels_cache_path": self.labels_cache_path,
            "cache_exists": os.path.exists(self.inputs_cache_path)
            and os.path.exists(self.labels_cache_path),
        }

        if hasattr(self, "cached_inputs"):
            info.update(
                {
                    "num_samples": len(self.cached_inputs),
                    "input_shape": self.cached_inputs.shape,
                    "labels_shape": self.cached_labels.shape,
                    "cache_size_mb": (
                        self.cached_inputs.nbytes + self.cached_labels.nbytes
                    )
                    / (1024 * 1024),
                }
            )

        return info


def build_hf_dataloader(
    dp_world_size: int,
    dp_rank: int,
    tokenizer: Tokenizer,
    job_config: JobConfig,
    infinite: bool = True,
    split: str = "train",
    cache_dir: str = "/home",
    use_cache: bool = False,
) -> ParallelAwareDataloader:
    """Build a data loader for HuggingFace datasets.

    Args:
        dp_world_size: Number of distributed processes
        dp_rank: Rank of current process
        tokenizer: Tokenizer instance
        job_config: Job configuration
        infinite: Whether to loop dataset infinitely
        split: Dataset split to use
        cache_dir: Directory for caching processed data (LongAlign only)
        use_cache: Whether to use caching (LongAlign only)
    """
    dataset_name = job_config.training.dataset
    dataset_path = job_config.training.dataset_path
    batch_size = job_config.training.batch_size
    seq_len = job_config.training.seq_len

    # Override cache settings from job config if available
    if hasattr(job_config.training, "cache_dir"):
        cache_dir = job_config.training.cache_dir
    if hasattr(job_config.training, "use_cache"):
        use_cache = job_config.training.use_cache

    # Use LongAlignDataset for longalign dataset, otherwise use HuggingFaceDataset
    if dataset_name.lower() == "longalign":
        # Determine model type from tokenizer or config if available
        # Default to "llama" for now - can be made configurable if needed
        model_type = getattr(job_config.model, "name", "llama")

        hf_ds = LongAlignDataset(
            dataset_name=dataset_name,
            dataset_path=dataset_path,
            tokenizer=tokenizer,
            seq_len=seq_len,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            infinite=infinite,
            split=split,
            model_type=model_type,
            cache_dir=cache_dir,
            use_cache=use_cache,
        )
    else:
        hf_ds = HuggingFaceDataset(
            dataset_name=dataset_name,
            dataset_path=dataset_path,
            tokenizer=tokenizer,
            seq_len=seq_len,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            infinite=infinite,
            split=split,
        )

    return ParallelAwareDataloader(
        dataset=hf_ds,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        batch_size=batch_size,
    )

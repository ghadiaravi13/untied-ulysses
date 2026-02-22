# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from dataclasses import dataclass
from typing import Any, Callable

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
    # os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"  # 5 minutes
    # os.environ["REQUESTS_TIMEOUT"] = "300"
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
    # os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"  # 5 minutes
    # os.environ["REQUESTS_TIMEOUT"] = "300"
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
    """Dataset class for LongAlign that handles conversation-based training with masked labels."""

    def __init__(self, model_type: str = "llama", **kwargs):
        super().__init__(**kwargs)
        self.model_type = model_type.lower()

        # Special tokens from longalign_tokenizer.py
        self.BEGIN_TOKEN, self.END_TOKEN = (
            "<|reserved_special_token_247|>",
            "<|reserved_special_token_246|>",
        )  # "<|reserved_special_token_0|>", "<|reserved_special_token_1|>" # for tiktokenizer
        # self._tokenizer.tokenizer.add_tokens([self.BEGIN_TOKEN, self.END_TOKEN])
        self.EOS_ID = self._tokenizer.eos_id
        # Use EOS token as pad token since tiktokenizer doesn't have a dedicated pad token
        self.PAD_ID = self._tokenizer.eos_id

        # # Add special tokens to tokenizer
        # special_tokens = {'cls_token': self.BEGIN_TOKEN, 'sep_token': self.END_TOKEN}
        # self._tokenizer.model.add_special_tokens(special_tokens)

        # Get token IDs for BEGIN and END tokens
        self.BEGIN_ID = self._tokenizer.tokenizer.convert_tokens_to_ids(
            self.BEGIN_TOKEN
        )  # self._tokenizer.special_tokens[self.BEGIN_TOKEN]
        self.END_ID = self._tokenizer.tokenizer.convert_tokens_to_ids(
            self.END_TOKEN
        )  # self._tokenizer.special_tokens[self.END_TOKEN]

        logger.info(f"BEGIN_ID: {self.BEGIN_ID}, END_ID: {self.END_ID}")

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
        if len(conversations) > 2:
            logger.warning(
                f"LongAlignDataset: {len(conversations)} conversations found, skipping"
            )
            return None

        user_input = conversations[0]["content"] if conversations[0]["content"] else ""
        assistant_response = (
            conversations[1]["content"] if conversations[1]["content"] else ""
        )

        if user_input == "" or assistant_response == "":
            logger.warning(
                f"LongAlignDataset: user_input or assistant_response is empty, skipping"
            )
            return None

        num_assistant_tokens = len(
            self._tokenizer.tokenizer.tokenize(assistant_response)
        )
        user_input = "".join(
            self._tokenizer.tokenizer.tokenize(user_input)[
                -(self.seq_len - num_assistant_tokens - 100) :
            ]
        )  # keeping a buffer of 100 tokens for additional prompt
        # Wrap assistant response with special tokens
        assistant_with_tokens = self.BEGIN_TOKEN + assistant_response + self.END_TOKEN

        full_text = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

                        Cutting Knowledge Date: December 2023
                        Today Date: 23 July 2024

                        You are a helpful assistant<|eot_id|><|start_header_id|>user<|end_header_id|>

                        {user_input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

                        {assistant_with_tokens}<|eot_id|>"""

        # for i in range(0, len(conversations), 2):
        #     local_rank = i // 2
        #     user_input = conversations[i]['content'] if conversations[i]['content'] else ''
        #     assistant_response = conversations[i + 1]['content'] if conversations[i + 1]['content'] else ''

        #     # Wrap assistant response with special tokens
        #     assistant_with_tokens = self.BEGIN_TOKEN + assistant_response + self.END_TOKEN

        #     if self.model_type == "chatglm":
        #         formatted_conversations.append(
        #             f'[Round {local_rank + 1}]\n\n问：{user_input}\n\n答：{assistant_with_tokens}'
        #         )
        #     else:  # llama format
        #         formatted_conversations.append(
        #             f'[INST]{user_input}[/INST]{assistant_with_tokens}'
        #         )

        # # Join all conversations
        # full_text = '\n\n'.join(formatted_conversations)

        # Tokenize
        input_tokens = self._tokenizer.tokenizer.encode(
            full_text, add_special_tokens=False
        )
        inputs = torch.tensor(input_tokens, dtype=torch.int64)

        # Create labels - initially all -100 (ignore)
        labels = torch.full_like(inputs, -100)

        # Find BEGIN and END token positions
        # assert False, f"Inputs: {inputs}, BEGIN_ID: {self.BEGIN_ID}, END_ID: {self.END_ID}"
        begin_positions = (inputs == self.BEGIN_ID).nonzero(as_tuple=True)[0].tolist()
        end_positions = (inputs == self.END_ID).nonzero(as_tuple=True)[0].tolist()
        assert len(begin_positions) == len(
            end_positions
        ), f"Mismatch between BEGIN and END tokens, skipping sample: {begin_positions} != {end_positions}"

        if len(begin_positions) != len(end_positions):
            logger.warning("Mismatch between BEGIN and END tokens, skipping sample")
            return None

        assert len(begin_positions) != 0, "No BEGIN token found, skipping sample"

        # Set labels for assistant response regions (between BEGIN and END tokens)
        for begin_pos, end_pos in zip(begin_positions, end_positions):
            labels[begin_pos : end_pos + 1] = inputs[begin_pos : end_pos + 1]
            # # Set EOS token after END token for proper completion
            # if end_pos + 1 < len(labels):
            #     labels[end_pos + 1] = self.EOS_ID

        # Remove BEGIN and END tokens from both inputs and labels
        mask = ~torch.isin(inputs, torch.tensor([self.BEGIN_ID, self.END_ID]))
        inputs = inputs[mask]
        labels = labels[mask]

        # Ensure last token is EOS for proper completion
        # labels = torch.cat([labels, torch.tensor([self.EOS_ID], dtype=torch.int64)])

        # Mask out empty responses (consecutive -100 followed by EOS)
        for i in range(1, len(labels)):
            if labels[i - 1] == -100 and labels[i] == self.EOS_ID:
                labels[i] = -100

        # confirm that not all entries in labels are -100
        assert not all(
            labels == -100
        ), "All entries in labels are -100, skipping sample"

        # Handle sequence length
        if inputs.shape[0] < self.seq_len:

            for i in range(1, 100):
                if inputs.shape[0] < 2**i:
                    input_pad_length = (
                        min(self.seq_len, 2**i) - inputs.shape[0] + 1
                    )  # +1 for shift
                    inputs = torch.cat(
                        [
                            inputs,
                            torch.full(
                                (input_pad_length,), self.PAD_ID, dtype=torch.int64
                            ),
                        ]
                    )
                    labels = torch.cat(
                        [
                            labels,
                            torch.full((input_pad_length,), -100, dtype=torch.int64),
                        ]
                    )
                    if torch.all(labels == -100):
                        logger.warning(
                            f"LongAlignDataset: all labels are -100, skipping"
                        )
                        return None
                    return inputs[:-1], labels[1:]

            # # Need to pad such that after shifting, both have seq_len length
            # # Pad inputs to seq_len, and labels to seq_len + 1 (so labels[1:] gives seq_len)
            # # Apply causal shift: labels are inputs shifted by 1

        else:
            logger.warning(f"LongAlignDataset: inputs.shape[0] > seq_len, skipping")
            return None

    def __iter__(self):
        """Iterator that yields input/label pairs for conversation-based training."""
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
        # LongAlignDataset doesn't use token buffer, so we only restore sample index
        if isinstance(self._data, Dataset):
            self._sample_idx = state_dict.get("sample_idx", 0)
        else:
            if "data" in state_dict:
                self._data.load_state_dict(state_dict["data"])

    def state_dict(self):
        """Save state for checkpointing."""
        _state_dict = {}

        if isinstance(self._data, Dataset):
            _state_dict["sample_idx"] = self._sample_idx
        else:
            _state_dict["data"] = self._data.state_dict()

        return _state_dict


def build_hf_dataloader(
    dp_world_size: int,
    dp_rank: int,
    tokenizer: Tokenizer,
    job_config: JobConfig,
    infinite: bool = True,
    split: str = "train",
) -> ParallelAwareDataloader:
    """Build a data loader for HuggingFace datasets."""
    dataset_name = job_config.training.dataset
    dataset_path = job_config.training.dataset_path
    batch_size = job_config.training.batch_size
    seq_len = job_config.training.seq_len

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

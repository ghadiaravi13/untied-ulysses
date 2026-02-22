# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

import json

import os
from collections.abc import Collection, Iterator, Sequence, Set as AbstractSet
from pathlib import Path
from typing import Any, cast, Literal, Optional, Union

import tiktoken
from tiktoken.load import load_tiktoken_bpe

from tokenizers import AddedToken, Tokenizer as HFTokenizer

from torchtitan.components.tokenizer import Tokenizer
from torchtitan.config_manager import JobConfig
from torchtitan.tools.logging import logger
from transformers import AutoTokenizer
from typing_extensions import override


class TikTokenizer(Tokenizer):
    """
    Tokenizing and encoding/decoding text using the Tiktoken tokenizer.

    Args:
        model_path (str): The path to the Tiktoken model file.
    """

    special_tokens: dict[str, int]

    num_reserved_special_tokens = 256

    pat_str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"  # noqa: E501, B950

    def __init__(self, model_path: str):
        super().__init__()
        assert os.path.exists(
            model_path
        ), f"The tokenizer path does not exist: {model_path}"
        assert os.path.isfile(model_path), model_path

        mergeable_ranks = load_tiktoken_bpe(model_path)
        num_base_tokens = len(mergeable_ranks)
        special_tokens = [
            "<|begin_of_text|>",
            "<|end_of_text|>",
            "<|reserved_special_token_0|>",
            "<|reserved_special_token_1|>",
            "<|reserved_special_token_2|>",
            "<|reserved_special_token_3|>",
            "<|start_header_id|>",
            "<|end_header_id|>",
            "<|reserved_special_token_4|>",
            "<|eot_id|>",  # end of turn
        ] + [
            f"<|reserved_special_token_{i}|>"
            for i in range(5, self.num_reserved_special_tokens - 5)
        ]
        self.special_tokens = {
            token: num_base_tokens + i for i, token in enumerate(special_tokens)
        }
        self.model = tiktoken.Encoding(
            name=Path(model_path).name,
            pat_str=self.pat_str,
            mergeable_ranks=mergeable_ranks,
            special_tokens=self.special_tokens,
        )

        self._n_words: int = self.model.n_vocab
        # BOS / EOS token IDs
        self.bos_id: int = self.special_tokens["<|begin_of_text|>"]
        self.eos_id: int = self.special_tokens["<|end_of_text|>"]
        self.pad_id: int = -1
        self.stop_tokens = {
            self.special_tokens["<|end_of_text|>"],
            self.special_tokens["<|eot_id|>"],
        }
        logger.info(
            f"TikTokenizer built: #words {self.n_words}, BOS ID {self.bos_id}, EOS ID {self.eos_id}"
        )

    def encode(
        self,
        s: str,
        *,
        bos: bool,
        eos: bool,
        allowed_special: Literal["all"] | AbstractSet[str] | None = None,
        disallowed_special: Literal["all"] | Collection[str] | None = None,
    ) -> list[int]:
        """
        Encodes a string into a list of token IDs.

        Args:
            s (str): The input string to be encoded.
            bos (bool): Whether to prepend the beginning-of-sequence token.
            eos (bool): Whether to append the end-of-sequence token.
            allowed_tokens ("all"|set[str]): allowed special tokens in string
            disallowed_tokens ("all"|set[str]): special tokens that raise an error when in string

        Returns:
            list[int]: A list of token IDs.

        By default, setting disallowed_special=() encodes a string by ignoring
        special tokens. Specifically:
        - Setting `disallowed_special` to () will cause all text corresponding
          to special tokens to be encoded as natural text (insteading of raising
          an error).
        - Setting `allowed_special` to "all" will treat all text corresponding
          to special tokens to be encoded as special tokens.
        """
        assert type(s) is str
        allowed_special = allowed_special or set()
        disallowed_special = disallowed_special or ()

        # The tiktoken tokenizer can handle <=400k chars without
        # pyo3_runtime.PanicException.
        TIKTOKEN_MAX_ENCODE_CHARS = 400_000

        # https://github.com/openai/tiktoken/issues/195
        # Here we iterate over subsequences and split if we exceed the limit
        # of max consecutive non-whitespace or whitespace characters.
        MAX_NO_WHITESPACES_CHARS = 25_000

        substrs = (
            substr
            for i in range(0, len(s), TIKTOKEN_MAX_ENCODE_CHARS)
            for substr in self._split_whitespaces_or_nonwhitespaces(
                s[i : i + TIKTOKEN_MAX_ENCODE_CHARS], MAX_NO_WHITESPACES_CHARS
            )
        )
        t: list[int] = []
        for substr in substrs:
            t.extend(
                self.model.encode(
                    substr,
                    allowed_special=allowed_special,
                    disallowed_special=disallowed_special,
                )
            )
        if bos:
            t.insert(0, self.bos_id)
        if eos:
            t.append(self.eos_id)
        return t

    def decode(self, t: Sequence[int]) -> str:
        """
        Decodes a list of token IDs into a string.

        Args:
            t (List[int]): The list of token IDs to be decoded.

        Returns:
            str: The decoded string.
        """
        # Typecast is safe here. Tiktoken doesn't do anything list-related with the sequence.
        return self.model.decode(cast(list[int], t))

    @staticmethod
    def _split_whitespaces_or_nonwhitespaces(
        s: str, max_consecutive_slice_len: int
    ) -> Iterator[str]:
        """
        Splits the string `s` so that each substring contains no more than `max_consecutive_slice_len`
        consecutive whitespaces or consecutive non-whitespaces.
        """
        current_slice_len = 0
        current_slice_is_space = s[0].isspace() if len(s) > 0 else False
        slice_start = 0

        for i in range(len(s)):
            is_now_space = s[i].isspace()

            if current_slice_is_space ^ is_now_space:
                current_slice_len = 1
                current_slice_is_space = is_now_space
            else:
                current_slice_len += 1
                if current_slice_len > max_consecutive_slice_len:
                    yield s[slice_start:i]
                    slice_start = i
                    current_slice_len = 1
        yield s[slice_start:]


class HuggingFaceTokenizer(Tokenizer):
    """
    A tokenizer wrapper that handles BOS/EOS token inference and encoding.

    This class loads tokenizer files and automatically infers BOS/EOS tokens from
    a configuration file (tokenizer_config.json). It provides an encode method that adds
    BOS/EOS tokens based on whether the underlying tokenizer adds them automatically.

    Args:
        tokenizer_path (str): Path to directory containing tokenizer files
    """

    def __init__(
        self,
        tokenizer_path: str,
    ):
        super().__init__()
        self.tokenizer_path = tokenizer_path

        # Initialize BOS/EOS token attributes (frequently used)
        self.bos_id = None
        self.eos_id = None
        self.bos_token = None
        self.eos_token = None

        # Load the underlying tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        self.bos_token = self.tokenizer.bos_token
        self.eos_token = self.tokenizer.eos_token
        self.bos_id = self.tokenizer.bos_token_id
        self.eos_id = self.tokenizer.eos_token_id

        # Load configuration files
        self.config = self._load_config(
            os.path.join(tokenizer_path, "tokenizer_config.json")
        )

        # # Infer special tokens and adding BOS/EOS behavior
        # self._infer_special_tokens()
        # self._infer_should_add_bos_eos()

        self._n_words: int = len(
            self.tokenizer
        )  # gives total vocab size including special tokens

    def _load_config(self, config_path: str) -> Optional[dict]:
        """Load configuration from JSON file if it exists."""
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                return json.load(f)
        return None

    def _load_tokenizer_from_path(self, tokenizer_path: str) -> HFTokenizer:
        """Load tokenizer from various file formats."""
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"Tokenizer path '{tokenizer_path}' does not exist")

        # Define paths for different tokenizer file types
        tokenizer_json_path = os.path.join(tokenizer_path, "tokenizer.json")
        vocab_txt_path = os.path.join(tokenizer_path, "vocab.txt")
        vocab_json_path = os.path.join(tokenizer_path, "vocab.json")
        merges_txt_path = os.path.join(tokenizer_path, "merges.txt")

        # Strategy 1: Load from tokenizer.json (preferred for modern tokenizers)
        if os.path.exists(tokenizer_json_path):
            logger.info("Loading tokenizer from tokenizer.json")
            return HFTokenizer.from_file(tokenizer_json_path)
        # Strategy 2: Load from vocab files (with or without merges.txt)
        elif os.path.exists(vocab_json_path) or os.path.exists(vocab_txt_path):
            # Load vocabulary
            if os.path.exists(vocab_json_path):
                logger.info("Loading vocabulary from vocab.json")
                with open(vocab_json_path, "r") as f:
                    vocab = json.load(f)
                vocab_source = "vocab.json"
            else:
                logger.info("Loading vocabulary from vocab.txt")
                vocab = {}
                with open(vocab_txt_path, "r") as f:
                    for i, line in enumerate(f):
                        token = line.strip()
                        if token:
                            vocab[token] = i
                vocab_source = "vocab.txt"

            # Strategy 2a: Use BPE if merges.txt exists
            if os.path.exists(merges_txt_path):
                logger.info(f"Loading BPE tokenizer from {vocab_source} + merges.txt")
                from tokenizers import decoders, pre_tokenizers, processors
                from tokenizers.models import BPE

                # Load merges from file and convert to tuples
                merges = []
                with open(merges_txt_path, "r") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith(
                            "#"
                        ):  # Skip comments and empty lines
                            parts = line.split()
                            if len(parts) >= 2:
                                merges.append((parts[0], parts[1]))

                # Create BPE model
                bpe_model = BPE(vocab=vocab, merges=merges)
                tokenizer = HFTokenizer(bpe_model)

                # Configure GPT-2 style components for proper space handling
                tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
                    add_prefix_space=False
                )
                tokenizer.decoder = decoders.ByteLevel()
                tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)

                return tokenizer

            # Strategy 2b: Use WordLevel if no merges.txt
            else:
                logger.info(f"Loading WordLevel tokenizer from {vocab_source}")
                from tokenizers.models import WordLevel

                word_level_model = WordLevel(vocab=vocab, unk_token="[UNK]")
                return HFTokenizer(word_level_model)

        else:
            # List available files for debugging
            available_files = [
                f
                for f in os.listdir(tokenizer_path)
                if os.path.isfile(os.path.join(tokenizer_path, f))
            ]
            raise FileNotFoundError(
                f"No supported tokenizer files found in '{tokenizer_path}'. "
                f"Available files: {available_files}. "
                "Looking for: tokenizer.json, tokenizer.model, vocab.txt+merges.txt, or vocab.json+merges.txt"
            )

    def _get_token_from_config(self, config: dict[str, Any], key: str) -> Optional[str]:
        """
        Parse special tokens from config that can be either strings or dicts.
        HF tokens are stored as either {'bos_token': '<bos>'} or {'bos_token': {'content': '<bos>', ...}}.
        """
        token = config.get(key)
        if isinstance(token, dict):
            if "content" not in token:
                raise ValueError(f"Could not parse {key} from config")
            token = token["content"]
        elif token is not None and not isinstance(token, str):
            raise ValueError(
                f"Could not parse {key} from config - expected string or dict"
            )
        return token

    def _process_special_token(
        self, token_str: str, token_config: dict, token_id: Optional[int] = None
    ) -> AddedToken:
        """
        Process a special token and update BOS/EOS attributes if applicable.

        Args:
            token_str: The token string content
            token_config: Token configuration dictionary
            token_id: Optional explicit token ID (for added_tokens_decoder)

        Returns:
            AddedToken object to be added to the tokenizer
        """
        # Get reference BOS/EOS tokens from config for comparison
        config_bos_token = (
            self._get_token_from_config(self.config, "bos_token")
            if self.config
            else None
        )
        config_eos_token = (
            self._get_token_from_config(self.config, "eos_token")
            if self.config
            else None
        )

        # Store BOS/EOS tokens as class attributes if they match
        if token_str == config_bos_token:
            self.bos_token = token_str
            self.bos_id = (
                token_id
                if token_id is not None
                else self.tokenizer.token_to_id(token_str)
            )
        elif token_str == config_eos_token:
            self.eos_token = token_str
            self.eos_id = (
                token_id
                if token_id is not None
                else self.tokenizer.token_to_id(token_str)
            )

        # Create AddedToken object based on config format
        if isinstance(token_config, dict):
            if token_config.get("__type") == "AddedToken" or "content" in token_config:
                # Handle both AddedToken format and added_tokens_decoder format
                return AddedToken(
                    content=token_str,
                    single_word=token_config.get("single_word", False),
                    lstrip=token_config.get("lstrip", False),
                    rstrip=token_config.get("rstrip", False),
                    normalized=token_config.get("normalized", True),
                    special=token_config.get("special", True),
                )

        # Fallback to simple special token
        return AddedToken(content=token_str, special=True)

    def _infer_special_tokens(self):
        """
        Read special tokens from config and add them to the underlying tokenizer.
        Store BOS/EOS tokens as class attributes since they are frequently used.

        This method handles multiple token configuration formats:
        1. Standard top-level keys (bos_token, eos_token, etc.)
        2. added_tokens_decoder dictionary (used by models like Llama 3.1)
        """
        standard_keys = [
            "bos_token",
            "eos_token",
            "pad_token",
            "unk_token",
            "sep_token",
            "cls_token",
            "mask_token",
        ]

        # List to collect AddedToken objects for updating the underlying tokenizer
        added_tokens_to_add = []

        if not self.config:
            return

        # Process standard top-level token keys
        for key in standard_keys:
            token_config = self.config.get(key)
            if token_config is not None:
                token_str = self._get_token_from_config(self.config, key)
                if token_str is not None:
                    added_token = self._process_special_token(token_str, token_config)
                    added_tokens_to_add.append(added_token)

        # Process added_tokens_decoder (comprehensive special token definitions)
        added_tokens_decoder = self.config.get("added_tokens_decoder", {})
        for token_id_str, token_config in added_tokens_decoder.items():
            if isinstance(token_config, dict) and "content" in token_config:
                token_str = token_config["content"]
                token_id = int(token_id_str)
                added_token = self._process_special_token(
                    token_str, token_config, token_id
                )
                added_tokens_to_add.append(added_token)

        # Update the underlying tokenizer with special tokens
        if added_tokens_to_add:
            self.tokenizer.add_special_tokens(added_tokens_to_add)

            # Update BOS/EOS token IDs after adding to tokenizer (in case they changed)
            if self.bos_token:
                self.bos_id = self.tokenizer.token_to_id(self.bos_token)
            if self.eos_token:
                self.eos_id = self.tokenizer.token_to_id(self.eos_token)

    def _infer_should_add_bos_eos(self):
        """
        Determine if we should add BOS/EOS tokens based on config settings.
        If config explicitly specifies add_bos_token/add_eos_token, follow that.
        Otherwise, determine if the underlying tokenizer automatically adds them.
        """
        self.default_add_bos = False
        self.default_add_eos = False
        self.hf_adds_bos = False
        self.hf_adds_eos = False

        # First, determine if underlying tokenizer auto-adds BOS/EOS tokens empirically
        encoded_empty_str = self.tokenizer.encode("").ids
        if self.bos_id is not None and self.bos_id in encoded_empty_str:
            self.hf_adds_bos = True
        if self.eos_id is not None and self.eos_id in encoded_empty_str:
            self.hf_adds_eos = True

        # Check tokenizer_config.json for explicit settings - these override empirical detection
        if self.config:
            config_add_bos = self.config.get("add_bos_token")
            config_add_eos = self.config.get("add_eos_token")
            if config_add_bos is not None:
                self.default_add_bos = bool(config_add_bos)
            if config_add_eos is not None:
                self.default_add_eos = bool(config_add_eos)

    def encode(self, *args, **kwargs) -> list[int]:
        """
        Encode text into token IDs with BOS/EOS handling.

        Args:
            text (str): The text to encode
            add_bos (bool): Whether to add BOS token (if not already added by tokenizer)
            add_eos (bool): Whether to add EOS token (if not already added by tokenizer)

        Returns:
            list[int]: List of token IDs
        """
        # Extract arguments
        if len(args) >= 1:
            text = args[0]
        else:
            text = kwargs.get("text", "")

        add_bos = kwargs.get("add_bos", self.default_add_bos)
        add_eos = kwargs.get("add_eos", self.default_add_eos)
        add_special_tokens = kwargs.get("add_special_tokens", True)

        # Get base token IDs from the underlying tokenizer
        token_ids = self.tokenizer.encode(
            text, add_special_tokens=add_special_tokens
        ).ids

        # Add BOS token if requested and not already added by tokenizer
        if add_bos:
            if self.bos_id is not None:
                token_ids.insert(0, self.bos_id)

        # Add EOS token if requested and not already added by tokenizer
        if add_eos:
            if self.eos_id is not None:
                token_ids.append(self.eos_id)

        return token_ids

    @override
    def decode(self, *args, **kwargs) -> str:
        """
        Decode token IDs back to text.

        Args:
            token_ids (list[int]): List of token IDs to decode
            **kwargs: Additional arguments passed to the underlying tokenizer's decode method
                     (e.g., skip_special_tokens)

        Returns:
            str: Decoded text
        """
        # Extract token_ids from arguments
        if len(args) >= 1:
            token_ids = args[0]
            # Pass through remaining kwargs
            return self.tokenizer.decode(token_ids, **kwargs)
        else:
            token_ids = kwargs.pop("token_ids", [])
            # Pass through remaining kwargs after removing token_ids
            return self.tokenizer.decode(token_ids, **kwargs)

    @property
    def vocab_size(self) -> int:
        """Get the vocabulary size."""
        return self.tokenizer.get_vocab_size()

    def get_vocab_size(self) -> int:
        """Get the vocabulary size."""
        return self.tokenizer.get_vocab_size()

    def get_vocab(self) -> dict[str, int]:
        """Get the vocabulary as a dictionary."""
        return self.tokenizer.get_vocab()

    def token_to_id(self, token: str) -> Optional[int]:
        """Convert token to ID."""
        return self.tokenizer.token_to_id(token)

    def id_to_token(self, token_id: int) -> Optional[str]:
        """Convert ID to token."""
        return self.tokenizer.id_to_token(token_id)


def build_tiktoken_tokenizer(job_config: JobConfig) -> TikTokenizer:
    return TikTokenizer(job_config.model.tokenizer_path)
    # return HuggingFaceTokenizer(job_config.model.tokenizer_path)

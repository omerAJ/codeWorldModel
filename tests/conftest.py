from __future__ import annotations

import sys
from pathlib import Path

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import BertConfig, BertModel, PreTrainedTokenizerFast


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _build_wordlevel_tokenizer() -> PreTrainedTokenizerFast:
    vocab_tokens = [
        "[PAD]",
        "[UNK]",
        "[CLS]",
        "[SEP]",
        "[MASK]",
        "def",
        "for",
        "if",
        "return",
        "numbers",
        "threshold",
        "idx",
        "elem",
        "state",
        "action",
        "context",
        "{",
        "}",
        "(",
        ")",
        "<",
        ">",
        "=",
        ":",
        "\\n",
    ]
    vocab = {token: idx for idx, token in enumerate(vocab_tokens)}
    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()

    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    )


@pytest.fixture()
def tiny_hf_model_dir(tmp_path: Path) -> Path:
    model_dir = tmp_path / "tiny_hf_model"
    model_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = _build_wordlevel_tokenizer()
    tokenizer.save_pretrained(model_dir)

    config = BertConfig(
        vocab_size=len(tokenizer),
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=512,
    )
    model = BertModel(config)
    model.save_pretrained(model_dir)

    return model_dir

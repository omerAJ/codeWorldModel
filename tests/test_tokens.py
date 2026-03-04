from __future__ import annotations

import torch

from cwmodel.config import ModelConfig, TokenConfig
from cwmodel.modeling.hf_setup import load_backbone, load_tokenizer, register_special_tokens


def test_special_tokens_registration_and_embedding_resize(tiny_hf_model_dir) -> None:
    model_cfg = ModelConfig(
        local_path=str(tiny_hf_model_dir),
        trust_remote_code=True,
        use_4bit=False,
        enable_lora=False,
    )
    token_cfg = TokenConfig()

    tokenizer = load_tokenizer(model_cfg)
    original_vocab_size = len(tokenizer)

    added_count, summary_ids = register_special_tokens(tokenizer, token_cfg)
    assert added_count >= 0
    assert len(tokenizer) >= original_vocab_size
    assert all(token_id >= 0 for token_id in summary_ids.values())

    model, _ = load_backbone(model_cfg, device=torch.device("cpu"))
    model.resize_token_embeddings(len(tokenizer))

    embedding = model.get_input_embeddings()
    assert embedding.num_embeddings == len(tokenizer)

    assert tokenizer.convert_tokens_to_ids(token_cfg.wrappers.context_bos) != tokenizer.unk_token_id
    assert tokenizer.convert_tokens_to_ids(token_cfg.wrappers.segment_sep) != tokenizer.unk_token_id

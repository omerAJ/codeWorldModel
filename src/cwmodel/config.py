from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


DEFAULT_LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "query",
    "key",
    "value",
    "dense",
]


@dataclass
class ModelConfig:
    local_path: str = "models/maincoder-maincoder-1b"
    trust_remote_code: bool = True
    use_fast_tokenizer: bool = True
    fix_mistral_regex: bool = True
    enable_lora: bool = True
    use_4bit: bool = True
    gradient_checkpointing: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: list(DEFAULT_LORA_TARGET_MODULES)
    )
    torch_dtype: str = "auto"
    target_eval_mode: bool = True
    predictor_dropout: float = 0.1
    predictor_hidden_multiplier: int = 2


@dataclass
class SummaryTokenConfig:
    context: str = "<CTX_SUM>"
    action: str = "<ACT_SUM>"
    state: str = "<STATE_SUM>"


@dataclass
class WrapperTokenConfig:
    context_bos: str = "<CTX_BOS>"
    action_bos: str = "<ACT_BOS>"
    state_bos: str = "<STATE_BOS>"
    next_action_bos: str = "<NEXT_ACT_BOS>"
    next_state_bos: str = "<NEXT_STATE_BOS>"
    segment_sep: str = "<SEG_SEP>"


@dataclass
class TokenConfig:
    summary: SummaryTokenConfig = field(default_factory=SummaryTokenConfig)
    wrappers: WrapperTokenConfig = field(default_factory=WrapperTokenConfig)


@dataclass
class MaxLengthConfig:
    code_context: int = 768
    action: int = 192
    current_state: int = 256
    next_action: int = 192
    next_state: int = 256


@dataclass
class DataConfig:
    path: Optional[str] = "data/humaneval_transition_1task.jsonl"
    paths: List[str] = field(default_factory=list)
    batch_size: int = 2
    num_workers: int = 0
    shuffle: bool = True
    pin_memory: bool = True
    max_length: MaxLengthConfig = field(default_factory=MaxLengthConfig)

    def resolved_paths(self) -> List[str]:
        if self.paths:
            return [str(p) for p in self.paths]
        if self.path:
            return [str(self.path)]
        raise ValueError("No dataset path configured. Set data.path or data.paths in the config.")


@dataclass
class TrainingConfig:
    epochs: int = 1
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    log_every_steps: int = 1
    save_every_steps: int = 50
    seed: int = 42
    amp: bool = True
    optimizer_beta1: float = 0.9
    optimizer_beta2: float = 0.999
    lr_scheduler: str = "cosine"
    warmup_ratio: float = 0.03


@dataclass
class LoggingConfig:
    tensorboard: bool = True
    jsonl: bool = True


@dataclass
class OutputConfig:
    root_dir: str = "outputs"
    run_name: Optional[str] = None


@dataclass
class TrainConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    tokens: TokenConfig = field(default_factory=TokenConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    @classmethod
    def from_yaml(
        cls, config_path: str | Path, overrides: Optional[List[str]] = None
    ) -> "TrainConfig":
        path = Path(config_path)
        raw: Dict[str, Any] = {}
        if path.exists():
            loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
            if loaded:
                raw = loaded

        for override in overrides or []:
            key, value = _parse_override(override)
            _set_dotted(raw, key, value)

        tokens_raw = raw.get("tokens", {})
        model = ModelConfig(**raw.get("model", {}))
        token_cfg = TokenConfig(
            summary=SummaryTokenConfig(**tokens_raw.get("summary", {})),
            wrappers=WrapperTokenConfig(**tokens_raw.get("wrappers", {})),
        )

        data_raw = raw.get("data", {})
        data_cfg = DataConfig(
            max_length=MaxLengthConfig(**data_raw.get("max_length", {})),
            **{k: v for k, v in data_raw.items() if k != "max_length"},
        )

        return cls(
            model=model,
            tokens=token_cfg,
            data=data_cfg,
            training=TrainingConfig(**raw.get("training", {})),
            logging=LoggingConfig(**raw.get("logging", {})),
            output=OutputConfig(**raw.get("output", {})),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def dump_yaml(self, out_path: str | Path) -> None:
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            yaml.safe_dump(self.to_dict(), sort_keys=False),
            encoding="utf-8",
        )


def _parse_override(raw: str) -> tuple[str, Any]:
    if "=" not in raw:
        raise ValueError(
            f"Invalid override '{raw}'. Use dotted.path=value, e.g. training.epochs=1"
        )
    key, value = raw.split("=", 1)
    return key.strip(), yaml.safe_load(value)


def _set_dotted(data: Dict[str, Any], dotted_key: str, value: Any) -> None:
    cursor = data
    parts = [part for part in dotted_key.split(".") if part]
    if not parts:
        raise ValueError(f"Invalid override key '{dotted_key}'")

    for part in parts[:-1]:
        if part not in cursor or not isinstance(cursor[part], dict):
            cursor[part] = {}
        cursor = cursor[part]

    cursor[parts[-1]] = value

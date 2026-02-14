from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class LLMConfig:
    model_id: str = "openai/gpt-oss-20b"
    cache_dir: Optional[str] = None
    device_map: str = "auto"
    torch_dtype: str = "auto"
    low_cpu_mem_usage: bool = True
    max_new_tokens: int = 2048
    temperature: float = 0.2
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    seed: Optional[int] = None


class LLMClient:
    def __init__(self, cfg: LLMConfig) -> None:
        self.cfg = cfg
        if cfg.seed is not None:
            torch.manual_seed(cfg.seed)

        self.tok = AutoTokenizer.from_pretrained(
            cfg.model_id,
            cache_dir=cfg.cache_dir,
            trust_remote_code=True,
        )
        self.mdl = AutoModelForCausalLM.from_pretrained(
            cfg.model_id,
            cache_dir=cfg.cache_dir,
            torch_dtype=self._resolve_torch_dtype(cfg.torch_dtype),
            device_map=cfg.device_map,
            trust_remote_code=True,
            low_cpu_mem_usage=cfg.low_cpu_mem_usage,
        )

    @staticmethod
    def _resolve_torch_dtype(dtype_name: str) -> Any:
        if dtype_name == "auto":
            return "auto"
        if not hasattr(torch, dtype_name):
            raise ValueError(f"Unsupported torch dtype: {dtype_name}")
        return getattr(torch, dtype_name)

    def _build_model_inputs(self, system: str, user: str) -> Dict[str, torch.Tensor]:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        if hasattr(self.tok, "apply_chat_template"):
            encoded = self.tok.apply_chat_template(
                messages,
                tokenize=True,
                return_tensors="pt",
                add_generation_prompt=True,
            )
            if isinstance(encoded, torch.Tensor):
                return {"input_ids": encoded}
            if isinstance(encoded, dict):
                return {k: v for k, v in encoded.items() if isinstance(v, torch.Tensor)}

        fallback_prompt = f"SYSTEM: {system}\n\nUSER: {user}\nASSISTANT:"
        return self.tok(fallback_prompt, return_tensors="pt")

    def chat(self, system: str, user: str) -> str:
        """Generate assistant response text for the given system/user pair."""
        model_inputs = {
            key: value.to(self.mdl.device)
            for key, value in self._build_model_inputs(system, user).items()
        }

        input_ids = model_inputs["input_ids"]
        input_len = input_ids.shape[1]

        pad_token_id = self.tok.pad_token_id
        if pad_token_id is None:
            pad_token_id = getattr(self.tok, "eos_token_id", None)

        do_sample = self.cfg.temperature > 0
        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": self.cfg.max_new_tokens,
            "eos_token_id": getattr(self.tok, "eos_token_id", None),
            "pad_token_id": pad_token_id,
            "do_sample": do_sample,
        }

        if do_sample:
            gen_kwargs["temperature"] = self.cfg.temperature
            if self.cfg.top_p is not None:
                gen_kwargs["top_p"] = self.cfg.top_p
            if self.cfg.top_k is not None:
                gen_kwargs["top_k"] = self.cfg.top_k

        outputs = self.mdl.generate(**model_inputs, **gen_kwargs)
        generated_ids = outputs[0, input_len:]
        return self.tok.decode(generated_ids, skip_special_tokens=True).strip()

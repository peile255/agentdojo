from __future__ import annotations
import os
import requests
from typing import Dict, Any

class LocalQwen:
    """
    Local-only inference adapter.

    backend=openai_compatible:
        Calls a LOCAL server such as vLLM / SGLang / TGI-compatible gateway.
        No paid API key is used.

    backend=transformers:
        Loads a LOCAL HuggingFace model path directly.
    """

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.backend = os.getenv("QWEN_BACKEND", cfg.get("backend", "openai_compatible"))
        self.model = os.getenv("QWEN_MODEL", cfg.get("model", "qwen36_27b_local"))
        self.base_url = os.getenv("QWEN_BASE_URL", cfg.get("base_url", "http://127.0.0.1:8000/v1")).rstrip("/")
        self.timeout = int(cfg.get("timeout_seconds", 180))
        self.temperature = float(cfg.get("temperature", 0.0))
        self._tokenizer = None
        self._hf_model = None

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        if self.backend == "openai_compatible":
            return self._generate_http(system_prompt, user_prompt)
        if self.backend == "transformers":
            return self._generate_transformers(system_prompt, user_prompt)
        raise ValueError(f"Unknown local Qwen backend: {self.backend}")

    def _generate_http(self, system_prompt: str, user_prompt: str) -> str:
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": int(self.cfg.get("max_new_tokens", 256)),
            "reasoning_effort": self.cfg.get("reasoning_effort", "none"),
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        r = requests.post(url, json=payload, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"]

    def _load_transformers(self):
        if self._hf_model is not None:
            return
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch

        path = os.getenv("QWEN_MODEL_PATH") or self.cfg.get("model_path")
        if not path:
            raise RuntimeError(
                "transformers backend selected but no model_path configured. "
                "Set QWEN_MODEL_PATH or local_model.model_path."
            )

        dtype_cfg = str(self.cfg.get("torch_dtype", "auto"))
        dtype = "auto"
        if dtype_cfg != "auto":
            dtype = getattr(torch, dtype_cfg)

        self._tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        self._hf_model = AutoModelForCausalLM.from_pretrained(
            path,
            device_map=self.cfg.get("device_map", "auto"),
            torch_dtype=dtype,
            trust_remote_code=True,
        )

    def _generate_transformers(self, system_prompt: str, user_prompt: str) -> str:
        self._load_transformers()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer([text], return_tensors="pt").to(self._hf_model.device)
        output_ids = self._hf_model.generate(
            **inputs,
            max_new_tokens=int(self.cfg.get("max_new_tokens", 512)),
            do_sample=self.temperature > 0,
            temperature=max(self.temperature, 1e-5),
        )
        generated = output_ids[:, inputs.input_ids.shape[1]:]
        return self._tokenizer.batch_decode(generated, skip_special_tokens=True)[0]

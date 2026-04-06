#!/usr/bin/env python3
"""
CleanFeed 推理 SDK（任务化版本）
"""
import json
import re
from typing import Dict, List, Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from task_config import DEFAULT_TASK_ORDER


class CleanFeedDetector:
    def __init__(
        self,
        base_model_name: str = "Qwen/Qwen1.5-0.5B-Chat",
        lora_model_path: str = "./generation_model/is_low_quality/final",
        task_id: str = "is_low_quality",
        device: Optional[str] = None,
    ):
        self.task_id = task_id
        if device is None:
            if torch.backends.mps.is_available():
                self.device = "mps"
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        self.model = PeftModel.from_pretrained(self.base_model, lora_model_path)
        self.model = self.model.to(self.device)
        self.model.eval()

    def detect(self, text: str, max_new_tokens: int = 200) -> Dict:
        import time

        start_time = time.perf_counter()
        prompt = f"""请对以下内容执行任务 {self.task_id}，只输出 JSON：
{{"task_id":"{self.task_id}","label":true/false/null,"reason":"原因","evidence":"证据片段","confidence":0.xx}}

内容：
{text[:1200]}
输出："""

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.1,
                do_sample=False,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        response = self.tokenizer.decode(
            outputs[0][len(inputs["input_ids"][0]):],
            skip_special_tokens=True,
        ).strip()

        result = self._parse_response(response)
        result["process_time"] = round(time.perf_counter() - start_time, 4)
        return result

    def _parse_response(self, response: str) -> Dict:
        default_result = {
            "task_id": self.task_id,
            "label": None,
            "reason": "解析失败",
            "evidence": "",
            "confidence": 0.0,
        }
        try:
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if not json_match:
                return default_result
            data = json.loads(json_match.group(0))
            label = data.get("label")
            if label not in [True, False, None]:
                label = None
            return {
                "task_id": data.get("task_id", self.task_id),
                "label": label,
                "reason": data.get("reason", ""),
                "evidence": data.get("evidence", ""),
                "confidence": float(data.get("confidence", 0.0)),
            }
        except Exception:
            return default_result

    def batch_detect(self, texts: List[str], max_workers: int = 4) -> List[Dict]:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            return list(executor.map(self.detect, texts))


if __name__ == "__main__":
    detector = CleanFeedDetector(task_id="is_low_quality")
    text = "加微信xxx领免费资料，都是干货哦"
    print(json.dumps(detector.detect(text), ensure_ascii=False, indent=2))

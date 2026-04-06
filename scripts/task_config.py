#!/usr/bin/env python3
from __future__ import annotations

from typing import Any, Dict

TASK_DEFINITIONS: Dict[str, Dict[str, Any]] = {
    "is_low_quality": {
        "name": "低质量内容",
        "description": "判断内容是否属于广告营销、灌水、标题党、引战或明显低价值内容",
        "positive_label": "是",
        "negative_label": "否",
        "reason_options": [
            "广告营销/引流",
            "无意义灌水/凑字数",
            "色情低俗/暴力",
            "政治敏感/违法",
            "诱导点击/标题党",
            "引战/人身攻击",
            "其他低质量",
        ],
    },
    "is_ai_generated": {
        "name": "AI生成内容",
        "description": "判断内容是否呈现明显的 AI 生成特征",
        "positive_label": "是",
        "negative_label": "否",
        "reason_options": [
            "内容空洞套话",
            "句式结构符合AI特征",
            "事实错误/胡编乱造",
            "其他AI特征",
        ],
    },
}

DEFAULT_TASK_ORDER = list(TASK_DEFINITIONS.keys())


def build_empty_task_annotations() -> Dict[str, Dict[str, Any]]:
    return {
        task_id: {
            "label": None,
            "status": "unlabeled",
            "reason": "",
            "annotator": "",
            "source": "pending",
            "confidence": None,
        }
        for task_id in DEFAULT_TASK_ORDER
    }


def build_task_annotation(
    label: bool | None,
    *,
    reason: str = "",
    annotator: str = "",
    source: str = "human",
    confidence: float | None = None,
) -> Dict[str, Any]:
    if label is None:
        status = "unlabeled"
    elif label:
        status = "labeled_positive"
    else:
        status = "labeled_negative"

    return {
        "label": label,
        "status": status,
        "reason": reason,
        "annotator": annotator,
        "source": source,
        "confidence": confidence,
    }


def derive_primary_label(task_annotations: Dict[str, Dict[str, Any]]) -> str:
    # 仅用于兼容旧链路；训练和标注应直接使用 task_annotations。
    if task_annotations.get("is_ai_generated", {}).get("label") is True:
        return "ai_generated"
    if task_annotations.get("is_low_quality", {}).get("label") is True:
        return "low_quality"
    return "genuine"


def get_task_label(record: Dict[str, Any], task_id: str, field: str = "task_annotations") -> bool | None:
    annotation = record.get(field, {}).get(task_id, {})
    return annotation.get("label")


def is_task_labeled(record: Dict[str, Any], task_id: str, field: str = "task_annotations") -> bool:
    return get_task_label(record, task_id, field) is not None

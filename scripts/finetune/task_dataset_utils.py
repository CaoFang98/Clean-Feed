#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path


def resolve_text_field(df, text_field: str) -> str:
    if text_field != "auto":
        if text_field not in df.columns:
            raise ValueError(f"指定的 text_field 不存在: {text_field}")
        return text_field

    for candidate in ["text", "model_input", "content"]:
        if candidate in df.columns:
            return candidate
    raise ValueError("数据中找不到可用的文本字段，请至少提供 text / model_input / content 之一")


def count_labeled_rows(df, annotation_field: str, task_id: str) -> int:
    if annotation_field not in df.columns:
        return 0
    return int(
        df[annotation_field]
        .apply(lambda x: (x or {}).get(task_id, {}).get("label"))
        .isin([True, False])
        .sum()
    )


def resolve_annotation_field(df, annotation_field: str, task_id: str) -> str:
    if annotation_field != "auto":
        if annotation_field not in df.columns:
            raise ValueError(f"指定的 annotation_field 不存在: {annotation_field}")
        return annotation_field

    candidates = ["task_annotations", "predicted_task_annotations"]
    available = {field: count_labeled_rows(df, field, task_id) for field in candidates}
    best_field = max(available, key=available.get)
    if available[best_field] == 0:
        raise ValueError(
            f"任务 {task_id} 在 task_annotations 和 predicted_task_annotations 中都没有可训练的 true/false 标签"
        )
    return best_field


def load_task_dataframe(data_path: str | Path, task_id: str, annotation_field: str, text_field: str):
    import pandas as pd

    df = pd.read_json(data_path, lines=True)
    resolved_annotation_field = resolve_annotation_field(df, annotation_field, task_id)
    resolved_text_field = resolve_text_field(df, text_field)

    df = df.copy()
    df["text"] = df[resolved_text_field]
    df["task_annotation"] = df[resolved_annotation_field].apply(lambda x: (x or {}).get(task_id, {}))
    df["task_label"] = df["task_annotation"].apply(lambda x: x.get("label"))
    df = df[df["task_label"].isin([True, False])].copy()
    df["task_reason"] = df["task_annotation"].apply(lambda x: x.get("reason", ""))

    label_counts = df["task_label"].value_counts().to_dict()
    positive = int(label_counts.get(True, 0))
    negative = int(label_counts.get(False, 0))
    if positive == 0 or negative == 0:
        raise ValueError(
            f"任务 {task_id} 的可训练样本不足：positive={positive}, negative={negative}。"
            "至少需要正负样本都存在。"
        )

    return df, resolved_annotation_field, resolved_text_field, positive, negative


def split_task_dataframe(df, *, val_size: float, test_size: float, random_state: int):
    from sklearn.model_selection import train_test_split

    if not 0 < val_size < 1:
        raise ValueError("val_size 必须在 0 和 1 之间")
    if not 0 < test_size < 1:
        raise ValueError("test_size 必须在 0 和 1 之间")
    if val_size + test_size >= 1:
        raise ValueError("val_size + test_size 必须小于 1")

    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["task_label"],
    )

    val_relative_size = val_size / (1 - test_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_relative_size,
        random_state=random_state,
        stratify=train_val_df["task_label"],
    )

    return train_df.copy(), val_df.copy(), test_df.copy()


#!/usr/bin/env python3
"""
为单个任务生成固定的 train / val / test 划分。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SCRIPTS_ROOT.parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from task_config import TASK_DEFINITIONS
from finetune.task_dataset_utils import load_task_dataframe, split_task_dataframe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CleanFeed 任务数据固定切分工具")
    parser.add_argument("--data_path", type=Path, required=True)
    parser.add_argument("--task_id", type=str, required=True, choices=sorted(TASK_DEFINITIONS.keys()))
    parser.add_argument(
        "--annotation_field",
        type=str,
        default="auto",
        choices=["auto", "task_annotations", "predicted_task_annotations"],
    )
    parser.add_argument(
        "--text_field",
        type=str,
        default="auto",
        choices=["auto", "text", "model_input", "content"],
    )
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--output_dir", type=Path, default=PROJECT_ROOT / "data" / "splits")
    return parser.parse_args()


def save_split(df, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(
        output_path,
        orient="records",
        lines=True,
        force_ascii=False,
        date_format="iso",
    )


def build_output_path(output_dir: Path, task_id: str, split_name: str) -> Path:
    return output_dir / f"{task_id}_{split_name}.jsonl"


def main():
    args = parse_args()
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 1e-8:
        raise ValueError("train_ratio + val_ratio + test_ratio 必须等于 1")

    df, annotation_field, text_field, positive, negative = load_task_dataframe(
        args.data_path,
        args.task_id,
        args.annotation_field,
        args.text_field,
    )

    train_df, val_df, test_df = split_task_dataframe(
        df,
        val_size=args.val_ratio,
        test_size=args.test_ratio,
        random_state=args.random_state,
    )

    train_path = build_output_path(args.output_dir, args.task_id, "train")
    val_path = build_output_path(args.output_dir, args.task_id, "val")
    test_path = build_output_path(args.output_dir, args.task_id, "test")

    save_split(train_df, train_path)
    save_split(val_df, val_path)
    save_split(test_df, test_path)

    print("=" * 72)
    print("CleanFeed 任务数据固定切分完成")
    print("=" * 72)
    print(f"任务: {args.task_id}")
    print(f"输入文件: {args.data_path}")
    print(f"标签字段: {annotation_field}")
    print(f"文本字段: {text_field}")
    print(f"总样本数: {len(df)} | positive={positive} | negative={negative}")
    print(f"train: {len(train_df)} -> {train_path}")
    print(f"val:   {len(val_df)} -> {val_path}")
    print(f"test:  {len(test_df)} -> {test_path}")


if __name__ == "__main__":
    main()

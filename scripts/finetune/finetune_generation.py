#!/usr/bin/env python3
"""
任务化生成式微调脚本：
针对单个 task_id 训练模型输出最小 JSON: {"label": true/false}
并且只对答案部分计算 loss。
"""
import argparse
import json
import os
import sys
from pathlib import Path

SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SCRIPTS_ROOT.parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from task_config import TASK_DEFINITIONS
from finetune.task_dataset_utils import load_task_dataframe, rebalance_train_dataframe, split_task_dataframe


def check_dependencies():
    try:
        import transformers  # noqa: F401
        import peft  # noqa: F401
        import datasets  # noqa: F401
        import torch  # noqa: F401
        print("✅ 所有依赖已安装")
        return True
    except ImportError as e:
        print(f"❌ 缺少依赖: {e}")
        return False


def parse_args():
    parser = argparse.ArgumentParser(description="CleanFeed 任务化生成式微调脚本")
    parser.add_argument("--data_path", type=str, default="./data/task_labeled_dataset.jsonl")
    parser.add_argument("--train_path", type=str, default=None, help="可选，显式指定训练集 JSONL")
    parser.add_argument("--val_path", type=str, default=None, help="可选，显式指定验证集 JSONL")
    parser.add_argument("--task_id", type=str, default="is_low_quality", choices=sorted(TASK_DEFINITIONS.keys()))
    parser.add_argument(
        "--annotation_field",
        type=str,
        default="auto",
        choices=["auto", "task_annotations", "predicted_task_annotations"],
        help="标签来源字段；auto 会优先选择当前任务下有 true/false 标签的字段",
    )
    parser.add_argument(
        "--text_field",
        type=str,
        default="auto",
        choices=["auto", "text", "model_input", "content"],
        help="训练文本字段；auto 会按 text -> model_input -> content 顺序选择",
    )
    parser.add_argument("--test_size", type=float, default=0.1)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen1.5-0.5B-Chat")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
        help="模型加载精度；auto 会按硬件自动选择",
    )
    parser.add_argument("--local_files_only", action="store_true", help="只从本地缓存或本地目录加载模型")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument(
        "--rebalance_train",
        type=str,
        default="on",
        choices=["on", "off"],
        help="是否只对训练集做轻度重平衡；默认开启",
    )
    parser.add_argument(
        "--max_negative_ratio",
        type=float,
        default=3.0,
        help="训练集重平衡后允许的最大负正样本比，例如 3.0 表示最多 3:1",
    )
    parser.add_argument("--output_dir", type=str, default="./generation_model")
    return parser.parse_args()


def maybe_rebalance_train_df(train_df, *, rebalance_train: str, max_negative_ratio: float):
    if rebalance_train == "off":
        print("📌 训练集重平衡: 关闭")
        return train_df

    balanced_train_df, stats = rebalance_train_dataframe(
        train_df,
        max_negative_ratio=max_negative_ratio,
        random_state=42,
    )
    print(
        "📌 训练集重平衡: "
        f"{stats['strategy']} | "
        f"original positive={stats['original_positive']}, negative={stats['original_negative']} -> "
        f"balanced positive={stats['balanced_positive']}, negative={stats['balanced_negative']} | "
        f"added_positive={stats['added_positive']} | "
        f"max_negative_ratio={max_negative_ratio:g}"
    )
    return balanced_train_df


def load_data(
    data_path: str,
    task_id: str,
    annotation_field: str,
    text_field: str,
    test_size: float,
    rebalance_train: str,
    max_negative_ratio: float,
):
    from datasets import Dataset, DatasetDict
    from sklearn.model_selection import train_test_split

    df, resolved_annotation_field, resolved_text_field, positive, negative = load_task_dataframe(
        data_path,
        task_id,
        annotation_field,
        text_field,
    )

    print(f"📌 标签字段: {resolved_annotation_field}")
    print(f"📌 文本字段: {resolved_text_field}")
    print(f"📌 样本数: {len(df)} | positive={positive} | negative={negative}")

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=42,
        stratify=df["task_label"],
    )
    train_df = maybe_rebalance_train_df(
        train_df,
        rebalance_train=rebalance_train,
        max_negative_ratio=max_negative_ratio,
    )
    return DatasetDict({
        "train": Dataset.from_pandas(train_df),
        "test": Dataset.from_pandas(test_df),
    })


def load_split_data(
    train_path: str,
    val_path: str,
    task_id: str,
    annotation_field: str,
    text_field: str,
    rebalance_train: str,
    max_negative_ratio: float,
):
    from datasets import Dataset, DatasetDict

    train_df, train_annotation_field, train_text_field, train_positive, train_negative = load_task_dataframe(
        train_path,
        task_id,
        annotation_field,
        text_field,
    )
    val_df, val_annotation_field, val_text_field, val_positive, val_negative = load_task_dataframe(
        val_path,
        task_id,
        annotation_field,
        text_field,
    )

    print(f"📌 标签字段: train={train_annotation_field}, val={val_annotation_field}")
    print(f"📌 文本字段: train={train_text_field}, val={val_text_field}")
    print(
        f"📌 训练集: {len(train_df)} | positive={train_positive} | negative={train_negative}"
    )
    print(
        f"📌 验证集: {len(val_df)} | positive={val_positive} | negative={val_negative}"
    )
    train_df = maybe_rebalance_train_df(
        train_df,
        rebalance_train=rebalance_train,
        max_negative_ratio=max_negative_ratio,
    )

    return DatasetDict({
        "train": Dataset.from_pandas(train_df),
        "test": Dataset.from_pandas(val_df),
    })


def resolve_runtime(device: str, dtype: str):
    import torch

    if device == "auto":
        if torch.backends.mps.is_available():
            resolved_device = "mps"
        elif torch.cuda.is_available():
            resolved_device = "cuda"
        else:
            resolved_device = "cpu"
    else:
        resolved_device = device

    if dtype == "auto":
        if resolved_device == "cuda":
            resolved_dtype = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
        elif resolved_device == "mps":
            resolved_dtype = "float16"
        else:
            resolved_dtype = "float32"
    else:
        resolved_dtype = dtype

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    torch_dtype = dtype_map[resolved_dtype]
    training_precision = {
        "bf16": resolved_device == "cuda" and resolved_dtype == "bfloat16",
        "fp16": resolved_device == "cuda" and resolved_dtype == "float16",
    }
    return resolved_device, resolved_dtype, torch_dtype, training_precision


def load_model_and_tokenizer_with_runtime(
    model_name: str,
    *,
    device: str,
    dtype: str,
    local_files_only: bool,
):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    resolved_device, resolved_dtype, torch_dtype, training_precision = resolve_runtime(device, dtype)
    print(f"📌 运行设备: {resolved_device}")
    print(f"📌 加载精度: {resolved_dtype}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch_dtype,
        local_files_only=local_files_only,
    )

    if resolved_device != "cpu":
        model = model.to(resolved_device)

    return model, tokenizer, resolved_device, resolved_dtype, training_precision


def setup_lora(model, lora_r: int, lora_alpha: int, lora_dropout: float):
    from peft import LoraConfig, get_peft_model

    model = get_peft_model(
        model,
        LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    model.print_trainable_parameters()
    return model


def preprocess_data(dataset, tokenizer, task_id: str, max_seq_len: int):
    task_def = TASK_DEFINITIONS[task_id]

    def format_example(example):
        text = example["text"]
        label = bool(example["task_label"])
        prompt = f"""请对以下内容执行任务 {task_id}。
任务说明：{task_def['description']}
只输出 JSON，格式：
{{"label":true/false}}

内容：
{text[:500]}
输出："""
        output = json.dumps({"label": label}, ensure_ascii=False) + tokenizer.eos_token
        return {"prompt_text": prompt, "answer_text": output}

    def tokenize_function(examples):
        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []

        for prompt_text, answer_text in zip(examples["prompt_text"], examples["answer_text"]):
            prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
            answer_ids = tokenizer(answer_text, add_special_tokens=False)["input_ids"]

            if len(answer_ids) >= max_seq_len:
                raise ValueError(
                    f"答案 token 长度 {len(answer_ids)} 超过 max_seq_len={max_seq_len}，"
                    "请增大 max_seq_len。"
                )

            max_prompt_len = max_seq_len - len(answer_ids)
            prompt_ids = prompt_ids[:max_prompt_len]

            input_ids = prompt_ids + answer_ids
            attention_mask = [1] * len(input_ids)
            labels = ([-100] * len(prompt_ids)) + answer_ids

            pad_len = max_seq_len - len(input_ids)
            if pad_len > 0:
                input_ids += [tokenizer.pad_token_id] * pad_len
                attention_mask += [0] * pad_len
                labels += [-100] * pad_len

            batch_input_ids.append(input_ids)
            batch_attention_mask.append(attention_mask)
            batch_labels.append(labels)

        return {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask,
            "labels": batch_labels,
        }

    dataset = dataset.map(format_example, desc=f"格式化 {task_id} 样本")
    tokenized_dataset = dataset.map(tokenize_function, batched=True, desc=f"Tokenize {task_id} 样本")
    tokenized_dataset = tokenized_dataset.remove_columns(
        [c for c in tokenized_dataset["train"].column_names if c not in ["input_ids", "attention_mask", "labels"]]
    )
    return tokenized_dataset


def train(model, tokenizer, tokenized_dataset, args, training_precision: dict):
    from transformers import Trainer, TrainingArguments

    task_output_dir = os.path.join(args.output_dir, args.task_id)
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=task_output_dir,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            num_train_epochs=args.epochs,
            logging_steps=5,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            bf16=training_precision["bf16"],
            fp16=training_precision["fp16"],
            disable_tqdm=False,
            report_to="none",
            optim="adamw_torch",
        ),
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
    )
    trainer.train()
    final_output_dir = os.path.join(task_output_dir, "final")
    model.save_pretrained(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)
    print(f"✅ 模型已保存到: {final_output_dir}")


def main():
    print("=" * 60)
    print("CleanFeed 任务化生成式微调脚本")
    print("=" * 60)
    args = parse_args()

    if not check_dependencies():
        sys.exit(1)

    if bool(args.train_path) != bool(args.val_path):
        raise ValueError("请同时提供 --train_path 和 --val_path，或都不提供")

    if args.train_path and args.val_path:
        dataset = load_split_data(
            args.train_path,
            args.val_path,
            args.task_id,
            args.annotation_field,
            args.text_field,
            args.rebalance_train,
            args.max_negative_ratio,
        )
    else:
        dataset = load_data(
            args.data_path,
            args.task_id,
            args.annotation_field,
            args.text_field,
            args.test_size,
            args.rebalance_train,
            args.max_negative_ratio,
        )
    model, tokenizer, _, _, training_precision = load_model_and_tokenizer_with_runtime(
        args.model_name,
        device=args.device,
        dtype=args.dtype,
        local_files_only=args.local_files_only,
    )
    model = setup_lora(model, args.lora_r, args.lora_alpha, args.lora_dropout)
    tokenized_dataset = preprocess_data(dataset, tokenizer, args.task_id, args.max_seq_len)
    train(model, tokenizer, tokenized_dataset, args, training_precision)


if __name__ == "__main__":
    main()

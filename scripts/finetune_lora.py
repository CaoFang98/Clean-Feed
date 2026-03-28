"""
CleanFeed LoRA 微调脚本

使用说明：
1. 准备数据集（csv 或 jsonl 格式）
2. 配置参数（模型、LoRA 参数等）
3. 租 GPU 服务器（阿里云 T4 / 腾讯云 A10）
4. 运行脚本：python finetune_lora.py
5. 等待 ~1 小时，得到微调后的模型
"""

import os
import sys
import json
import argparse
from typing import Dict, List, Optional

# ==========================================
# 1. 环境检查与依赖安装
# ==========================================

def check_dependencies():
    """检查依赖是否安装"""
    try:
        import transformers
        import peft
        import datasets
        import torch
        print("✅ 所有依赖已安装")
        return True
    except ImportError as e:
        print(f"❌ 缺少依赖: {e}")
        print("请运行: pip install transformers peft datasets torch pandas scikit-learn")
        return False

# ==========================================
# 2. 参数配置
# ==========================================

def parse_args():
    parser = argparse.ArgumentParser(description="CleanFeed LoRA 微调脚本")
    
    # 数据参数
    parser.add_argument("--data_path", type=str, default="./data/data_labeled.csv", 
                        help="数据集路径 (csv 或 jsonl)")
    parser.add_argument("--text_column", type=str, default="text",
                        help="文本列名")
    parser.add_argument("--label_column", type=str, default="label",
                        help="标签列名")
    
    # 模型参数
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen1.5-1.8B-Chat",
                        help="基础模型名称")
    parser.add_argument("--max_seq_len", type=int, default=512,
                        help="最大序列长度")
    
    # LoRA 参数
    parser.add_argument("--lora_r", type=int, default=8,
                        help="LoRA rank (r)")
    parser.add_argument("--lora_alpha", type=int, default=16,
                        help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                        help="LoRA dropout")
    
    # 训练参数
    parser.add_argument("--batch_size", type=int, default=4,
                        help="batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                        help="学习率")
    parser.add_argument("--epochs", type=int, default=15,
                        help="训练轮数")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2,
                        help="梯度累积步数")
    
    # 输出参数
    parser.add_argument("--output_dir", type=str, default="./finetuned_model",
                        help="输出目录")
    
    return parser.parse_args()

# ==========================================
# 3. 数据加载
# ==========================================

def load_data(data_path: str, text_column: str, label_column: str):
    """加载数据集"""
    from datasets import Dataset, DatasetDict
    import pandas as pd
    
    print(f"📥 加载数据: {data_path}")
    
    # 根据扩展名选择加载方式
    if data_path.endswith(".csv"):
        df = pd.read_csv(data_path)
    elif data_path.endswith(".jsonl"):
        df = pd.read_json(data_path, lines=True)
    else:
        raise ValueError(f"不支持的文件格式: {data_path}")
    
    print(f"✅ 数据加载完成，共 {len(df)} 条")
    
    # 检查列是否存在
    if text_column not in df.columns:
        raise ValueError(f"文本列 {text_column} 不存在")
    if label_column not in df.columns:
        raise ValueError(f"标签列 {label_column} 不存在")
    
    # 显示标签分布
    print("\n📊 标签分布:")
    print(df[label_column].value_counts())
    
    # 划分训练集和测试集 (80/20)
    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df[label_column])
    
    print(f"\n✅ 数据集划分: 训练集 {len(train_df)}, 测试集 {len(test_df)}")
    
    # 构造 DatasetDict
    dataset = DatasetDict({
        "train": Dataset.from_pandas(train_df),
        "test": Dataset.from_pandas(test_df)
    })
    
    return dataset

# ==========================================
# 4. 模型与 Tokenizer 加载
# ==========================================

def load_model_and_tokenizer(model_name: str, max_seq_len: int):
    """加载模型和 Tokenizer"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print(f"🤖 加载模型: {model_name}")
    
    # 加载 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载模型 (4-bit 量化，节省显存)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_4bit=True,
        device_map="auto",
        trust_remote_code=True
    )
    
    print("✅ 模型和 Tokenizer 加载完成")
    return model, tokenizer

# ==========================================
# 5. LoRA 配置
# ==========================================

def setup_lora(model, lora_r: int, lora_alpha: int, lora_dropout: float):
    """配置 LoRA"""
    from peft import LoraConfig, get_peft_model
    
    print(f"🔧 配置 LoRA: r={lora_r}, alpha={lora_alpha}")
    
    # LoRA 配置
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # 对于 Qwen，可能需要调整
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # 应用 LoRA
    model = get_peft_model(model, lora_config)
    
    # 打印可训练参数数量
    model.print_trainable_parameters()
    
    return model

# ==========================================
# 6. 数据预处理
# ==========================================

def preprocess_data(dataset, tokenizer, text_column: str, label_column: str, max_seq_len: int):
    """预处理数据，构造格式"""
    
    # 标签映射
    label_to_id = {
        "genuine": 0,
        "low_quality": 1,
        "ai_generated": 2
    }
    
    id_to_label = {v: k for k, v in label_to_id.items()}
    
    def format_example(example):
        """构造提示词格式"""
        text = example[text_column]
        label = example[label_column]
        
        # 构造提示词 (可以根据需要调整)
        prompt = f"""请对以下内容进行分类，类别为：genuine（真实内容）、low_quality（低质量内容）、ai_generated（AI生成内容）。

内容：{text}

类别："""
        
        # 对于训练，我们需要完整的 prompt + completion
        completion = f"{label}"
        
        return {
            "prompt": prompt,
            "completion": completion,
            "label_id": label_to_id.get(label, 0),
            "text": text
        }
    
    def tokenize_function(examples):
        """Tokenize"""
        texts = [p + c for p, c in zip(examples["prompt"], examples["completion"])]
        tokenized = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_seq_len,
            return_tensors="pt"
        )
        
        # 对于因果语言模型，labels = input_ids
        tokenized["labels"] = tokenized["input_ids"].clone()
        
        return tokenized
    
    # 格式化
    print("🔨 格式化数据...")
    dataset = dataset.map(format_example)
    
    # Tokenize
    print("🔢 Tokenizing...")
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    # 只保留需要的列
    tokenized_dataset = tokenized_dataset.remove_columns(
        [c for c in tokenized_dataset["train"].column_names 
         if c not in ["input_ids", "attention_mask", "labels"]]
    )
    
    print("✅ 数据预处理完成")
    return tokenized_dataset, id_to_label

# ==========================================
# 7. 训练
# ==========================================

def train(model, tokenized_dataset, args):
    """训练模型"""
    from transformers import Trainer, TrainingArguments
    
    print(f"🚀 开始训练，共 {args.epochs} 轮")
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        fp16=True,  # 混合精度训练
        report_to="none",
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
    )
    
    # 开始训练
    trainer.train()
    
    print("✅ 训练完成！")
    
    # 保存 LoRA 权重
    final_output_dir = os.path.join(args.output_dir, "final")
    model.save_pretrained(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)
    
    print(f"✅ 模型已保存到: {final_output_dir}")
    return trainer

# ==========================================
# 8. 评估（可选）
# ==========================================

def evaluate(trainer, tokenized_dataset, id_to_label):
    """评估模型"""
    print("📊 开始评估...")
    
    # 预测
    predictions = trainer.predict(tokenized_dataset["test"])
    
    # 这里可以添加更详细的评估逻辑
    # 计算准确率、精确率、召回率等
    
    print("✅ 评估完成")

# ==========================================
# 主函数
# ==========================================

def main():
    print("=" * 60)
    print("CleanFeed LoRA 微调脚本")
    print("=" * 60)
    
    # 解析参数
    args = parse_args()
    
    # 检查依赖
    if not check_dependencies():
        sys.exit(1)
    
    # 加载数据
    dataset = load_data(args.data_path, args.text_column, args.label_column)
    
    # 加载模型和 Tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model_name, args.max_seq_len)
    
    # 配置 LoRA
    model = setup_lora(model, args.lora_r, args.lora_alpha, args.lora_dropout)
    
    # 预处理数据
    tokenized_dataset, id_to_label = preprocess_data(
        dataset, tokenizer, args.text_column, args.label_column, args.max_seq_len
    )
    
    # 训练
    trainer = train(model, tokenized_dataset, args)
    
    # 评估
    evaluate(trainer, tokenized_dataset, id_to_label)
    
    print("\n" + "=" * 60)
    print("🎉 所有步骤完成！")
    print("=" * 60)

if __name__ == "__main__":
    main()

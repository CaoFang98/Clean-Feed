# 数据集指南

本指南说明 CleanFeed 当前使用的数据结构、样本来源、清洗流程、打标流程和训练输入约定。

## 目标

CleanFeed 的数据链路服务于两个核心目标：

- 让浏览器插件和训练数据使用同一份输入视角
- 让每条样本可以同时承载多个独立任务标签

当前默认任务：

- `is_low_quality`
- `is_ai_generated`

## 样本结构

清洗后的主样本文件为 `data/zhihu_cleaned_data.jsonl`。每条样本包含以下核心字段：

```json
{
  "sample_id": "7f5a2e91f6a8b2c1",
  "question": "为什么山火一定要灭，不能让它自己烧完吗？",
  "model_input": "问题：为什么山火一定要灭，不能让它自己烧完吗？\n回答预览：建议了解一下1987年大兴安岭特大森林火灾……",
  "answer_preview": "建议了解一下1987年大兴安岭特大森林火灾……",
  "answer_full_clean": "建议了解一下1987年大兴安岭特大森林火灾。持续燃烧了28个昼夜……",
  "task_annotations": {
    "is_low_quality": {
      "label": null,
      "status": "unlabeled",
      "reason": "",
      "annotator": "",
      "source": "pending",
      "confidence": null
    },
    "is_ai_generated": {
      "label": null,
      "status": "unlabeled",
      "reason": "",
      "annotator": "",
      "source": "pending",
      "confidence": null
    }
  }
}
```

## 输入视角

训练和插件推理共用统一输入：

```text
问题：{question}
回答预览：{answer_preview}
```

字段说明：

- `answer_full_clean`
  - 清洗后的完整回答
  - 用于人工参考
- `answer_preview`
  - 模拟插件在 Feed 中可见的回答片段
- `model_input`
  - 训练和推理的统一输入
- `task_annotations`
  - 当前样本的任务化标签

## 数据来源

### 原始采集文件

- `data/zhihu_raw_data.jsonl`

主要字段包括：

- `question`
- `question_detail`
- `answer_full`
- `answer_truncated`
- `author`
- `votes`
- `comment_count`
- `platform`
- `url`
- `collected_at`

### 清洗后样本文件

- `data/zhihu_cleaned_data.jsonl`

这是后续自动打标、人工复核和训练的主输入文件。

## 清洗流程

清洗脚本：

- `scripts/label/clean_data.py`

运行方式：

```bash
python3 scripts/label/clean_data.py
```

当前清洗规则：

- 解 HTML 实体
- 去除 HTML 标签
- 去除零宽字符
- 压缩空白字符
- 过滤过短回答
- 生成 `answer_preview`
- 生成 `model_input`
- 初始化空的 `task_annotations`

清洗流程不删除广告词、引流词、AI 写作痕迹等语义信号。

## 标注结构

每个任务使用统一标注结构：

```json
{
  "label": true,
  "status": "labeled_positive",
  "reason": "广告营销/引流",
  "annotator": "human_default",
  "source": "human",
  "confidence": 1.0
}
```

字段说明：

- `label`
  - `true`
  - `false`
  - `null`
- `status`
  - `unlabeled`
  - `labeled_positive`
  - `labeled_negative`
- `reason`
  - 当前任务下的命中原因
- `annotator`
  - 标注人或模型名
- `source`
  - `human`
  - `auto`
  - `human_reviewed_auto`
  - `pending`
- `confidence`
  - 自动打标或人工确认的置信信息

## 自动打标

自动打标入口：

- `scripts/label/auto_labeler.py`

示例：

```bash
python3 scripts/label/auto_labeler.py \
  --provider local \
  --input-path data/zhihu_cleaned_data.jsonl \
  --output-path data/task_prelabeled_dataset.jsonl \
  --text-field model_input \
  --output-format hybrid
```

支持的 provider：

- `local`
- `doubao`
- `qwen`
- `openai`

输出模式：

- `review`
  - 只写 `predicted_task_annotations`
- `train`
  - 直接写 `task_annotations`
- `hybrid`
  - 同时写 `predicted_task_annotations` 和 `task_annotations`

## 人工复核

人工复核入口：

- `scripts/label/manual_labeler.py`

示例：

```bash
python3 scripts/label/manual_labeler.py \
  --input-path data/zhihu_cleaned_data.jsonl \
  --prelabel-path data/task_prelabeled_dataset.jsonl \
  --output-path data/task_labeled_dataset.jsonl
```

人工复核器会展示：

- `model_input`
- `answer_preview`
- `answer_full_clean`
- 自动打标建议

并按 `sample_id` 自动跳过已完成样本。

## 训练数据约定

训练时按单个任务切片：

- 训练 `is_low_quality`
  - 只取 `task_annotations.is_low_quality.label in {true, false}` 的样本
- 训练 `is_ai_generated`
  - 只取 `task_annotations.is_ai_generated.label in {true, false}` 的样本

这允许后续继续增加新任务，而不需要改动整体数据结构。

## 常用文件

- `data/zhihu_raw_data.jsonl`
- `data/zhihu_cleaned_data.jsonl`
- `data/task_prelabeled_dataset.jsonl`
- `data/task_labeled_dataset.jsonl`
- `scripts/task_config.py`
- `scripts/label/clean_data.py`
- `scripts/label/auto_labeler.py`
- `scripts/label/manual_labeler.py`

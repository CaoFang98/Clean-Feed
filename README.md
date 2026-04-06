# CleanFeed

CleanFeed 是一个面向知乎、小红书等信息流场景的内容过滤实验项目。项目包含浏览器插件、本地后端服务、数据清洗与标注脚本、任务化微调脚本，以及基于向量检索的 RAG 判断链路。

## 功能概览

- 浏览器插件扫描页面卡片内容并应用过滤效果
- 支持知乎、小红书两个平台
- 默认任务包括：
  - `is_low_quality`
  - `is_ai_generated`
- 任务结果使用统一结构 `task_results`
- 数据链路支持：
  - 原始采集
  - 清洗与输入构造
  - 自动打标
  - 人工复核
  - 按任务训练
  - Benchmark 评估

## 项目结构

```text
clean-feed/
├── src/                        # 浏览器插件源码
├── backend/                    # FastAPI 后端与 RAG 逻辑
├── data/                       # 数据文件与中间产物
├── docs/                       # 数据与标注规范
├── scripts/
│   ├── crawl_zhihu.py          # 知乎采集脚本
│   ├── inference_sdk.py        # 本地 LoRA 推理实验工具
│   ├── task_config.py          # 任务定义与共享标注结构
│   ├── label/
│   │   ├── clean_data.py       # 原始数据清洗与训练输入构造
│   │   ├── auto_labeler.py     # 自动打标
│   │   └── manual_labeler.py   # 人工复核
│   ├── finetune/
│   │   ├── finetune_lora.py
│   │   └── finetune_generation.py
│   └── benchmark/
│       ├── benchmark_models.py
│       ├── benchmark_concurrency.py
│       └── unified_benchmark.py
├── manifest.json
└── package.json
```

## 核心概念

### 任务化标注

每条样本都可以同时拥有多个独立任务标签，而不是只能属于一个互斥类别。当前默认任务定义在 [scripts/task_config.py](./scripts/task_config.py)。

典型样本结构如下：

```json
{
  "sample_id": "7f5a2e91f6a8b2c1",
  "model_input": "问题：...\n回答预览：...",
  "task_annotations": {
    "is_low_quality": {
      "label": true,
      "status": "labeled_positive",
      "reason": "广告营销/引流",
      "annotator": "human_default",
      "source": "human",
      "confidence": 1.0
    },
    "is_ai_generated": {
      "label": false,
      "status": "labeled_negative",
      "reason": "",
      "annotator": "human_default",
      "source": "human",
      "confidence": 1.0
    }
  }
}
```

### 统一输入视角

训练和插件推理共用统一输入：

```text
问题：{question}
回答预览：{answer_preview}
```

其中：

- `answer_full_clean` 用于人工参考
- `answer_preview` 表示插件应看到的预览内容
- `model_input` / `content` 是模型实际使用的文本

## 安装

### 前端依赖

```bash
npm install
```

### 后端依赖

```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install requests
```

### 脚本常用依赖

如果需要运行采集、自动打标、微调或 benchmark，通常还需要按用途安装这些依赖：

```bash
pip install requests tqdm tiktoken playwright
pip install transformers peft datasets torch pandas scikit-learn psutil numpy
playwright install chromium
```

## 启动方式

### 启动浏览器插件开发环境

```bash
npm run dev
```

### 构建浏览器插件

```bash
npm run build
```

构建结果输出到 `dist/`，可在浏览器扩展管理页中加载。

### 启动后端服务

```bash
cd backend
source venv/bin/activate
python app.py
```

服务默认监听：

- `http://127.0.0.1:8765`

健康检查接口：

- `GET /health`

分类接口：

- `POST /classify`

请求示例：

```json
{
  "text": "问题：为什么山火一定要灭？\n回答预览：建议了解一下1987年大兴安岭特大森林火灾……",
  "platform": "zhihu",
  "task_ids": ["is_low_quality", "is_ai_generated"]
}
```

响应中的主字段包括：

- `task_results`
- `is_low_quality`
- `is_ai_generated`
- `confidence`
- `label`
- `reason`
- `detect_method`

## 数据工作流

### 1. 采集原始数据

```bash
python3 scripts/crawl_zhihu.py
```

输出文件：

- `data/zhihu_raw_data.jsonl`

### 2. 清洗数据并构造训练输入

```bash
python3 scripts/label/clean_data.py
```

输出文件：

- `data/zhihu_cleaned_data.jsonl`

主要处理逻辑：

- 解 HTML 实体
- 去除 HTML 标签
- 归一化空白
- 过滤过短回答
- 生成 `answer_preview`
- 生成 `model_input`
- 初始化空的 `task_annotations`

### 3. 自动打标

统一入口：

- [scripts/label/auto_labeler.py](./scripts/label/auto_labeler.py)

本地模型示例：

```bash
python3 scripts/label/auto_labeler.py \
  --provider local \
  --input-path data/zhihu_cleaned_data.jsonl \
  --output-path data/task_prelabeled_dataset.jsonl \
  --text-field model_input \
  --output-format hybrid
```

远程 API 示例：

```bash
python3 scripts/label/auto_labeler.py \
  --provider doubao \
  --input-path data/zhihu_cleaned_data.jsonl \
  --output-path data/task_prelabeled_dataset.jsonl \
  --text-field model_input \
  --output-format hybrid
```

常用参数：

- `--provider`
  - `local`
  - `doubao`
  - `qwen`
  - `openai`
- `--input-path`
- `--output-path`
- `--text-field`
- `--limit`
- `--request-delay`
- `--review-threshold`
- `--output-format`
  - `review`
  - `train`
  - `hybrid`

输出模式说明：

- `review`
  - 只写 `predicted_task_annotations`
  - 适合后续人工复核
- `train`
  - 直接写 `task_annotations`
  - 适合把自动标签直接作为训练数据
- `hybrid`
  - 同时保留 `predicted_task_annotations` 和 `task_annotations`

### 4. 人工复核

```bash
python3 scripts/label/manual_labeler.py \
  --input-path data/zhihu_cleaned_data.jsonl \
  --prelabel-path data/task_prelabeled_dataset.jsonl \
  --output-path data/task_labeled_dataset.jsonl
```

人工复核器支持：

- 按 `sample_id` 自动跳过已完成样本
- 展示 `model_input`
- 展示 `answer_preview`
- 展示 `answer_full_clean`
- 展示自动打标建议
- 直接接受自动打标结果或手动修改

### 5. 比较清洗前后的自动打标差异

```bash
python3 scripts/label/auto_labeler.py \
  --provider local \
  --input-path data/zhihu_raw_data.jsonl \
  --output-path data/zhihu_raw_task_predictions.jsonl \
  --text-field answer_full \
  --output-format review

python3 scripts/label/auto_labeler.py \
  --provider local \
  --input-path data/zhihu_cleaned_data.jsonl \
  --output-path data/zhihu_clean_task_predictions.jsonl \
  --text-field model_input \
  --output-format review
```

## 微调

### LoRA 二分类训练

```bash
python3 scripts/finetune/finetune_lora.py \
  --data_path data/task_labeled_dataset.jsonl \
  --task_id is_low_quality
```

### 生成式任务训练

```bash
python3 scripts/finetune/finetune_generation.py \
  --data_path data/task_labeled_dataset.jsonl \
  --task_id is_ai_generated
```

训练脚本按 `task_id` 过滤样本，因此可以只针对单个任务训练模型。

## Benchmark

### 模型效果对比

```bash
python3 scripts/benchmark/benchmark_models.py
```

### 并发性能测试

```bash
python3 scripts/benchmark/benchmark_concurrency.py
```

### 综合评估

```bash
python3 scripts/benchmark/unified_benchmark.py
```

## 关键文件

- [backend/app.py](./backend/app.py)
  - 后端分类接口
  - 本地模型调用
  - RAG 检索与兜底判断
- [scripts/task_config.py](./scripts/task_config.py)
  - 任务定义
  - 标注结构生成函数
- [docs/DATASET.md](./docs/DATASET.md)
  - 数据集结构与打标流程
- [docs/QUALITY_STANDARD.md](./docs/QUALITY_STANDARD.md)
  - 质量判断标准

## 注意事项

- 浏览器插件和训练数据使用同一份 `model_input` 视角，避免训练/推理输入不一致
- 自动打标调用远程 API 时，需要提前设置对应的 API Key 环境变量
- 后端服务默认依赖本地 Ollama，可通过环境变量调整模型与阈值
- `seed_data.json` 用于后端向量库初始化，适合作为少量高质量示例集合
- `scripts/__pycache__/` 为 Python 缓存目录，不属于项目逻辑

## License

MIT

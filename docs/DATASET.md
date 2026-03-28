# 数据集构建指南

本指南说明如何构建 CleanFeed 项目的训练数据集。

---

## 📋 目录

- [数据规模](#数据规模)
- [数据来源](#数据来源)
- [标注规范](#标注规范)
- [数据格式](#数据格式)
- [LLM 批量打标脚本](#llm-批量打标脚本)

---

## 📊 数据规模

- **总样本数**：200-500 条
- **类别分布**：
  - `genuine`（真实内容）：40%
  - `low_quality`（低质量内容）：35%
  - `ai_generated`（AI生成内容）：25%

---

## 🌐 数据来源

### 1. 小红书

- **探索页/推荐页** - 自然浏览，复制笔记内容
- **搜索页** - 搜索关键词，复制结果
- **注意**：只复制公开内容，尊重版权

### 2. 知乎

- **推荐页** - 自然浏览，复制回答/文章
- **热榜** - 复制热榜内容
- **话题页** - 从不同话题复制

### 3. AI 生成内容（可选）

可以用 LLM 生成一些样本作为辅助：

```
请生成 5 条小红书风格的笔记，内容是关于美妆/穿搭/美食的
```

---

## 🏷️ 标注规范

### 三分类标注

| 标签 | 说明 | 示例 |
|------|------|------|
| `genuine` | 真实、高质量内容 | "今天去了西湖，人真的很多..." |
| `low_quality` | 低质量、水帖、营销 | "家人们谁懂啊！这个绝绝子..." |
| `ai_generated` | AI 生成内容 | "随着人工智能技术的发展..." |

### 标注原则

1. **优先判断**：如果同时符合多个，按 `ai_generated` > `low_quality` > `genuine` 优先级
2. **模糊案例**：拿不准的标 `genuine`，宁缺毋滥
3. **边界感**：不要过度标注，明显有问题的才标

---

## 📄 数据格式

### CSV 格式（推荐，简单易编辑）

```csv
text,label,platform,notes
"今天去了西湖...","genuine","xiaohongshu",""
"家人们谁懂啊...","low_quality","xiaohongshu",""
"随着人工智能的发展...","ai_generated","zhihu",""
```

### JSONL 格式（HuggingFace 标准）

```jsonl
{"text": "今天去了西湖...", "label": "genuine", "platform": "xiaohongshu"}
{"text": "家人们谁懂啊...", "label": "low_quality", "platform": "xiaohongshu"}
{"text": "随着人工智能的发展...", "label": "ai_generated", "platform": "zhihu"}
```

---

## 🤖 LLM 批量打标脚本

这是一个简单的 Python 脚本，可以用 LLM 批量打标，然后人工校验。

```python
import json
import time
from typing import List, Dict
from openai import OpenAI

# 配置你的 API（DeepSeek / OpenAI / 通义千问都可以）
client = OpenAI(
    api_key="your_api_key_here",
    base_url="https://api.deepseek.com/v1"  # 或其他兼容端点
)

def classify_with_llm(text: str) -> Dict:
    """用 LLM 分类单条文本"""
    
    prompt = f"""你是一个内容分类助手，请将以下内容分类为三类之一：
- genuine: 真实、高质量的内容
- low_quality: 低质量、水帖、营销内容
- ai_generated: AI生成的内容

只返回 JSON，格式：{{"label": "xxx", "confidence": 0.xx, "reason": "简短原因"}}

要分类的内容：
{text}
"""
    
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        return {
            "label": result.get("label", "genuine"),
            "confidence": result.get("confidence", 0.5),
            "reason": result.get("reason", ""),
            "text": text
        }
    except Exception as e:
        print(f"Error: {e}")
        return {"label": "genuine", "confidence": 0.5, "reason": "failed", "text": text}

def batch_classify(texts: List[str], delay: float = 1.0) -> List[Dict]:
    """批量分类"""
    results = []
    for i, text in enumerate(texts):
        print(f"Classifying {i+1}/{len(texts)}...")
        result = classify_with_llm(text)
        results.append(result)
        time.sleep(delay)  # 避免触发速率限制
    return results

# 使用示例
if __name__ == "__main__":
    # 你的未标注数据
    raw_texts = [
        "今天去了西湖，人真的很多...",
        "家人们谁懂啊！这个绝绝子...",
        # ... 更多数据
    ]
    
    # 批量分类
    results = batch_classify(raw_texts)
    
    # 保存结果
    with open("data_labeled.jsonl", "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    
    print("Done! Now please review the labels manually.")
```

### 使用流程

1. **收集原始数据** - 把收集到的文本放到一个列表里
2. **LLM 批量打标** - 运行上面的脚本
3. **人工校验** - 逐条检查 LLM 的标注，修正错误
4. **导出数据集** - 保存为 csv 或 jsonl 格式

---

## ✅ 质量检查

标注完成后，检查：

- [ ] 类别分布基本合理（不要某一类太少）
- [ ] 边界案例都检查过了
- [ ] 文本长度适中（太短/太长的可以过滤）
- [ ] 没有重复数据
- [ ] 格式正确，没有乱码

---

准备好数据集后，就可以开始微调了！详见 [scripts/finetune_lora.py](../scripts/finetune_lora.py)

from pathlib import Path
from typing import Dict, List, Optional

import chromadb
import json
import os
import re
import requests
import traceback
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

app = FastAPI(title="CleanFeed API")

CORS_ORIGIN_REGEX = os.getenv(
    "CLEANFEED_CORS_ORIGIN_REGEX",
    r"^(chrome-extension://.*|http://localhost(:\d+)?)$",
)
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=CORS_ORIGIN_REGEX,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

TASK_DEFINITIONS = {
    "is_low_quality": {
        "description": "是否属于低质量内容，包括广告营销、灌水、引战、标题党等",
    },
    "is_ai_generated": {
        "description": "是否呈现明显 AI 生成特征",
    },
}
DEFAULT_TASK_IDS = list(TASK_DEFINITIONS.keys())

print("[CleanFeed] Loading embedding model...")
try:
    embedding_model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
    print("[CleanFeed] Embedding model loaded successfully!")
except Exception as e:
    print(f"[CleanFeed] Warning: Failed to load embedding model: {e}")
    embedding_model = None

OLLAMA_ENABLED = os.getenv("OLLAMA_ENABLED", "true").lower() == "true"
OLLAMA_API = os.getenv("OLLAMA_API", "http://localhost:11434/api/chat")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:8b")
LOCAL_MODEL_CONFIDENCE_THRESHOLD = float(os.getenv("OLLAMA_CONFIDENCE_THRESHOLD", "0.9"))

print(f"[CleanFeed] Local model enabled: {OLLAMA_ENABLED}, model: {OLLAMA_MODEL}")

CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = chroma_client.get_or_create_collection(name="content_examples")

SEED_DATA_PATH = Path(__file__).parent / "seed_data.json"
try:
    with open(SEED_DATA_PATH, "r", encoding="utf-8") as f:
        SAMPLE_DATA = json.load(f)
    print(f"[CleanFeed] Loaded {len(SAMPLE_DATA)} seed examples from {SEED_DATA_PATH}")
except Exception as e:
    print(f"[CleanFeed] Warning: Failed to load seed data: {e}")
    SAMPLE_DATA = []


class TaskPrediction(BaseModel):
    label: Optional[bool] = None
    confidence: float = 0.0
    reason: str = ""


class ClassifyRequest(BaseModel):
    text: str
    platform: Optional[str] = None
    task_ids: Optional[List[str]] = None


class ClassifyResponse(BaseModel):
    task_results: Dict[str, TaskPrediction]
    is_low_quality: bool
    is_ai_generated: bool
    confidence: float
    label: str
    reason: str
    rag_example: Optional[str] = None
    detect_method: Optional[str] = None


def empty_task_result() -> Dict[str, TaskPrediction]:
    return {
        task_id: TaskPrediction(label=None, confidence=0.0, reason="")
        for task_id in DEFAULT_TASK_IDS
    }


def derive_primary_label(task_results: Dict[str, TaskPrediction]) -> str:
    # 仅用于兼容旧前端字段；主返回结构应消费 task_results。
    if task_results["is_ai_generated"].label is True:
        return "ai_generated"
    if task_results["is_low_quality"].label is True:
        return "low_quality"
    return "genuine"


def derive_primary_reason(task_results: Dict[str, TaskPrediction]) -> str:
    reasons = []
    for task_id, result in task_results.items():
        if result.label is True and result.reason:
            reasons.append(f"{task_id}: {result.reason}")
    return " | ".join(reasons) if reasons else "Content appears normal"


def derive_primary_confidence(task_results: Dict[str, TaskPrediction]) -> float:
    return round(max(result.confidence for result in task_results.values()), 2)


def normalize_seed_annotations(item: Dict) -> Dict[str, Dict]:
    if "task_annotations" in item:
        return item["task_annotations"]

    # Backward compatibility for older seed_data entries.
    label = item.get("label")
    return {
        "is_low_quality": {
            "label": label == "low_quality",
            "reason": item.get("category", ""),
        },
        "is_ai_generated": {
            "label": label == "ai_generated",
            "reason": item.get("category", ""),
        },
    }


def init_knowledge_base():
    print("[CleanFeed] Initializing knowledge base...")
    if collection.count() > 0:
        print(f"[CleanFeed] Knowledge base already has {collection.count()} examples")
        return

    documents = [d["text"] for d in SAMPLE_DATA]
    metadatas = []
    ids = []
    for idx, item in enumerate(SAMPLE_DATA):
        annotations = normalize_seed_annotations(item)
        metadatas.append({
            "category": item.get("category", ""),
            "is_low_quality": int(bool(annotations["is_low_quality"]["label"])),
            "is_ai_generated": int(bool(annotations["is_ai_generated"]["label"])),
        })
        ids.append(f"example_{idx}")

    collection.add(documents=documents, metadatas=metadatas, ids=ids)
    print(f"[CleanFeed] Added {len(SAMPLE_DATA)} examples to knowledge base")


def rag_retrieve(text: str, top_k: int = 3) -> List[Dict]:
    try:
        results = collection.query(query_texts=[text], n_results=top_k)
        examples = []
        if results["documents"] and results["documents"][0]:
            for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
                examples.append({
                    "text": doc,
                    "category": meta.get("category", ""),
                    "is_low_quality": bool(meta.get("is_low_quality", 0)),
                    "is_ai_generated": bool(meta.get("is_ai_generated", 0)),
                })
        return examples
    except Exception as e:
        print(f"[CleanFeed] RAG retrieve error: {e}")
        return []


def build_local_prompt(text: str, task_ids: List[str]) -> str:
    task_lines = "\n".join(
        f"- {task_id}: {TASK_DEFINITIONS[task_id]['description']}"
        for task_id in task_ids
    )
    output_schema = ",\n".join(
        f'    "{task_id}": {{"label": true/false/null, "confidence": 0.xx, "reason": "20字以内原因"}}'
        for task_id in task_ids
    )
    return f"""
你是内容审核助手，请对以下内容按多个独立任务分别判断。

任务定义：
{task_lines}

只输出 JSON，不要任何解释。格式：
{{
  "task_results": {{
{output_schema}
  }}
}}

内容：
{text[:1200]}
"""


def parse_task_results(content: str) -> Dict[str, TaskPrediction]:
    json_match = re.search(r"\{.*\}", content, re.DOTALL)
    if not json_match:
        raise ValueError("No JSON found in output")

    data = json.loads(json_match.group(0))
    raw_task_results = data.get("task_results", {})
    task_results = empty_task_result()

    for task_id in DEFAULT_TASK_IDS:
        raw = raw_task_results.get(task_id, {})
        label = raw.get("label")
        if label not in [True, False, None]:
            label = None
        task_results[task_id] = TaskPrediction(
            label=label,
            confidence=round(float(raw.get("confidence", 0.0)), 2),
            reason=raw.get("reason", ""),
        )

    return task_results


def detect_with_local_model(text: str, task_ids: List[str]) -> Optional[ClassifyResponse]:
    if not OLLAMA_ENABLED:
        return None

    try:
        prompt = build_local_prompt(text, task_ids)
        response = requests.post(
            OLLAMA_API,
            json={
                "model": OLLAMA_MODEL,
                "temperature": 0.1,
                "stream": False,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=60,
        )
        if response.status_code != 200:
            print(f"[CleanFeed] Local model error response: {response.text[:200]}")
            return None

        content = response.json()["message"]["content"].strip()
        task_results = parse_task_results(content)
        confidence = derive_primary_confidence(task_results)
        result = ClassifyResponse(
            task_results=task_results,
            is_low_quality=task_results["is_low_quality"].label is True,
            is_ai_generated=task_results["is_ai_generated"].label is True,
            confidence=confidence,
            label=derive_primary_label(task_results),
            reason=derive_primary_reason(task_results),
            rag_example=None,
            detect_method="local_model",
        )
        return result
    except Exception as e:
        print(f"[CleanFeed] Local model detect error: {str(e)}")
        traceback.print_exc()
        return None


def classify_with_rag(text: str, platform: Optional[str] = None) -> ClassifyResponse:
    examples = rag_retrieve(text)
    task_results = empty_task_result()

    lq_keywords = ["家人们", "谁懂啊", "绝绝子", "yyds", "宝子们", "挖到宝", "赶紧冲", "码住", "太香了", "好哭了", "我真的会谢"]
    if platform == "zhihu":
        lq_keywords = ["谢邀", "先占个坑", "先问是不是", "以上", "互关互赞", "加微信"]
    ai_patterns = ["随着", "在当今社会", "综上所述", "由此可见", "一方面", "另一方面", "作为AI"]

    low_quality_votes = sum(1 for ex in examples if ex["is_low_quality"])
    ai_votes = sum(1 for ex in examples if ex["is_ai_generated"])
    total_examples = len(examples) or 1

    lq_matches = [kw for kw in lq_keywords if kw in text]
    ai_matches = [kw for kw in ai_patterns if kw in text]

    lq_score = low_quality_votes / total_examples * 0.6 + min(0.4, len(lq_matches) * 0.15)
    ai_score = ai_votes / total_examples * 0.6 + min(0.4, len(ai_matches) * 0.18)

    task_results["is_low_quality"] = TaskPrediction(
        label=lq_score > 0.45,
        confidence=round(lq_score if lq_score > 0.45 else 1 - lq_score, 2),
        reason=f"命中低质量模式: {', '.join(lq_matches[:2])}" if lq_matches else "RAG 相似样本判断",
    )
    task_results["is_ai_generated"] = TaskPrediction(
        label=ai_score > 0.5,
        confidence=round(ai_score if ai_score > 0.5 else 1 - ai_score, 2),
        reason=f"命中 AI 模式: {', '.join(ai_matches[:2])}" if ai_matches else "RAG 相似样本判断",
    )

    rag_example = examples[0]["text"][:80] + "..." if examples else None
    return ClassifyResponse(
        task_results=task_results,
        is_low_quality=task_results["is_low_quality"].label is True,
        is_ai_generated=task_results["is_ai_generated"].label is True,
        confidence=derive_primary_confidence(task_results),
        label=derive_primary_label(task_results),
        reason=derive_primary_reason(task_results),
        rag_example=rag_example,
        detect_method="rag",
    )


@app.on_event("startup")
async def startup_event():
    init_knowledge_base()


@app.get("/")
async def root():
    return {"message": "CleanFeed API", "status": "running", "examples_count": collection.count()}


@app.post("/classify", response_model=ClassifyResponse)
async def classify(request: ClassifyRequest):
    try:
        task_ids = request.task_ids or DEFAULT_TASK_IDS
        local_result = detect_with_local_model(request.text, task_ids)
        if local_result and local_result.confidence >= LOCAL_MODEL_CONFIDENCE_THRESHOLD:
            return local_result
        return classify_with_rag(request.text, request.platform)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "examples_count": collection.count(),
        "embedding_model": "BAAI/bge-small-zh-v1.5" if embedding_model else "none",
        "task_ids": DEFAULT_TASK_IDS,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8765)

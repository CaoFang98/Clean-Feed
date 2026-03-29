from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import chromadb
import os
import requests
import json
import re
import traceback
from pathlib import Path
from typing import Optional, List, Dict

app = FastAPI(title="CleanFeed API")

# CORS: 仅允许浏览器扩展和本地开发访问
# allow_origin_regex 支持正则匹配，覆盖所有 chrome-extension:// 和 localhost 端口
CORS_ORIGIN_REGEX = os.getenv(
    "CLEANFEED_CORS_ORIGIN_REGEX",
    r"^(chrome-extension://.*|http://localhost(:\d+)?)$"
)
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=CORS_ORIGIN_REGEX,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Initialize models and DB
print("[CleanFeed] Loading embedding model...")
try:
    embedding_model = SentenceTransformer('BAAI/bge-small-zh-v1.5')
    print("[CleanFeed] Embedding model loaded successfully!")
except Exception as e:
    print(f"[CleanFeed] Warning: Failed to load embedding model: {e}")
    embedding_model = None

# Local Ollama model configuration (从环境变量读取，方便部署时调整)
OLLAMA_ENABLED = os.getenv("OLLAMA_ENABLED", "true").lower() == "true"
OLLAMA_API = os.getenv("OLLAMA_API", "http://localhost:11434/api/chat")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:8b")
LOCAL_MODEL_CONFIDENCE_THRESHOLD = float(os.getenv("OLLAMA_CONFIDENCE_THRESHOLD", "0.9"))

print(f"[CleanFeed] Local model enabled: {OLLAMA_ENABLED}, model: {OLLAMA_MODEL}")

# Initialize Chroma DB
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = chroma_client.get_or_create_collection(name="content_examples")

# Load seed data from external JSON file
SEED_DATA_PATH = Path(__file__).parent / "seed_data.json"
try:
    with open(SEED_DATA_PATH, "r", encoding="utf-8") as f:
        SAMPLE_DATA = json.load(f)
    print(f"[CleanFeed] Loaded {len(SAMPLE_DATA)} seed examples from {SEED_DATA_PATH}")
except Exception as e:
    print(f"[CleanFeed] Warning: Failed to load seed data: {e}")
    SAMPLE_DATA = []

class ClassifyRequest(BaseModel):
    text: str
    platform: Optional[str] = None  # xiaohongshu | zhihu

class ClassifyResponse(BaseModel):
    is_low_quality: bool
    is_ai_generated: bool
    confidence: float
    label: str
    reason: str
    rag_example: Optional[str] = None
    detect_method: Optional[str] = None  # 新增：检测方式 local_model / rag

def init_knowledge_base():
    """Initialize Chroma DB with sample data"""
    print("[CleanFeed] Initializing knowledge base...")
    
    # Check if already initialized
    if collection.count() > 0:
        print(f"[CleanFeed] Knowledge base already has {collection.count()} examples")
        return
    
    # Add documents
    documents = [d["text"] for d in SAMPLE_DATA]
    metadatas = [{"label": d["label"], "category": d["category"]} for d in SAMPLE_DATA]
    ids = [f"example_{i}" for i in range(len(SAMPLE_DATA))]
    
    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )
    print(f"[CleanFeed] Added {len(SAMPLE_DATA)} examples to knowledge base")

def rag_retrieve(text: str, top_k: int = 3) -> List[Dict]:
    """Retrieve similar examples from knowledge base"""
    try:
        results = collection.query(
            query_texts=[text],
            n_results=top_k
        )
        
        examples = []
        if results["documents"] and results["documents"][0]:
            for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
                examples.append({
                    "text": doc,
                    "label": meta["label"],
                    "category": meta["category"]
                })
        return examples
    except Exception as e:
        print(f"[CleanFeed] RAG retrieve error: {e}")
        return []

def detect_with_local_model(text: str, platform: Optional[str] = None) -> Optional[ClassifyResponse]:
    """用本地Ollama模型检测内容，失败返回None走RAG兜底"""
    if not OLLAMA_ENABLED:
        return None
    
    try:
        prompt = f"""
        你是知乎内容分类助手，严格按照以下要求输出：
        1. 判断内容是否为低质量，标签分为三类：low_quality（低质量）、ai_generated（AI生成）、genuine（正常）
        2. 只输出JSON，不要任何其他内容、不要markdown、不要解释、不要```包裹：
        {{"label": "xxx", "confidence": 0.xx, "reason": "20字以内简短原因"}}
        3. 置信度0-1之间，越高越确定
        要判断的内容：{text[:500]}
        """
        
        print(f"\n[CleanFeed] === Calling local model: {OLLAMA_MODEL} ===")
        response = requests.post(OLLAMA_API, json={
            "model": OLLAMA_MODEL,
            "temperature": 0.1,
            "stream": False,  # 关闭流式输出，一次返回
            "messages": [{"role": "user", "content": prompt}]
        }, timeout=60)  # 超时拉长到60秒，首次加载足够
        
        print(f"[CleanFeed] Local model response status: {response.status_code}")
        if response.status_code != 200:
            print(f"[CleanFeed] Local model error response: {response.text[:200]}")
            return None
            
        result = response.json()
        content = result["message"]["content"].strip()
        print(f"[CleanFeed] Local model raw output: {content}")
        
        # 容错提取JSON，兼容模型输出带markdown/多余内容的情况
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if not json_match:
            print(f"[CleanFeed] No JSON found in output")
            return None
            
        json_str = json_match.group(0)
        data = json.loads(json_str)
        
        # 校验字段完整性
        required_fields = ["label", "confidence", "reason"]
        if not all(f in data for f in required_fields):
            print(f"[CleanFeed] Missing required fields in output")
            return None
            
        # 转换为标准响应格式
        res = ClassifyResponse(
            is_low_quality=data["label"] in ["low_quality", "ai_generated"],
            is_ai_generated=data["label"] == "ai_generated",
            confidence=round(float(data["confidence"]), 2),
            label=data["label"],
            reason=data["reason"],
            rag_example=None,
            detect_method="local_model"
        )
        print(f"[CleanFeed] Local model parsed result: {res.model_dump()}")
        return res
    except Exception as e:
        print(f"[CleanFeed] Local model detect error: {str(e)}")
        traceback.print_exc()
        return None

def classify_with_rag(text: str, platform: Optional[str] = None) -> ClassifyResponse:
    """Classify content with RAG retrieval"""
    examples = rag_retrieve(text)
    
    # Count labels in retrieved examples
    label_counts = {}
    for ex in examples:
        label = ex["label"]
        label_counts[label] = label_counts.get(label, 0) + 1
    
    # Simple keyword-based detection
    low_quality_keywords_xiaohongshu = ["家人们", "谁懂啊", "绝绝子", "yyds", "宝子们", "挖到宝", "赶紧冲", "码住", "太香了", "好哭了", "我真的会谢"]
    low_quality_keywords_zhihu = ["谢邀", "先占个坑", "先问是不是", "以上"]
    ai_generated_patterns = ["随着...的发展", "在当今社会", "综上所述", "由此可见", "一方面...另一方面"]
    
    # Choose keywords based on platform
    low_quality_keywords = low_quality_keywords_xiaohongshu
    if platform == "zhihu":
        low_quality_keywords = low_quality_keywords_zhihu
    
    # Check matches
    lq_matches = [kw for kw in low_quality_keywords if kw in text]
    ai_matches = [pat for pat in ai_generated_patterns if pat in text]
    
    # Calculate confidence
    lq_score = 0.0
    ai_score = 0.0
    
    # RAG-based scores
    total_examples = len(examples)
    if total_examples > 0:
        lq_score += (label_counts.get("low_quality", 0) + label_counts.get("ai_generated", 0)) / total_examples * 0.6
        ai_score += label_counts.get("ai_generated", 0) / total_examples * 0.6
    
    # Keyword-based scores
    if lq_matches:
        lq_score += min(0.4, len(lq_matches) * 0.15)
    if ai_matches:
        ai_score += min(0.4, len(ai_matches) * 0.2)
    
    # Final decisions
    is_low_quality = lq_score > 0.4
    is_ai_generated = ai_score > 0.5
    
    # Determine label and confidence
    if is_ai_generated:
        label = "ai_generated"
        confidence = ai_score
    elif is_low_quality:
        label = "low_quality"
        confidence = lq_score
    else:
        label = "genuine"
        confidence = 1.0 - max(lq_score, ai_score)
    
    # Generate reason
    reasons = []
    if examples:
        top_example = examples[0]
        if top_example["label"] != "genuine":
            reasons.append(f"Similar to {top_example['label']} content")
    if lq_matches:
        reasons.append(f"Contains low-quality patterns: {', '.join(lq_matches[:2])}")
    if ai_matches:
        reasons.append(f"Contains AI-style patterns: {', '.join(ai_matches[:2])}")
    
    reason = " | ".join(reasons) if reasons else "Content appears to be genuine"
    rag_example = examples[0]["text"][:80] + "..." if examples else None
    
    return ClassifyResponse(
        is_low_quality=is_low_quality,
        is_ai_generated=is_ai_generated,
        confidence=round(confidence, 2),
        label=label,
        reason=reason,
        rag_example=rag_example,
        detect_method="rag"
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
        # 第一步：优先调用本地小模型
        local_result = detect_with_local_model(request.text, request.platform)
        
        # 本地模型返回成功，且置信度足够，直接返回结果
        if local_result and local_result.confidence >= LOCAL_MODEL_CONFIDENCE_THRESHOLD:
            print(f"[CleanFeed] Used local model, confidence: {local_result.confidence}")
            return local_result
        
        # 本地模型失败/置信度不够，走RAG兜底
        print(f"[CleanFeed] Fallback to RAG detection")
        rag_result = classify_with_rag(request.text, request.platform)
        return rag_result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {
        "status": "healthy", 
        "examples_count": collection.count(),
        "embedding_model": "BAAI/bge-small-zh-v1.5" if embedding_model else "none"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8765)

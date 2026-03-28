from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import chromadb
import os
from typing import Optional, List, Dict

app = FastAPI(title="CleanFeed API")

# Enable CORS for browser extension
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
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

# Initialize Chroma DB
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="content_examples")

# Sample knowledge base (50 examples)
SAMPLE_DATA = [
    # Low-quality / AI-generated content examples
    {"text": "家人们谁懂啊！今天真的是绝绝子！挖到宝了！这个东西真的是yyds！一定要买！", "label": "low_quality", "category": "xiaohongshu_spam"},
    {"text": "姐妹们！这个真的太好用了！我已经用了一个月了，皮肤真的变好了！赶紧冲！", "label": "low_quality", "category": "xiaohongshu_ad"},
    {"text": "家人们！今天给大家分享一个好东西！真的是太香了！不看后悔一辈子！", "label": "low_quality", "category": "clickbait"},
    {"text": "宝子们！这个真的是绝了！我不允许还有人不知道！赶紧码住！", "label": "low_quality", "category": "xiaohongshu_spam"},
    {"text": "家人们谁懂啊！这个真的是太好哭了！我真的会谢！", "label": "low_quality", "category": "xiaohongshu_spam"},
    
    {"text": "在当今社会，人工智能技术的发展日新月异。随着深度学习算法的不断优化，我们在自然语言处理、计算机视觉等领域取得了显著成就。", "label": "ai_generated", "category": "generic_ai"},
    {"text": "随着经济全球化的深入发展，企业面临着前所未有的机遇和挑战。在这样的背景下，如何提升核心竞争力成为了每个企业必须思考的问题。", "label": "ai_generated", "category": "generic_ai"},
    {"text": "教育是国之大计、党之大计。培养什么人、怎样培养人、为谁培养人是教育的根本问题。", "label": "ai_generated", "category": "generic_ai"},
    
    # High-quality / genuine content examples
    {"text": "今天去了西湖，人真的很多，但是风景真的很美。拍了很多照片，虽然太阳很大有点晒，但是很开心。", "label": "genuine", "category": "personal"},
    {"text": "最近在读《活着》，第三次读了，每次都有新的感受。余华的笔力真的太厉害了，富贵的一生让人唏嘘。", "label": "genuine", "category": "personal"},
    {"text": "今天做了番茄炒蛋，有点咸了，下次少放点盐。不过整体味道还可以，配米饭吃了两大碗。", "label": "genuine", "category": "personal"},
    {"text": "杭州的春天真的太美了，太子湾的郁金香开了，虽然人挤人，但是看到那么美的花还是觉得值了。", "label": "genuine", "category": "travel"},
    {"text": "最近在学Python，跟着网上的教程做了一个小爬虫，虽然很简单，但是跑通的时候真的很有成就感！", "label": "genuine", "category": "tech"},
    
    # Zhihu low-quality examples
    {"text": "谢邀。这个问题很简单。先说结论：是的。然后分几点：1. 首先2. 其次3. 最后。以上。", "label": "low_quality", "category": "zhihu_template"},
    {"text": "这个问题我来回答一下。先占个坑，有空再来写。", "label": "low_quality", "category": "zhihu_placeholder"},
    {"text": "先问是不是，再问为什么。", "label": "low_quality", "category": "zhihu_cliche"},
    
    # Zhihu high-quality examples
    {"text": "这个问题我刚好有研究。从学术角度来看，主要有三个流派：第一个流派认为...第二个流派认为...第三个流派认为...我个人倾向于第一个流派，因为...", "label": "genuine", "category": "zhihu_quality"},
    {"text": "分享一下我的亲身经历。三年前我也遇到了同样的问题，当时尝试了很多方法都没有用。后来偶然间看到一篇论文，里面提到一个思路，我试着调整了一下，没想到真的有效。具体来说是这样的...", "label": "genuine", "category": "personal_experience"},
]

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
        rag_example=rag_example
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
        result = classify_with_rag(request.text, request.platform)
        return result
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

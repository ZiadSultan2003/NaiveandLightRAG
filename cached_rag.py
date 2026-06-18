import os
import json
import numpy as np
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

load_dotenv()

STORAGE_DIR = "./cached_naive_storage"
CACHE_FILE = "./cached_naive_storage/semantic_cache.json"

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}
)

global_cached_db = None

def load_cache() -> dict:
    """تحميل الـ Cache من ملف JSON لو موجود"""
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_cache(cache_data: dict):
    """حفظ الـ Cache محلياً"""
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache_data, f, ensure_ascii=False, indent=4)

def cosine_similarity(v1, v2):
    """حساب التشابه الدلالي بين متجهين"""
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    return dot_product / (norm_v1 * norm_v2) if (norm_v1 * norm_v2) > 0 else 0.0

def check_semantic_cache(question: str) -> str:
    """البحث الدلالي جوه الـ Cache قبل الذهاب للـ LLM"""
    cache = load_cache()
    if not cache:
        return None
        
    # تحويل السؤال الحالي لـ Vector
    q_vector = embeddings.embed_query(question)
    
    best_score = 0.0
    best_answer = None
    
    # المقارنة مع كل الأسئلة اللي اتسألت قبل كده
    for cached_q, cached_data in cache.items():
        cached_vector = cached_data["vector"]
        score = cosine_similarity(q_vector, cached_vector)
        
        if score > best_score:
            best_score = score
            best_answer = cached_data["answer"]
            
    # 🔥 صياعة الـ Semantic Cache: لو التشابه عالي جداً (أكبر من 92%)، رجع الإجابة فوراً!
    if best_score >= 0.92:
        print(f"🚀 [Semantic Cache HIT!] - Similarity: {best_score*100:.1f}% - Returning cached answer.")
        return f"⚡ [الرد مسترجع فوراً من الـ Semantic Cache بجودة تشابه {best_score*100:.1f}%]\n\n{best_answer}"
        
    return None

def add_to_cache(question: str, answer: str):
    """إضافة السؤال وإجابته ومتجهه للـ Cache"""
    cache = load_cache()
    q_vector = embeddings.embed_query(question)
    
    cache[question] = {
        "answer": answer,
        "vector": q_vector
    }
    save_cache(cache)
    print("💾 Answer successfully added to Semantic Cache.")

def process_cached_doc(full_text: str, chunk_size: int = 600):
    """بناء الـ Vector DB للمستند بأمان للـ Windows"""
    global global_cached_db
    
    if global_cached_db is not None:
        try:
            global_cached_db.delete_collection()
        except: pass
        global_cached_db = None
        
    # مسح ملف الـ Cache القديم عند رفع ملف جديد
    if os.path.exists(CACHE_FILE):
        try: os.remove(CACHE_FILE)
        except: pass

    overlap_size = int(chunk_size * 0.1)
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap_size)
    texts = splitter.split_text(full_text)
    docs = [Document(page_content=t) for t in texts]

    global_cached_db = Chroma.from_documents(
        docs, 
        embeddings, 
        persist_directory=STORAGE_DIR
    )
    print(f"✅ [Cached RAG Base DB Indexed] - Chunks: {len(docs)}")

def query_cached_rag_context(question: str) -> list:
    """استرجاع السياق من الـ Vector Store في حالة الـ Cache Miss"""
    global global_cached_db
    if global_cached_db is None:
        global_cached_db = Chroma(persist_directory=STORAGE_DIR, embedding_function=embeddings)
        
    retrieved_docs = global_cached_db.as_retriever(search_kwargs={"k": 4}).invoke(question)
    return [doc.page_content for doc in retrieved_docs]
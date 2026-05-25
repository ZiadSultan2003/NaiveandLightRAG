import os
import asyncio
import numpy as np
from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc
from groq import Groq, RateLimitError
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

WORKING_DIR = "./lightrag_storage"
if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

print("⏳ Loading HuggingFace Embeddings model for LightRAG...")
model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
api_key = os.getenv("GROQ_API_KEY")

groq_lock = asyncio.Lock()

# محرك Groq الذكي المزود بخاصية الـ Auto-Retry
async def groq_llm_interface(prompt, system_prompt=None, history=[], **kwargs) -> str:
    async with groq_lock:
        client = Groq(api_key=api_key)
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        for h in history:
            messages.append(h)
        messages.append({"role": "user", "content": prompt})
        
        max_retries = 5
        retry_delay = 12.0
        
        for attempt in range(max_retries):
            try:
                await asyncio.sleep(2.0) # أمان بين الطلبات
                response = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=messages,
                    temperature=kwargs.get("temperature", 0.1),
                )
                return response.choices[0].message.content
            except RateLimitError:
                print(f"\n⚠️ [Groq Rate Limit Hit (429)] - Sleeping for {retry_delay}s... (Attempt {attempt + 1}/{max_retries})")
                await asyncio.sleep(retry_delay)
                retry_delay *= 1.5 
            except Exception as e:
                raise e
        raise RuntimeError("❌ Failed to clear Groq Rate Limit after maximum retries.")

async def local_embedding(texts: list[str]) -> np.ndarray:
    embeddings = model.embed_documents(texts)
    return np.array(embeddings)

# دالة مساعدة لإنشاء الـ RAG ديناميكياً بالحجم المطلوب
def get_rag_instance(chunk_size: int = 600):
    # حساب الـ Overlap تلقائياً بنسبة 10% من حجم الـ Chunk
    overlap_size = int(chunk_size * 0.1)
    
    return LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=groq_llm_interface,
        chunk_token_size=chunk_size,          # 🔥 هنا هيسمع كلام الـ Input بتاعك بالظبط!
        chunk_overlap_token_size=overlap_size,   
        embedding_func=EmbeddingFunc(
            embedding_dim=384,
            max_token_size=8192,
            func=local_embedding
        )
    )

# ─── الدوال المخصصة للاستدعاء الخارجي ───

async def process_lightrag_doc(full_text: str, chunk_size: int = 600):
    """دالة لحقن النص وبناء الـ Graph بناءً على الـ chunk_size الممرر من الـ Input"""
    if not os.path.exists(WORKING_DIR):
        os.mkdir(WORKING_DIR)
    
    # بناء الـ Instance بالـ size اللي أنت طالبه
    rag = get_rag_instance(chunk_size=chunk_size)
    
    await rag.initialize_storages()
    await rag.ainsert(full_text)

async def query_lightrag(question: str) -> str:
    """دالة استعلام الـ Graph الهجين (تستخدم الـ Default Instance المستقر)"""
    try:
        rag = get_rag_instance()
        await rag.initialize_storages()
        response = await rag.aquery(
            question, 
            param=QueryParam(mode="hybrid")
        )
        return response
    except Exception as e:
        return f"❌ LightRAG Search error: {str(e)}"
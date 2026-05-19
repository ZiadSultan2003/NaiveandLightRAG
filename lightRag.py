import os
import asyncio  # مطلوب للتحكم الصارم في الوقت والـ Lock
import numpy as np
from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc
from groq import Groq, RateLimitError  # 🔥 عملنا import للـ RateLimitError عشان نلقطها
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

WORKING_DIR = "./lightrag_storage"
if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

print("⏳ Loading HuggingFace Embeddings model for LightRAG...")
model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
api_key = os.getenv("GROQ_API_KEY")

# 🔒 قفل صارم لمنع أي تداخل أو طلبات متوازية من الـ LightRAG Workers
groq_lock = asyncio.Lock()

# محرك Groq الذكي المزود بخاصية الـ Auto-Retry عند الـ Rate Limit
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
        retry_delay = 12.0  # الـ Window المثالية لتصفير الـ TPM في Groq Free Tier
        
        for attempt in range(max_retries):
            try:
                # تأخير أساسي أمان (2 ثانية) بين كل طلب وطلب طبيعي
                await asyncio.sleep(2.0)
                
                response = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=messages,
                    temperature=kwargs.get("temperature", 0.1),
                )
                return response.choices[0].message.content
                
            except RateLimitError as e:
                # 🚨 هنا بنمسك الـ 429 ونمنع انهيار الـ Pipeline
                print(f"\n⚠️ [Groq Rate Limit Hit (429)] - Used Tokens exceeded. Sleeping for {retry_delay}s before retry... (Attempt {attempt + 1}/{max_retries})")
                await asyncio.sleep(retry_delay)
                # زيادة وقت الانتظار بشكل تصاعدي للمرة القادمة لو لسه مقفول
                retry_delay *= 1.5 
                
            except Exception as e:
                # لو في إيرور تاني مختلف نخرجه علطول
                raise e
                
        raise RuntimeError("❌ Failed to clear Groq Rate Limit after maximum retries.")

async def local_embedding(texts: list[str]) -> np.ndarray:
    embeddings = model.embed_documents(texts)
    return np.array(embeddings)

# إعداد المحرك مع ضبط حجم الـ Chunks
rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=groq_llm_interface,
    chunk_token_size=300,          
    chunk_overlap_token_size=30,   
    embedding_func=EmbeddingFunc(
        embedding_dim=384,
        max_token_size=8192,
        func=local_embedding
    )
)

# ─── الدوال المخصصة للاستدعاء الخارجي ───

async def process_lightrag_doc(full_text: str):
    """دالة لحقن النص وبناء الـ Graph بشكل تتابعي آمن ومقاوم للـ Rate Limit"""
    if not os.path.exists(WORKING_DIR):
        os.mkdir(WORKING_DIR)
        
    await rag.initialize_storages()
    await rag.ainsert(full_text)

async def query_lightrag(question: str) -> str:
    """دالة استعلام الـ Graph الهجين"""
    try:
        await rag.initialize_storages()
        response = await rag.aquery(
            question, 
            param=QueryParam(mode="hybrid")
        )
        return response
    except Exception as e:
        return f"❌ LightRAG Search error: {str(e)}"
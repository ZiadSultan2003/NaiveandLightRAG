import os
import asyncio
import tkinter as tk
from tkinter import filedialog
import numpy as np
from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc
from groq import Groq
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv  # استيراد المكتبة

# تحميل الإعدادات من ملف .env
load_dotenv()

# 1. إعداد المسارات
WORKING_DIR = "./lightrag_storage"
if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)
   
 

print("⏳ Loading HuggingFace Embeddings model...")
model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


api_key = os.getenv("GROQ_API_KEY")

# 2. محرك Groq المخصص
async def groq_llm_interface(prompt, system_prompt=None, history=[], **kwargs) -> str:
    client = Groq(api_key=api_key)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    for h in history:
        messages.append(h)
    messages.append({"role": "user", "content": prompt})
    
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages,
        temperature=kwargs.get("temperature", 0.1),
    )
    return response.choices[0].message.content

# 3. دالة الـ Embeddings
async def local_embedding(texts: list[str]) -> np.ndarray:
    
    embeddings = model.embed_documents(texts)
    return np.array(embeddings)

# 4. إعداد محرك LightRAG
rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=groq_llm_interface,
    chunk_token_size=600, 
    chunk_overlap_token_size=100,
    embedding_func=EmbeddingFunc(
        embedding_dim=384,
        max_token_size=8192,
        func=local_embedding
    )
)

# 5. دالة اختيار الملف
def select_file_dialog():
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    file_path = filedialog.askopenfilename(
        title="إختر ملف الـ PDF لتحليله",
        filetypes=[("PDF Files", "*.pdf")]
    )
    root.destroy()
    return file_path

# 6. التشغيل
async def run_light_rag():
    print("\n--- 📂 LightRAG Power Mode ---")
    
    if not api_key:
        print("❌ Error: GROQ_API_KEY not found in .env file")
        return

    print("⏳ Preparing storage...")
    await rag.initialize_storages()
    
    file_path = select_file_dialog()
    if not file_path:
        print("❌ No file selected.")
        return

    print(f"✅ Selected: {file_path}")

    try:
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        full_text = "\n".join([page.page_content for page in pages])
        print(f"📄 Pages: {len(pages)}")
    except Exception as e:
        print(f"❌ Error reading PDF: {e}")
        return

    print(f"⏳ Building knowledge graph...")
    try:
        await rag.ainsert(full_text)
        print("\n✅ Graph built successfully!")
    except Exception as e:
        print(f"❌ Graph failure: {e}")
        return 
    
    while True:
        question = input("\n🔎 Ask (or 'exit'): ")
        if question.lower() in ['exit', 'quit']: break
            
        try:
            response = await rag.aquery(
                question, 
                param=QueryParam(mode="hybrid")
            )
            print(f"\n🤖 Answer:\n{'-'*30}\n{response}\n{'-'*30}")
        except Exception as e:
            print(f"❌ Search error: {e}")

if __name__ == "__main__":
    asyncio.run(run_light_rag())
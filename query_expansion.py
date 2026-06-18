import os
import json
from dotenv import load_dotenv
from groq import Groq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

STORAGE_DIR = "./expansion_storage"
TARGET_MODEL = "llama-3.3-70b-versatile"

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}
)

global_expansion_db = None

def generate_sub_queries(original_query: str) -> list:
    """توليد 3 صياغات بديلة ومختلفة للسؤال لضمان جلب كافة القطع النصية المتعلقة بالمعنى"""
    api_key = os.getenv("GROQ_API_KEY")
    client = Groq(api_key=api_key)
    try:
        prompt = (
            f"You are an AI language expert. Generate 3 alternative versions/phrasings of the following user question "
            f"to help retrieve relevant documents. Provide the output as a valid JSON list of strings, nothing else.\n"
            f"Question: {original_query}"
        )
        completion = client.chat.completions.create(
            model=TARGET_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=200
        )
        
        # محاولة تنظيف وجلب الـ List من الـ JSON
        response_text = completion.choices[0].message.content.strip()
        sub_queries = json.loads(response_text)
        if isinstance(sub_queries, list):
            # ندمج السؤال الأصلي مع الأسئلة الجديدة
            return [original_query] + sub_queries
    except Exception as e:
        print(f"⚠️ Error expanding query: {e}")
    
    return [original_query] # Fallback لو حصل أي إيرور

def process_expansion_doc(full_text: str, chunk_size: int = 600):
    """تقطيع المستند وحفظه في الـ Vector Store التقليدي"""
    global global_expansion_db
    
    # تفريغ الـ Collection بأمان للـ Windows
    if global_expansion_db is not None:
        try:
            global_expansion_db.delete_collection()
        except: pass
        global_expansion_db = None

    overlap_size = int(chunk_size * 0.1)
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap_size)
    
    # تحويل النص لـ Documents خفيفة للـ Langchain
    from langchain_core.documents import Document
    texts = splitter.split_text(full_text)
    docs = [Document(page_content=t) for t in texts]

    global_expansion_db = Chroma.from_documents(
        docs, 
        embeddings, 
        persist_directory=STORAGE_DIR
    )
    print(f"✅ [Query Expansion RAG Indexed] - Total Chunks: {len(docs)}")

def query_expansion_rag(question: str) -> list:
    """توسيع السؤال الأصلي، والبحث بكل الأسئلة وتجميع الـ Chunks بدون تكرار"""
    global global_expansion_db
    if global_expansion_db is None:
        global_expansion_db = Chroma(persist_directory=STORAGE_DIR, embedding_function=embeddings)

    # 1. صياعة الـ Expansion: توليد الأسئلة البديلة
    all_queries = generate_sub_queries(question)
    print(f"🔍 Expanded Queries: {all_queries}")

    retrieved_chunks = []
    seen_contents = set()

    # 2. البحث في الـ DB بكل سؤال على حدة
    retriever = global_expansion_db.as_retriever(search_kwargs={"k": 2})
    for q in all_queries:
        docs = retriever.invoke(q)
        for doc in docs:
            if doc.page_content not in seen_contents:
                seen_contents.add(doc.page_content)
                retrieved_chunks.append(doc.page_content)

    return retrieved_chunks[:4] # نرجع أفضل 4 قطع فريدة ومترابطة بالمعنى الكامل
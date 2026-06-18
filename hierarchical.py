import os
import re
import json
from dotenv import load_dotenv
from groq import Groq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

STORAGE_DIR = "./hierarchical_storage"
TARGET_MODEL = "llama-3.3-70b-versatile"

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}
)

global_hierarchical_db = None

def generate_chunk_summary(chunk_text: str) -> str:
    """توليد ملخص سريع ومكثف للقطعة الكبيرة عبر الـ LLM"""
    api_key = os.getenv("GROQ_API_KEY")
    client = Groq(api_key=api_key)
    try:
        prompt = f"Summarize the main core topics of this text in 2 short bullet points or sentences:\n\n{chunk_text}"
        completion = client.chat.completions.create(
            model=TARGET_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=150
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"⚠️ Error generating summary: {e}")
        return "Summary unavailable."

def process_hierarchical_doc(full_text: str, chunk_size: int = 600):
    """بناء الشجرة الهرمية: Parent -> Summary -> Child"""
    global global_hierarchical_db
    
    # ─── 🛠️ الحل الآمن للـ Windows لمنع File Locking Error ───
    if global_hierarchical_db is not None:
        try:
            global_hierarchical_db.delete_collection()
            print("🧹 Existing Hierarchical Chroma collection cleared safely.")
        except Exception as e:
            print(f"⚠️ Note: Could not delete hierarchical collection cleanly: {e}")
        global_hierarchical_db = None

    child_size = chunk_size
    parent_size = chunk_size * 3  # القطعة الكبيرة تضمن سياق عريض جداً
    
    # 1. تقطيع النص لـ Parents كبار
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=parent_size, chunk_overlap=int(parent_size * 0.1))
    parent_texts = parent_splitter.split_text(full_text)
    
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=child_size, chunk_overlap=int(child_size * 0.1))
    child_documents = []
    
    print(f"⏳ Generating Summaries for {len(parent_texts)} Parent Chapters...")
    
    # 2. لكل Parent، هنولد ملخص، ونقطعها لـ Children أصغر
    for i, p_text in enumerate(parent_texts):
        # توليد ملخص سريع للـ Parent ده
        p_summary = generate_chunk_summary(p_text)
        
        sub_chunks = child_splitter.split_text(p_text)
        for c_text in sub_chunks:
            # نربط الـ Child بالـ Parent والـ Summary معاً في الـ Metadata
            doc = Document(
                page_content=c_text,
                metadata={
                    "parent_content": p_text,
                    "parent_summary": p_summary,
                    "source": "hierarchical_upload"
                }
            )
            child_documents.append(doc)

    # 3. حفظ الـ Children الـ Vectors في Chroma
    global_hierarchical_db = Chroma.from_documents(
        child_documents, 
        embeddings, 
        persist_directory=STORAGE_DIR
    )
    print(f"✅ [Hierarchical RAG Indexed] - Total Leaf/Child Chunks: {len(child_documents)}")

def query_hierarchical(question: str) -> list:
    """البحث في الـ Children واسترجاع الـ السياق الهرمي المدمج (الـ Parent + ملخصه)"""
    global global_hierarchical_db
    if global_hierarchical_db is None:
        global_hierarchical_db = Chroma(persist_directory=STORAGE_DIR, embedding_function=embeddings)
        
    try:
        # البحث عن أفضل قطع دقيقة
        retrieved_children = global_hierarchical_db.as_retriever(search_kwargs={"k": 3}).invoke(question)
        
        hierarchical_contexts = []
        seen_parents = set()
        
        for child in retrieved_children:
            parent_text = child.metadata.get("parent_content")
            parent_summary = child.metadata.get("parent_summary", "No Summary")
            
            if parent_text and parent_text not in seen_parents:
                seen_parents.add(parent_text)
                
                # صياعة الـ Hierarchical: ندمج الملخص مع النص الكامل للـ Parent لتغذية الـ LLM بأعلى جودة سياق
                formatted_context = f"[Section Overview/Summary]: {parent_summary}\n[Detailed Section Content]: {parent_text}"
                hierarchical_contexts.append(formatted_context)
                
        return hierarchical_contexts[:2]  # إرجاع أفضل سياقين هرميين مدمجين لعدم تخطي الـ Context Window
    except Exception as e:
        print(f"❌ Error in Hierarchical Search: {e}")
        return []
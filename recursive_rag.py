import os
import uuid  # 🔥 تم إضافتها لتوليد أسماء كوليكشن ديناميكية وفريدة
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

STORAGE_DIR = "./recursive_storage"

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}
)

# متغيرات عامة لحفظ الداتا بيس واسم الكوليكشن النشطة في الميموري
global_db = None
current_collection_name = "recursive_default"  # 🔥 لتتبع الكوليكشن النشطة عبر الطلبات

def process_recursive_doc(full_text: str, chunk_size: int = 600):
    """
    بناء الـ Recursive RAG بدون كلاسات معقدة:
    الـ chunk_size هو الـ Child الصغير للبحث الدقيق.
    الـ Parent هيكون ضعف الحجم للسياق الكامل.
    """
    global global_db, current_collection_name
    
    # ─── 🛠️ الحل الآمن للـ Windows لمنع File Locking Error ───
    if global_db is not None:
        try:
            # مسح الـ Collection الحالية وتفريغ البيانات برفق بدل مسح الفولدر بالكامل
            global_db.delete_collection()
            print(f"🧹 Existing Chroma collection [{current_collection_name}] cleared safely.")
        except Exception as e:
            print(f"⚠️ Note: Could not delete collection cleanly: {e}")
        global_db = None

    # توليد اسم كوليكشن فريد تماماً لهذه الجلسة لمنع الـ Overriding والـ Caching
    current_collection_name = f"rec_col_{uuid.uuid4().hex[:8]}"

    child_size = chunk_size
    parent_size = chunk_size * 2
    parent_overlap = int(parent_size * 0.1)
    child_overlap = int(child_size * 0.1)

    # 1. تقطيع النص كـ Parents (المستندات الكبيرة للسياق)
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=parent_size, chunk_overlap=parent_overlap)
    parent_texts = parent_splitter.split_text(full_text)

    child_splitter = RecursiveCharacterTextSplitter(chunk_size=child_size, chunk_overlap=child_overlap)
    
    child_documents = []
    
    # 2. لكل قطعة كبيرة، هنطلع قطع صغيرة ونربطهم ببعض في الـ Metadata
    for p_text in parent_texts:
        sub_chunks = child_splitter.split_text(p_text)
        for c_text in sub_chunks:
            # الصياعة هنا: الـ Child بيتسيف وجواه الـ Parent بتاعه بالكامل
            doc = Document(
                page_content=c_text,
                metadata={"parent_content": p_text, "source": "uploaded_file"}
            )
            child_documents.append(doc)

    # حماية ضد النصوص الفارغة
    if not child_documents:
        child_documents = [Document(page_content="Placeholder context because text parsing returned empty.", metadata={"parent_content": "Placeholder"})]

    # 3. حفظ الـ Children الـ Vectors في Chroma مع اسم كوليكشن فريد
    global_db = Chroma.from_documents(
        child_documents, 
        embeddings, 
        persist_directory=STORAGE_DIR,
        collection_name=current_collection_name  # 🔥 السطر السحري لمنع الكاش القديم
    )
    print(f"✅ [Custom Recursive RAG Cleanly Indexed] - Collection: {current_collection_name} - Total Child Chunks: {len(child_documents)}")

def query_recursive(question: str) -> list:
    """البحث بالـ Child واسترجاع الـ Parent الأصلي فوراً"""
    global global_db, current_collection_name
    
    if global_db is None:
        global_db = Chroma(
            persist_directory=STORAGE_DIR, 
            embedding_function=embeddings,
            collection_name=current_collection_name
        )
        
    try:
        # البحث في الـ Children بقيمة k=10 لضمان البحث الموسع والوصول للملفين
        retrieved_children = global_db.as_retriever(search_kwargs={"k": 20}).invoke(question)
        
        # استخراج الـ Parents الكبار من الـ Metadata (مع منع التكرار باستخدام set)
        parent_contexts = []
        seen = set()
        for child in retrieved_children:
            parent_text = child.metadata.get("parent_content")
            if parent_text and parent_text not in seen:
                seen.add(parent_text)
                parent_contexts.append(parent_text)
                
        return parent_contexts[:4]  # تم رفعها لـ 4 لضمان استيعاب سياق كافٍ من الملفين معاً
    except Exception as e:
        print(f"❌ Error in Custom Recursive Search: {e}")
        return []
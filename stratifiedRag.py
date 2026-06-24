import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

load_dotenv()

# مسارات التخزين للطبقات الثلاث المنفصلة
STORAGE_DIR_L0 = "./stratified_storage/layer_0_global"
STORAGE_DIR_L1 = "./stratified_storage/layer_1_chunk"
STORAGE_DIR_L2 = "./stratified_storage/layer_2_atomic"

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}
)

db_l0 = None
db_l1 = None
db_l2 = None

def process_stratified_doc(full_text: str, chunk_size: int = 600):
    """تقسيم المستند لثلاث طبقات منفصلة (Strata) بناءً على مستوى التفاصيل"""
    global db_l0, db_l1, db_l2
    
    # تفريغ الـ Collections السابقة بأمان للويندوز
    for db in [db_l0, db_l1, db_l2]:
        if db is not None:
            try: db.delete_collection()
            except: pass
            
    db_l0, db_l1, db_l2 = None, None, None

    # ─── الطبقة الأولى: Layer 1 (القطع المتوسطة التقليدية) ───
    splitter_l1 = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=int(chunk_size * 0.1))
    texts_l1 = splitter_l1.split_text(full_text)
    docs_l1 = [Document(page_content=t, metadata={"stratum": "l1_chunk"}) for t in texts_l1]
    
    # ─── الطبقة الصفرية: Layer 0 (قطع ضخمة / نظرة شاملة كالأبواب والفصول) ───
    size_l0 = chunk_size * 4
    splitter_l0 = RecursiveCharacterTextSplitter(chunk_size=size_l0, chunk_overlap=int(size_l0 * 0.1))
    texts_l0 = splitter_l0.split_text(full_text)
    docs_l0 = [Document(page_content=t, metadata={"stratum": "l0_global"}) for t in texts_l0]

    # ─── الطبقة الذرية: Layer 2 (جمل قصيرة ودقيقة جداً للحقائق المحددة) ───
    size_l2 = 150
    splitter_l2 = RecursiveCharacterTextSplitter(chunk_size=size_l2, chunk_overlap=20)
    texts_l2 = splitter_l2.split_text(full_text)
    docs_l2 = [Document(page_content=t, metadata={"stratum": "l2_atomic"}) for t in texts_l2]

    # بناء الفيكتور ستور المستقل لكل طبقة
    db_l0 = Chroma.from_documents(docs_l0, embeddings, persist_directory=STORAGE_DIR_L0)
    db_l1 = Chroma.from_documents(docs_l1, embeddings, persist_directory=STORAGE_DIR_L1)
    db_l2 = Chroma.from_documents(docs_l2, embeddings, persist_directory=STORAGE_DIR_L2)
    
    print(f"✅ [Stratified RAG Indexed] - L0(Global): {len(docs_l0)} | L1(Chunk): {len(docs_l1)} | L2(Atomic): {len(docs_l2)}")

def query_stratified(question: str) -> list:
    """البحث الطبقي: دمج نتائج البحث من كافة الطبقات لتوفير سياق هجين ومتوازن"""
    global db_l0, db_l1, db_l2
    
    if db_l0 is None: db_l0 = Chroma(persist_directory=STORAGE_DIR_L0, embedding_function=embeddings)
    if db_l1 is None: db_l1 = Chroma(persist_directory=STORAGE_DIR_L1, embedding_function=embeddings)
    if db_l2 is None: db_l2 = Chroma(persist_directory=STORAGE_DIR_L2, embedding_function=embeddings)
    
    stratified_contexts = []
    
    try:
        # استرجاع القطعة الأفضل من الرؤية الشاملة (L0) لضمان السياق العريض
        res_l0 = db_l0.as_retriever(search_kwargs={"k": 1}).invoke(question)
        if res_l0: stratified_contexts.append(f"[Global/Chapter Overview Stratum]:\n{res_l0[0].page_content}")
        
        # استرجاع قطعتين من الطبقة المتوسطة (L1) للشرح المتوازن
        res_l1 = db_l1.as_retriever(search_kwargs={"k": 2}).invoke(question)
        for doc in res_l1:
            stratified_contexts.append(f"[Detailed Paragraph Stratum]:\n{doc.page_content}")
            
        # استرجاع قطعتين دقيقتين جداً من الطبقة الذرية (L2) لاصطياد الأرقام والأسماء بدقة
        res_l2 = db_l2.as_retriever(search_kwargs={"k": 2}).invoke(question)
        for doc in res_l2:
            stratified_contexts.append(f"[Atomic Fact/Sentence Stratum]:\n{doc.page_content}")
            
        return stratified_contexts
    except Exception as e:
        print(f"❌ Error in Stratified Search: {e}")
        return []
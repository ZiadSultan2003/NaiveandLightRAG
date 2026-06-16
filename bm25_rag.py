import os
import pickle
import nltk
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv

load_dotenv()

# التأكد من تحميل الـ Tokenizer الخاص بـ NLTK لتقسيم النصوص بشكل صحيح
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
except (LookupError, AttributeError):
    print("⏳ Downloading required NLTK resources...")
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)

STORAGE_DIR = "./bm25_storage"
INDEX_PATH = os.path.join(STORAGE_DIR, "bm25_index.pkl")
CHUNKS_PATH = os.path.join(STORAGE_DIR, "chunks.pkl")

def chunk_text_by_words(text: str, chunk_size: int = 600):
    """تقسيم النص لـ Chunks بناءً على عدد الكلمات مع Overlap بنسبة 10%"""
    words = nltk.word_tokenize(text)
    chunks = []
    overlap = int(chunk_size * 0.1)
    
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))
        # التحرك للأمام مع احتساب الـ Overlap
        start += (chunk_size - overlap) if (chunk_size - overlap) > 0 else chunk_size
        
    return chunks

# ─── الدوال المخصصة للاستدعاء الخارجي ───

def process_bm25_doc(full_text: str, chunk_size: int = 600):
    """بناء الـ BM25 Index محلياً تماماً بصفر تكلفة توكنز وبسرعة فائقة"""
    if not os.path.exists(STORAGE_DIR):
        os.makedirs(STORAGE_DIR)
        
    # 1. تقطيع النص بناءً على الـ Chunk Size الممرر من الـ Input
    chunks = chunk_text_by_words(full_text, chunk_size=chunk_size)
    
    if not chunks:
        print("⚠️ Warning: No chunks generated from the document.")
        return
        
    # 2. عمل Tokenization لكل Chunk (تحويله لـ Lowercase وقائمة كلمات)
    tokenized_corpus = [nltk.word_tokenize(chunk.lower()) for chunk in chunks]
    
    # 3. حساب معادلات الـ BM25 على الـ Corpus
    bm25 = BM25Okapi(tokenized_corpus)
    
    # 4. حفظ الـ Index والـ Chunks في ملفات Pickle لوكال
    with open(INDEX_PATH, "wb") as f:
        pickle.dump(bm25, f)
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)
        
    print(f"✅ [BM25 Index Built] - Total Chunks: {len(chunks)} (Chunk Size: {chunk_size})")

def query_bm25(question: str, top_k: int = 3) -> list:
    """البحث المحلي السريع بالكلمات المفتاحية لاسترجاع أفضل النصوص المتطابقة"""
    if not os.path.exists(INDEX_PATH) or not os.path.exists(CHUNKS_PATH):
        return ["❌ لم يتم معالجة أي ملف للـ BM25 حتى الآن. يرجى رفع ملف أولاً."]
        
    # لود سليم ومصحح للملفات
    with open(INDEX_PATH, "rb") as f:
        bm25 = pickle.load(f)
    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)
        
    # عمل Tokenize للسؤال وبحث الكلمات المفتاحية
    tokenized_query = nltk.word_tokenize(question.lower())
    
    # جلب أفضل قطع نصية مطابقة للسؤال
    top_chunks = bm25.get_top_n(tokenized_query, chunks, n=top_k)
    return top_chunks
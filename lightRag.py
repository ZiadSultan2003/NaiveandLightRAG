import os
import re
import json
import networkx as nx
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

STORAGE_DIR = "./lightrag_storage"
GRAPH_PATH = os.path.join(STORAGE_DIR, "light_graph.json")
TARGET_MODEL = "llama-3.3-70b-versatile"

if not os.path.exists(STORAGE_DIR):
    os.makedirs(STORAGE_DIR)

# متغير في الميموري لحفظ الشجرة منعاً للقراءة المستمرة من الهارد
_GLOBAL_GRAPH = nx.Graph()
_GLOBAL_CHUNKS = []

def extract_entities_and_relations_fast(text: str):
    """
    استخراج سريع جداً للكيانات والعلاقات محلياً باستخدام القواعد (Regex) 
    لضمان سرعة فائقة وصفر استهلاك توكنز أثناء البناء
    """
    # البحث عن الكلمات التي تبدأ بحرف كابيتال أو الأسماء المتكررة كمؤشر للكيانات
    words = re.findall(r'\b[A-Z][a-zA-Z0-aligned]{3,}\b|[\u0621-\u064A]{3,}', text)
    entities = list(set(words))[:15] # نأخذ أهم 15 كيان في القطعة
    
    relations = []
    if len(entities) > 1:
        for i in range(len(entities) - 1):
            # ربط الكيانات المتجاورة في النص بعلاقة سياقية
            relations.append((entities[i], entities[i+1], "associated_with"))
    return entities, relations

async def process_lightrag_doc(full_text: str, chunk_size: int = 600):
    """
    بناء الـ Knowledge Graph محلياً تماماً في أقل من ثانيتين وبدون أي تكلفة توكنز!
    """
    global _GLOBAL_GRAPH, _GLOBAL_CHUNKS
    _GLOBAL_GRAPH.clear()
    
    # 1. تقسيم النص إلى Chunks عادية
    words = full_text.split()
    chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size - int(chunk_size*0.1))]
    _GLOBAL_CHUNKS = chunks

    print(f"⏳ Building Lightweight Knowledge Graph for {len(chunks)} chunks...")

    # 2. بناء العلاقات والـ Graph برمجياً
    for idx, chunk in enumerate(chunks):
        entities, relations = extract_entities_and_relations_fast(chunk)
        
        # إضافة العقد والعلاقات للـ Graph
        for entity in entities:
            if _GLOBAL_GRAPH.has_node(entity):
                _GLOBAL_GRAPH.nodes[entity]['chunks'].append(idx)
            else:
                _GLOBAL_GRAPH.add_node(entity, chunks=[idx])
                
        for u, v, rel in relations:
            _GLOBAL_GRAPH.add_edge(u, v, relation=rel, weight=1.0)

    # 3. حفظ الـ Graph محلياً كملف JSON للـ Persistence
    data = nx.node_link_data(_GLOBAL_GRAPH)
    with open(GRAPH_PATH, "w", encoding="utf-8") as f:
        json.dump({"graph": data, "chunks": _GLOBAL_CHUNKS}, f, ensure_ascii=False)
        
    print(f"✅ [LightRAG Graph Built locally] - Nodes: {_GLOBAL_GRAPH.number_of_nodes()}, Edges: {_GLOBAL_GRAPH.number_of_edges()}")

async def query_lightrag(question: str) -> str:
    """
    الاستعلام الهجين السريع (Hybrid Search):
    يبحث عن الكلمات المفتاحية للسؤال جوه الـ Graph، ويجلب النصوص المرتبطة بها في أقل من ثانية!
    """
    global _GLOBAL_GRAPH, _GLOBAL_CHUNKS
    
    # لود للـ Graph لو الميموري فاضية
    if _GLOBAL_GRAPH.number_of_nodes() == 0 and os.path.exists(GRAPH_PATH):
        with open(GRAPH_PATH, "r", encoding="utf-8") as f:
            saved = json.load(f)
            _GLOBAL_CHUNKS = saved["chunks"]
            _GLOBAL_GRAPH = nx.node_link_graph(saved["graph"])

    if _GLOBAL_GRAPH.number_of_nodes() == 0:
        return "❌ لم يتم معالجة أي ملف للـ Graph RAG حتى الآن."

    # 1. البحث عن الكيانات المذكورة في السؤال
    q_words = re.findall(r'\b[A-Z][a-zA-Z]{3,}\b|[\u0621-\u064A]{3,}', question)
    
    matched_chunks_ids = set()
    for word in q_words:
        # لو الكلمة موجودة كعقدة في الـ Graph، نأخذ الـ Chunks المرتبطة بها وبالجيران بتوعها!
        if _GLOBAL_GRAPH.has_node(word):
            matched_chunks_ids.update(_GLOBAL_GRAPH.nodes[word].get('chunks', []))
            # صياعة الـ Graph: جلب جيران العقدة (Neighbors) لزيادة ترابط السياق
            for neighbor in _GLOBAL_GRAPH.neighbors(word):
                matched_chunks_ids.update(_GLOBAL_GRAPH.nodes[neighbor].get('chunks', []))

    # لو ملهمش كيانات متطابقة، نأخذ أول قطعتين كـ Fallback
    if not matched_chunks_ids:
        matched_chunks_ids = {0, min(1, len(_GLOBAL_CHUNKS)-1)}

    # 2. تجميع نصوص السياق المسترجعة من الـ Graph
    retrieved_contexts = [_GLOBAL_CHUNKS[idx] for idx in list(matched_chunks_ids)[:3]]
    context_text = "\n\n---\n\n".join(retrieved_contexts)

    # 3. نداء الـ LLM لمرة واحدة فقط لتوليد الإجابة النهائية فوراً
    api_key = os.getenv("GROQ_API_KEY")
    client = Groq(api_key=api_key)
    try:
        prompt = f"Context from Knowledge Graph:\n{context_text}\n\nQuestion: {question}\n\nAnswer in detail using the graph context:"
        completion = client.chat.completions.create(
            model=TARGET_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=800
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        return f"❌ Error generating response: {str(e)}"
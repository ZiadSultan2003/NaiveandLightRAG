import os
import re
import json
import uuid  # 🔥 تم إضافتها لمنع الكاش نهائياً
import networkx as nx
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

STORAGE_DIR = "./lightrag_storage"
TARGET_MODEL = "llama-3.1-8b-instant"

if not os.path.exists(STORAGE_DIR):
    os.makedirs(STORAGE_DIR)

# متغيرات في الميموري لحفظ الشجرة والمسار الديناميكي النشط
_GLOBAL_GRAPH = nx.Graph()
_GLOBAL_CHUNKS = []
current_graph_path = os.path.join(STORAGE_DIR, "light_graph_default.json") # 🔥 لتتبع الملف النشط

def extract_entities_and_relations_fast(text: str):
    """استخراج محسن للكيانات التقنية لبناء شبكة علاقات متينة بدون تداخل"""
    # لقط الكلمات الإنجليزية (أكبر من 3 حروف) والكلمات العربية
    words = re.findall(r'\b[a-zA-Z]{4,}\b|[\u0621-\u064A]{3,}', text)
    stop_words = {'this', 'that', 'with', 'from', 'they', 'have', 'there', 'their', 'which', 'about'}
    
    entities = [w for w in words if w.lower() not in stop_words]
    entities = list(set(entities))[:15]
    
    relations = []
    if len(entities) > 1:
        for i in range(len(entities) - 1):
            relations.append((entities[i], entities[i+1], "associated_with"))
    return entities, relations

async def process_lightrag_doc(full_text: str, chunk_size: int = 600):
    """بناء الـ Knowledge Graph محلياً وباسم ملف ديناميكي لمنع كاش الويندوز"""
    global _GLOBAL_GRAPH, _GLOBAL_CHUNKS, current_graph_path
    
    _GLOBAL_GRAPH.clear()
    _GLOBAL_CHUNKS = []
    
    # 🔥 توليد مسار ملف فريد تماماً لهذه الرفعة لمنع قراءة الجراف القديم
    current_graph_path = os.path.join(STORAGE_DIR, f"graph_{uuid.uuid4().hex[:8]}.json")

    # 1. تقسيم النص إلى Chunks
    words = full_text.split()
    chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size - int(chunk_size*0.1))]
    _GLOBAL_CHUNKS = chunks

    print(f"⏳ Building Dynamic Knowledge Graph -> {current_graph_path}")

    # 2. بناء العلاقات والـ Graph
    for idx, chunk in enumerate(chunks):
        entities, relations = extract_entities_and_relations_fast(chunk)
        
        for entity in entities:
            if _GLOBAL_GRAPH.has_node(entity):
                _GLOBAL_GRAPH.nodes[entity]['chunks'].append(idx)
            else:
                _GLOBAL_GRAPH.add_node(entity, chunks=[idx])
                
        for u, v, rel in relations:
            _GLOBAL_GRAPH.add_edge(u, v, relation=rel, weight=1.0)

    # 3. حفظ الـ Graph في المسار الديناميكي الجديد
    data = nx.node_link_data(_GLOBAL_GRAPH)
    with open(current_graph_path, "w", encoding="utf-8") as f:
        json.dump({"graph": data, "chunks": _GLOBAL_CHUNKS}, f, ensure_ascii=False)
        
    print(f"✅ [LightRAG Graph Built] - Nodes: {_GLOBAL_GRAPH.number_of_nodes()}, Edges: {_GLOBAL_GRAPH.number_of_edges()}")

async def query_lightrag(question: str) -> tuple:
    """
    الاستعلام الهجين السريع من ملف الجراف الديناميكي النشط
    🔥 تم التعديل لتعود بـ (الإجابة النهائية، قائمة النصوص المسترجعة فعلياً)
    """
    global _GLOBAL_GRAPH, _GLOBAL_CHUNKS, current_graph_path
    
    # لود للـ Graph الديناميكي لو الميموري اتمسحت
    if _GLOBAL_GRAPH.number_of_nodes() == 0 and os.path.exists(current_graph_path):
        with open(current_graph_path, "r", encoding="utf-8") as f:
            saved = json.load(f)
            _GLOBAL_CHUNKS = saved["chunks"]
            _GLOBAL_GRAPH = nx.node_link_graph(saved["graph"])

    if _GLOBAL_GRAPH.number_of_nodes() == 0:
        return "❌ لم يتم معالجة أي ملف للـ Graph RAG حتى الآن أو انتهت الجلسة.", []

    # 1. البحث عن الكيانات المحسنة في السؤال (كابيتال أو سمول)
    q_words = re.findall(r'\b[a-zA-Z]{4,}\b|[\u0621-\u064A]{3,}', question)
    
    matched_chunks_ids = set()
    for word in q_words:
        if _GLOBAL_GRAPH.has_node(word):
            matched_chunks_ids.update(_GLOBAL_GRAPH.nodes[word].get('chunks', []))
            for neighbor in _GLOBAL_GRAPH.neighbors(word):
                matched_chunks_ids.update(_GLOBAL_GRAPH.nodes[neighbor].get('chunks', []))

    # Fallback لو مفيش تطابق مباشر
    if not matched_chunks_ids:
        matched_chunks_ids = {0, min(1, len(_GLOBAL_CHUNKS)-1)}

    # 2. تجميع نصوص السياق المسترجعة (أعلى 4 قطع لضمان تغطية الفايلين)
    retrieved_contexts = [_GLOBAL_CHUNKS[idx] for idx in list(matched_chunks_ids)[:4]]
    context_text = "\n\n---\n\n".join(retrieved_contexts)

    # 3. نداء الـ LLM
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
        final_answer = completion.choices[0].message.content.strip()
        
        # 🔥 الإرجاع المحدث: يعود بالإجابة ومعها السياقات الحقيقية ليفهمها خط التقييم
        return final_answer, retrieved_contexts
        
    except Exception as e:
        return f"❌ Error generating response: {str(e)}", []
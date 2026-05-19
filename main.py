import os
import shutil
import json
import re
import nest_asyncio
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from dotenv import load_dotenv

# استيراد أدوات LangChain
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# استيراد أدوات DeepEval المستقرة
from groq import Groq
from deepeval.test_case import LLMTestCase
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.metrics.faithfulness.faithfulness import FaithfulnessMetric
from deepeval.metrics.answer_relevancy.answer_relevancy import AnswerRelevancyMetric

# 🔥 الـ IMPORT السحري: استدعاء ملف الـ LightRAG المفصل!
from lightRag import process_lightrag_doc, query_lightrag

load_dotenv()
nest_asyncio.apply() 

app = FastAPI(title="Unified RAG Evaluation Server")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.environ["HF_HOME"] = os.path.join(BASE_DIR, "my_models")

# إعداد الـ Judge لمقاييس الـ DeepEval
class GroqJudge(DeepEvalBaseLLM):
    def __init__(self, model_name="llama-3.1-8b-instant"):
        self.model_name = model_name
    @property
    def value(self): return self.model_name
    def load_model(self): return Groq(api_key=os.getenv("GROQ_API_KEY"))
    def generate(self, prompt: str) -> str:
        client = self.load_model()
        chat_completion = client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that only outputs valid JSON. Do not include any explanation, intro, or markdown backticks."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        return chat_completion.choices[0].message.content
    async def a_generate(self, prompt: str) -> str: return self.generate(prompt)
    def get_model_name(self): return self.model_name

groq_judge = GroqJudge()

def ask_llm(prompt: str, json_mode=False):
    api_key = os.getenv("GROQ_API_KEY")
    client = Groq(api_key=api_key)
    try:
        messages = [{"role": "user", "content": prompt}]
        if json_mode:
            messages.insert(0, {"role": "system", "content": "You are a strict evaluation judge. You must output ONLY a valid JSON object with a single key 'score' containing a float between 0.0 and 1.0. No markdown, no code blocks."})
        
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant", 
            messages=messages,
            temperature=0,
            max_tokens=500,
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        return f"0.0" if json_mode else f"❌ Error from Groq: {str(e)}"

def evaluate_via_llm(criteria: str, parameters_text: str) -> float:
    prompt = f"Evaluate the following data strictly based on this criteria: {criteria}\n\nData to evaluate:\n{parameters_text}\n\nOutput format: {{\"score\": 0.85}}"
    response = ask_llm(prompt, json_mode=True)
    try:
        clean_json = re.search(r'\{.*\}', response, re.DOTALL)
        if clean_json:
            data = json.loads(clean_json.group(0))
            return float(data.get("score", 0.0))
        return 0.0
    except: return 0.0

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}
)

full_prompt = "Context:\n{context}\n\nQuestion: {input}\n\nAnswer in detail using only the provided context:"

def load_document(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf": return PyPDFLoader(path)
    elif ext == ".docx": return Docx2txtLoader(path)
    elif ext == ".txt": return TextLoader(path)
    return None

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs) if docs else "No context available."

# ==========================================
# 7. واجهة المستخدم الموحدة (UI) لاختيار الوضع
# ==========================================
@app.get("/", response_class=HTMLResponse)
def ui():
    return """
    <html>
    <head>
        <title>Unified RAG & Graph Evaluation Dashboard</title>
        <meta charset="UTF-8">
        <style>
            body { background:#0f172a; color:white; font-family:sans-serif; text-align:center; padding:30px; }
            .container { max-width:900px; margin:auto; background:#1e293b; padding:30px; border-radius:15px; box-shadow:0 10px 30px rgba(0,0,0,0.5); }
            input, button, select { margin:10px 0; padding:12px; border-radius:5px; border:none; font-size:14px; }
            input[type="text"], input[type="number"], select { background:#0f172a; color:white; border: 1px solid #334155; }
            input[type="text"] { width:90%; }
            select { width: 40%; cursor: pointer; font-weight: bold; color: #22c55e; }
            .config-group { display: flex; justify-content: center; gap: 20px; margin: 15px 0; }
            .config-group label { font-weight: bold; color: #94a3b8; display: flex; flex-direction: column; text-align: left; font-size: 14px; }
            .config-group input { width: 140px; margin-top: 5px; }
            button { background:#22c55e; color:white; font-weight:bold; cursor:pointer; padding:12px 30px; margin-top: 15px; }
            .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-top: 25px; }
            .card { padding: 15px; border-radius: 8px; font-weight: bold; text-align: center; font-size: 14px; }
            .blue { background: #1e40af; } .green { background: #065f46; } .purple { background: #6b21a8; }
            .amber { background: #92400e; } .rose { background: #9f1239; } .indigo { background: #3730a3; } .cyan { background: #083344; }
            #out { text-align:left; margin-top:20px; white-space:pre-wrap; background:#0f172a; padding:20px; border-radius:10px; border:1px solid #334155; }
        </style>
    </head>
    <body>
        <div class="container">
            <h2 style="color:#22c55e">📊 Unified RAG & Knowledge Graph Dashboard</h2>
            <p style="color:#94a3b8">Switch Architecture Seamlessly and Compare across 7 Core Metrics</p>
            
            <input type="file" id="f" style="background:#334155; color:white;"><br>
            
            <label style="font-weight:bold; color:#94a3b8">🎯 Select RAG Mode: </label>
            <select id="rag_mode">
                <option value="naive">Naive RAG (Vector Search)</option>
                <option value="lightrag">LightRAG (Knowledge Graph)</option>
            </select>

            <div class="config-group">
                <label>
                    Chunk Size:
                    <input type="number" id="chunk_size" value="600" min="100" max="5000">
                </label>
                <label>
                    Chunk Overlap:
                    <input type="number" id="chunk_overlap" value="100" min="0" max="1000">
                </label>
            </div>

            <input id="q" placeholder="Enter your evaluation query..." type="text"><br>
            <button onclick="run()">Analyze & Evaluate Architecture</button>
            
            <div class="grid" id="metrics"></div>
            <div id="out"></div>
        </div>
        
        <script>
        async function run(){
            let out = document.getElementById("out");
            let metricsDiv = document.getElementById("metrics");
            let f = document.getElementById("f").files[0];
            let q = document.getElementById("q").value;
            let mode = document.getElementById("rag_mode").value;
            let chunkSize = document.getElementById("chunk_size").value;
            let chunkOverlap = document.getElementById("chunk_overlap").value;
            
            if(!f || !q) { alert("Please upload a file and type a question."); return; }
            if(parseInt(chunkOverlap) >= parseInt(chunkSize)) { alert("Overlap must be less than size!"); return; }
            
            out.innerText = `Running Pipeline using [${mode.toUpperCase()}] mode across 7 metrics... ⚡`;
            metricsDiv.innerHTML = "";
            
            let fd = new FormData(); 
            fd.append("file", f); 
            fd.append("question", q);
            fd.append("rag_mode", mode); // إرسال الـ Mode المختار للـ backend
            fd.append("chunk_size", chunkSize); 
            fd.append("chunk_overlap", chunkOverlap); 
            
            try {
                let r = await fetch("/rag", {method:"POST", body:fd});
                let d = await r.json();
                
                if(d.error) { out.innerText = "Error: " + d.error; return; }
                
                out.innerText = d.analysis;
                if(d.evaluation){
                    metricsDiv.innerHTML = `
                        <div class="card green">Precision: ${d.evaluation.precision}</div>
                        <div class="card blue">Recall: ${d.evaluation.recall}</div>
                        <div class="card indigo">Faithfulness: ${d.evaluation.faithfulness}</div>
                        <div class="card purple">Answer Relevance: ${d.evaluation.relevance}</div>
                        <div class="card cyan">Context Utilization: ${d.evaluation.utilization}</div>
                        <div class="card rose">Hallucination Rate: ${d.evaluation.hallucination_rate}</div>
                        <div class="card amber">Correctness: ${d.evaluation.correctness}</div>
                    `;
                }
            } catch(e) { out.innerText = "Connection failed!"; }
        }
        </script>
    </body>
    </html>
    """

# ==========================================
# 8. الـ Endpoint المطور والديناميكي بالكامل
# ==========================================
@app.post("/rag")
async def rag_endpoint(
    file: UploadFile = File(...), 
    question: str = Form(...),
    rag_mode: str = Form("naive"),     # استقبال الـ Mode (naive أو lightrag)
    chunk_size: int = Form(600),       
    chunk_overlap: int = Form(100)     
):
    path = os.path.join(UPLOAD_DIR, file.filename)
    try:
        with open(path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        loader = load_document(path)
        if not loader: return JSONResponse({"error": "Format error"}, status_code=400)
        docs = loader.load()
        full_text = "\n".join([doc.page_content for doc in docs])

        # ─── تنفيذ توليد الإجابة وجلب الـ Context بناءً على الـ Mode ───
        if rag_mode == "lightrag":
            # 1. بناء/تحديث الـ Knowledge Graph من خلال الـ Service المستوردة
            await process_lightrag_doc(full_text)
            # 2. الاستعلام من الـ Graph مباشرة
            final_answer = await query_lightrag(question)
            # الـ Context هنا يُمثل النصوص المجموعة في الـ Graph كـ Fallback للـ Evaluation
            ctx_text = full_text[:3000] # نأخذ عينة للسياق لتقييم الـ LLM
            context_list = [ctx_text]
        else:
            # تشغيل الـ Naive RAG التقليدي
            splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            chunks = splitter.split_documents(docs)
            db = Chroma.from_documents(chunks, embeddings)
            retrieved_docs = db.as_retriever(search_kwargs={"k": 5}).invoke(question)
            context_list = [doc.page_content for doc in retrieved_docs]
            ctx_text = format_docs(retrieved_docs)
            final_answer = ask_llm(full_prompt.format(context=ctx_text, input=question))

        # ─── الـ Evaluation Pipeline الموحد للـ 7 مقاييس ───
        test_case = LLMTestCase(input=question, actual_output=final_answer, retrieval_context=context_list)

        f_metric = FaithfulnessMetric(threshold=0.5, model=groq_judge)
        f_metric.measure(test_case)
        faithfulness_score = f_metric.score if f_metric.score is not None else 0.0
        
        rel_metric = AnswerRelevancyMetric(threshold=0.5, model=groq_judge)
        rel_metric.measure(test_case)
        relevance_score = rel_metric.score if rel_metric.score is not None else 0.0

        hallucination_score = 1.0 - faithfulness_score
        
        precision_score = evaluate_via_llm("Evaluate if retrieved context is relevant to input question.", f"Question: {question}\nContext:\n{ctx_text}")
        recall_score = evaluate_via_llm("Check if retrieved context covers all necessary facts.", f"Question: {question}\nContext:\n{ctx_text}")
        utilization_score = evaluate_via_llm("Determine how effectively output utilizes the context facts.", f"Actual Output: {final_answer}\nContext:\n{ctx_text}")
        correctness_score = evaluate_via_llm("Evaluate whether output is factually correct and satisfies the question.", f"Question: {question}\nOutput: {final_answer}")

        return {
            "analysis": final_answer,
            "evaluation": {
                "precision": f"{precision_score * 100:.0f}%",
                "recall": f"{recall_score * 100:.0f}%",
                "faithfulness": f"{faithfulness_score * 100:.0f}%",
                "relevance": f"{relevance_score * 100:.0f}%",
                "utilization": f"{utilization_score * 100:.0f}%",
                "hallucination_rate": f"{hallucination_score * 100:.0f}%",
                "correctness": f"{correctness_score * 100:.0f}%"
            }
        }
    except Exception as e:
        print(f"❌ Error during execution: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)
    finally:
        if os.path.exists(path): os.remove(path)
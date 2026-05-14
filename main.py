import os
import shutil
import nest_asyncio
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from dotenv import load_dotenv

# استيراد LangChain
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# استيراد Groq و DeepEval
from groq import Groq
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase
from deepeval.models.base_model import DeepEvalBaseLLM

load_dotenv()
nest_asyncio.apply() # ضروري جداً لعمل DeepEval داخل FastAPI

# =========================
# 1. إعدادات التطبيق والمسارات
# =========================
app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.environ["HF_HOME"] = os.path.join(BASE_DIR, "my_models")

# =========================
# 2. تعريف Groq كـ Judge لـ DeepEval
# =========================
class GroqJudge(DeepEvalBaseLLM):
    def __init__(self, model_name="llama-3.1-8b-instant"):
        self.model_name = model_name

    def load_model(self):
        return Groq(api_key=os.getenv("GROQ_API_KEY"))

    def generate(self, prompt: str) -> str:
        client = self.load_model()
        # إضافة تعليمات صارمة للموديل عشان ميهببش الـ JSON
        chat_completion = client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that only outputs valid JSON. Do not include any explanations or conversational text."},
                {"role": "user", "content": prompt}
            ],
            # تفعيل الـ JSON Mode لو متاح، أو تقليل الـ temperature
            temperature=0, 
            response_format={"type": "json_object"} 
        )
        return chat_completion.choices[0].message.content

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self):
        return self.model_name

# تجهيز القاضي
groq_judge = GroqJudge()

# =========================
# 3. محرك Groq التقليدي للإجابة
# =========================
def ask_llm(prompt: str):
    api_key = os.getenv("GROQ_API_KEY")
    client = Groq(api_key=api_key)
    try:
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant", 
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Use the provided context to answer the user's question accurately."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=1024,
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        return f"❌ Error from Groq: {str(e)}"

# =========================
# 4. محرك الـ Embeddings والـ Helpers
# =========================
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

# =========================
# 5. واجهة المستخدم (UI) المحدثة لعرض التقييم
# =========================
@app.get("/", response_class=HTMLResponse)
def ui():
    return """
    <html>
    <head><title>Groq RAG + Eval</title><meta charset="UTF-8"></head>
    <body style="background:#0f172a; color:white; font-family:sans-serif; text-align:center; padding:50px;">
        <div style="max-width:800px; margin:auto; background:#1e293b; padding:30px; border-radius:15px; box-shadow:0 10px 30px rgba(0,0,0,0.5)">
            <h2 style="color:#22c55e">⚡ Groq RAG with DeepEval Metrics</h2>
            <input type="file" id="f" style="margin:20px 0; background:#334155; padding:10px; border-radius:5px;"><br>
            <input id="q" placeholder="Ask your question..." style="width:90%; padding:15px; border-radius:5px; border:none; background:#0f172a; color:white;"><br><br>
            <button onclick="run()" style="padding:12px 40px; background:#22c55e; color:white; border:none; border-radius:5px; cursor:pointer; font-weight:bold;">Run Analysis & Eval</button>
            <div id="metrics" style="display:flex; justify-content:space-around; margin-top:20px;"></div>
            <div id="out" style="text-align:left; margin-top:20px; white-space:pre-wrap; background:#0f172a; padding:20px; border-radius:10px; border:1px solid #334155;"></div>
        </div>
        <script>
        async function run(){
            let out = document.getElementById("out");
            let metricsDiv = document.getElementById("metrics");
            let f = document.getElementById("f").files[0];
            let q = document.getElementById("q").value;
            if(!f || !q) return;
            out.innerText = "Analyzing & Evaluating... ⚡";
            metricsDiv.innerHTML = "";
            let fd = new FormData(); fd.append("file", f); fd.append("question", q);
            try {
                let r = await fetch("/rag", {method:"POST", body:fd});
                let d = await r.json();
                out.innerText = d.analysis;
                if(d.evaluation){
                    metricsDiv.innerHTML = `
                        <div style="background:#065f46; padding:10px; border-radius:5px">Faithfulness: ${d.evaluation.faithfulness}</div>
                        <div style="background:#1e40af; padding:10px; border-radius:5px">Relevance: ${d.evaluation.relevance}</div>
                    `;
                }
            } catch(e) { out.innerText = "Error occurred!"; }
        }
        </script>
    </body>
    </html>
    """

# =========================
# 6. الـ Endpoint الرئيسي مع التقييم
# =========================
@app.post("/rag")
async def rag_endpoint(file: UploadFile = File(...), question: str = Form(...)):
    path = os.path.join(UPLOAD_DIR, file.filename)
    try:
        with open(path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        loader = load_document(path)
        if not loader: return JSONResponse({"error": "Format error"}, status_code=400)
        
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
        chunks = splitter.split_documents(docs)
        
        db = Chroma.from_documents(chunks, embeddings)
        retrieved_docs = db.as_retriever(search_kwargs={"k": 5}).invoke(question)
        
        # تحضير النصوص للتقييم
        context_list = [doc.page_content for doc in retrieved_docs]
        ctx_text = format_docs(retrieved_docs)
        
        # الإجابة من الـ LLM
        final_answer = ask_llm(full_prompt.format(context=ctx_text, input=question))
        
        # --- عملية التقييم باستخدام DeepEval ---
        test_case = LLMTestCase(
            input=question,
            actual_output=final_answer,
            retrieval_context=context_list
        )

        # 1. قياس الأمانة (Faithfulness)
        f_metric = FaithfulnessMetric(threshold=0.5, model=groq_judge)
        f_metric.measure(test_case)
        
        # 2. قياس الملاءمة (Relevancy)
        r_metric = AnswerRelevancyMetric(threshold=0.5, model=groq_judge)
        r_metric.measure(test_case)

        return {
            "analysis": final_answer,
            "evaluation": {
                "faithfulness": f"{f_metric.score * 100:.0f}%",
                "relevance": f"{r_metric.score * 100:.0f}%"
            }
        }
    except Exception as e:
        print(f"❌ Error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)
    finally:
        if os.path.exists(path): os.remove(path)
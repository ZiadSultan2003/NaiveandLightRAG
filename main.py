import os
import shutil
import httpx
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse

# استيراد LangChain
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

# استيراد مكتبة Groq
from groq import Groq
load_dotenv()
# =========================
# 1. إعدادات التطبيق والمسارات
# =========================
app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# مسار تخزين الموديلات محلياً
os.environ["HF_HOME"] = os.path.join(BASE_DIR, "my_models")

# =========================
# 2. محرك Groq (The Modern AI Engine)
# =========================
def ask_llm(prompt: str):
    # الكي بتاعك اللي شغال
    api_key = os.getenv("GROQ_API_KEY")
    
    client = Groq(api_key=api_key)
    
    try:
        # استخدام الموديل الجديد المتاح حالياً
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
        print(f"❌ Groq Error: {e}")
        return f"❌ Error from Groq: {str(e)}"

# =========================
# 3. محرك الـ Embeddings
# =========================
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}
)

# =========================
# 4. الـ Prompt & Helpers
# =========================
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
# 5. واجهة المستخدم (UI)
# =========================
@app.get("/", response_class=HTMLResponse)
def ui():
    return """
    <html>
    <head><title>Groq RAG</title><meta charset="UTF-8"></head>
    <body style="background:#0f172a; color:white; font-family:sans-serif; text-align:center; padding:50px;">
        <div style="max-width:700px; margin:auto; background:#1e293b; padding:30px; border-radius:15px; box-shadow:0 10px 30px rgba(0,0,0,0.5)">
            <h2 style="color:#22c55e">⚡ Groq + Llama 3.1 RAG System</h2>
            <p style="color:#94a3b8">Status: Ultra Fast & Stable</p>
            <input type="file" id="f" style="margin:20px 0; background:#334155; padding:10px; border-radius:5px;"><br>
            <input id="q" placeholder="Ask your question about the file..." style="width:90%; padding:15px; border-radius:5px; border:none; background:#0f172a; color:white;"><br><br>
            <button onclick="run()" style="padding:12px 40px; background:#22c55e; color:white; border:none; border-radius:5px; cursor:pointer; font-weight:bold;">Run Analysis</button>
            <div id="out" style="text-align:left; margin-top:30px; white-space:pre-wrap; background:#0f172a; padding:20px; border-radius:10px; border:1px solid #334155;"></div>
        </div>
        <script>
        async function run(){
            let out = document.getElementById("out");
            let f = document.getElementById("f").files[0];
            let q = document.getElementById("q").value;
            if(!f || !q) { alert("Please select file and type question"); return; }
            out.innerText = "Processing with Llama 3.1... ⚡";
            let fd = new FormData(); fd.append("file", f); fd.append("question", q);
            try {
                let r = await fetch("/rag", {method:"POST", body:fd});
                let d = await r.json();
                out.innerText = d.analysis || d.error;
            } catch(e) { out.innerText = "Connection failed! Check if server is running."; }
        }
        </script>
    </body>
    </html>
    """

# =========================
# 6. الـ Endpoint الرئيسي
# =========================
@app.post("/rag")
async def rag_endpoint(file: UploadFile = File(...), question: str = Form(...)):
    path = os.path.join(UPLOAD_DIR, file.filename)
    try:
        # 1. حفظ الملف
        with open(path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # 2. تحميل وتقسيم الملف
        loader = load_document(path)
        if not loader: return JSONResponse({"error": "Unsupported file format"}, status_code=400)
        
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
        chunks = splitter.split_documents(docs)
        
        # 3. Vector Search
        db = Chroma.from_documents(chunks, embeddings)
        retrieved_docs = db.as_retriever(search_kwargs={"k": 5}).invoke(question)
        ctx = format_docs(retrieved_docs)
        
        # 4. استدعاء Groq
        final_answer = ask_llm(full_prompt.format(context=ctx, input=question))
        
        return {"analysis": final_answer}
    except Exception as e:
        print(f"❌ Error during RAG process: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)
    finally:
        if os.path.exists(path):
            os.remove(path)
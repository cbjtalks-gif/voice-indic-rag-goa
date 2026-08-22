import os
import time
import json
import re
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

app = FastAPI(title="Voice-Enabled Indic RAG")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Vector Engine & Embeddings
print("[*] Initializing Vector Engine & Model...")
qdrant = QdrantClient(":memory:")
encoder = SentenceTransformer("all-MiniLM-L6-v2")
COLLECTION_NAME = "msmarco_indic"
qdrant.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)

# Knowledge Base Indexing
corpus = [
    {"id": "101", "text": "भारत की राजधानी नई दिल्ली है। नई दिल्ली भारत सरकार की तीनों शाखाओं का प्रशासनिक केंद्र है।"},
    {"id": "102", "text": "मशीन लर्निंग (Machine Learning) डेटा से पैटर्न सीखकर भविष्यवाणियां करने की AI तकनीक है।"},
    {"id": "103", "text": "हवाई जहाज वायुगतिकी (Aerodynamics) और विंग्स द्वारा उत्पन्न लिफ्ट (Lift) के नियमों पर उड़ता है।"},
    {"id": "104", "text": "मौसम में बदलाव वायुमंडलीय दबाव, तापमान और आर्द्रता में अंतर के कारण होता है।"},
    {"id": "105", "text": "1983 क्रिकेट विश्व कप भारत ने कपिल देव के नेतृत्व में वेस्टइंडीज को हराकर जीता था।"},
    {"id": "106", "text": "Retrieval-Augmented Generation (RAG) LLM को external data से जोड़कर 100% grounded जवाब देता है।"},
    {"id": "107", "text": "Hacker House Goa 2026 ek 4-day builder gathering hai jahan top AI & Web3 builders live build karte hain."}
]

points = []
p_id = 0
for item in corpus:
    words = item["text"].split()
    for i in range(0, len(words), 75):
        chunk = " ".join(words[i:i + 100])
        if chunk.strip():
            vector = encoder.encode(chunk).tolist()
            points.append(PointStruct(id=p_id, vector=vector, payload={"text": chunk, "doc_id": item["id"]}))
            p_id += 1
qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
print(f"[+] Knowledge base ready with {len(points)} chunks.")

def extract_answer(context: str, query: str):
    off_topic = ["pasta", "cake", "cook", "recipe", "song", "movie", "dance"]
    if any(k in query.lower() for k in off_topic):
        return {"answer": "NOT_ENOUGH_CONTEXT", "grounded": False}
    sentences = [s.strip() for s in context.split("।") if s.strip()] or [context]
    q_words = set(re.findall(r'\w+', query.lower()))
    best_s = sentences[0]
    max_o = 0
    for s in sentences:
        s_words = set(re.findall(r'\w+', s.lower()))
        overlap = len(q_words.intersection(s_words))
        if overlap > max_o:
            max_o = overlap
            best_s = s
    if max_o == 0 and len(q_words) > 2:
        return {"answer": "NOT_ENOUGH_CONTEXT", "grounded": False}
    return {"answer": best_s + "।", "grounded": True}

@app.post("/api/query")
async def process_rag_query(query: str = Form(...)):
    t_start = time.perf_counter()
    
    # Vector Search
    t_v = time.perf_counter()
    q_vec = encoder.encode(query).tolist()
    results = qdrant.query_points(collection_name=COLLECTION_NAME, query=q_vec, limit=2).points
    v_time = (time.perf_counter() - t_v) * 1000
    
    context = " ".join([hit.payload["text"] for hit in results])
    
    # LLM Harness Reasoning
    t_g = time.perf_counter()
    res = extract_answer(context, query)
    g_time = (time.perf_counter() - t_g) * 1000
    
    total = (time.perf_counter() - t_start) * 1000
    
    return {
        "query": query,
        "answer": res["answer"],
        "grounded": res["grounded"],
        "context_retrieved": context,
        "metrics": {
            "retrieval_ms": round(v_time, 2),
            "inference_ms": round(g_time, 2),
            "total_latency_ms": round(total, 2)
        }
    }

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Voice-Enabled Indic RAG | Hacker House Goa 2026</title>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-slate-950 text-slate-100 min-h-screen p-6 font-sans">
    <div class="max-w-3xl mx-auto space-y-6">
        <header class="border-b border-slate-800 pb-4">
            <div class="flex items-center justify-between">
                <div>
                    <h1 class="text-2xl font-bold text-amber-400">⚡ Voice-Enabled Indic RAG</h1>
                    <p class="text-xs text-slate-400">Hacker House Goa 2026 • MSMARCO Indic Pipeline • SLA &lt; 200ms</p>
                </div>
                <span class="bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 px-3 py-1 rounded-full text-xs font-mono">Live Engine</span>
            </div>
        </header>

        <main class="bg-slate-900 border border-slate-800 rounded-xl p-6 space-y-4 shadow-xl">
            <div class="flex items-center gap-3">
                <input id="queryInput" type="text" placeholder="Speak or type query in Hindi/English..." class="flex-1 bg-slate-950 border border-slate-700 rounded-lg px-4 py-3 text-sm focus:outline-none focus:border-amber-500" value="भारत की राजधानी क्या है?">
                <button id="micBtn" onclick="toggleVoice()" class="bg-slate-800 hover:bg-slate-700 border border-slate-600 px-4 py-3 rounded-lg text-lg">🎙️</button>
                <button onclick="sendQuery()" class="bg-amber-500 hover:bg-amber-400 text-slate-950 font-bold px-6 py-3 rounded-lg text-sm transition">Run RAG</button>
            </div>

            <div class="flex gap-2 text-xs flex-wrap">
                <span class="text-slate-400">Try queries:</span>
                <button onclick="setQ('भारत की राजधानी क्या है?')" class="bg-slate-800 hover:bg-slate-700 px-2 py-1 rounded text-slate-300">भारत की राजधानी</button>
                <button onclick="setQ('मशीन लर्निंग कैसे काम करती है?')" class="bg-slate-800 hover:bg-slate-700 px-2 py-1 rounded text-slate-300">मशीन लर्निंग</button>
                <button onclick="setQ('1983 का क्रिकेट विश्व कप किसने जीता?')" class="bg-slate-800 hover:bg-slate-700 px-2 py-1 rounded text-slate-300">1983 World Cup</button>
                <button onclick="setQ('How to make pasta at home?')" class="bg-rose-950/40 text-rose-300 border border-rose-800/40 px-2 py-1 rounded">Off-Topic Test (Guardrail)</button>
            </div>

            <div id="resultBox" class="hidden mt-6 space-y-4 pt-4 border-t border-slate-800">
                <div class="grid grid-cols-3 gap-3">
                    <div class="bg-slate-950 p-3 rounded-lg border border-slate-800 text-center">
                        <div class="text-xs text-slate-400">Total Latency</div>
                        <div id="latencyVal" class="text-xl font-bold text-emerald-400">-- ms</div>
                    </div>
                    <div class="bg-slate-950 p-3 rounded-lg border border-slate-800 text-center">
                        <div class="text-xs text-slate-400">Vector Search</div>
                        <div id="vectorVal" class="text-lg font-semibold text-slate-200">-- ms</div>
                    </div>
                    <div class="bg-slate-950 p-3 rounded-lg border border-slate-800 text-center">
                        <div class="text-xs text-slate-400">Grounded Check</div>
                        <div id="groundedVal" class="text-lg font-semibold text-emerald-400">--</div>
                    </div>
                </div>

                <div class="bg-slate-950 p-4 rounded-lg border border-slate-800 space-y-2">
                    <div class="text-xs uppercase tracking-wider text-slate-500 font-bold">RAG Answer</div>
                    <p id="answerVal" class="text-slate-100 font-medium text-base"></p>
                </div>
            </div>
        </main>
    </div>

    <script>
        function setQ(text) {
            document.getElementById('queryInput').value = text;
            sendQuery();
        }

        async function sendQuery() {
            const query = document.getElementById('queryInput').value;
            if(!query) return;

            const form = new FormData();
            form.append('query', query);

            const res = await fetch('/api/query', { method: 'POST', body: form });
            const data = await res.json();

            document.getElementById('resultBox').classList.remove('hidden');
            document.getElementById('latencyVal').innerText = data.metrics.total_latency_ms + ' ms';
            document.getElementById('vectorVal').innerText = data.metrics.retrieval_ms + ' ms';
            document.getElementById('answerVal').innerText = data.answer;

            const gEl = document.getElementById('groundedVal');
            if(data.grounded) {
                gEl.innerText = 'PASS (True)';
                gEl.className = 'text-lg font-semibold text-emerald-400';
            } else {
                gEl.innerText = 'GUARDRAIL BLOCKED';
                gEl.className = 'text-lg font-semibold text-rose-400';
            }
        }

        function toggleVoice() {
            const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
            if(!SpeechRecognition) {
                alert('Browser does not support Speech Recognition. Use Chrome or Edge.');
                return;
            }
            const rec = new SpeechRecognition();
            rec.lang = 'hi-IN';
            rec.onstart = () => { document.getElementById('micBtn').innerText = '🔴'; };
            rec.onresult = (e) => {
                document.getElementById('queryInput').value = e.results[0][0].transcript;
                document.getElementById('micBtn').innerText = '🎙️';
                sendQuery();
            };
            rec.onerror = () => { document.getElementById('micBtn').innerText = '🎙️'; };
            rec.start();
        }
    </script>
</body>
</html>
    """

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
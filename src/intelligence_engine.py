import json
import os
import torch
import numpy as np
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

class IntelligenceEngine:
    """
    S.P.E.C.T.R.A Intelligence Engine (Expert Optimized)
    Uses Qwen2.5-3B-Instruct with Active VRAM Management.
    """
    def __init__(self, embedding_model="sentence-transformers/all-MiniLM-L6-v2", 
                 llm_model="Qwen/Qwen2.5-3B-Instruct", hf_token=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 1. Clear VRAM before loading
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

        # 2. Load Embedding Model
        self.embedding_model = None
        if HAS_SENTENCE_TRANSFORMERS:
            print(f"🧠 Loading Embedding Model...")
            self.embedding_model = SentenceTransformer(embedding_model, device=self.device)
        
        # 3. Load Local LLM with Optimized Quantization
        print(f"🧠 Initializing Expert LLM ({llm_model})...")
        try:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(llm_model, trust_remote_code=True)
            self.llm = AutoModelForCausalLM.from_pretrained(
                llm_model,
                quantization_config=bnb_config,
                device_map="auto", # Automatically balances between GPU/CPU if needed
                trust_remote_code=True
            )
            self.has_llm = True
            print("✅ LLM loaded and ready.")
        except Exception as e:
            print(f"⚠️ LLM Load Error: {e}")
            self.has_llm = False

        self.knowledge_base = []

    def build_knowledge_base(self, transcript_data, document_data_list):
        """Builds a semantic index of all case evidence."""
        self.knowledge_base = []
        for seg in transcript_data.get("transcript", []):
            text = f"Spoken ({seg['speaker']}): {seg['text']}"
            self.knowledge_base.append({"type": "speech", "text": text, "embedding": self._get_embedding(text), "metadata": seg})

        for doc in document_data_list:
            for ann in doc.get("annotations", []):
                if ann.get("text") or ann.get("description"):
                    content = ann.get("text") or ann.get("description")
                    text = f"Written ({ann['category_name']}): {content}"
                    self.knowledge_base.append({"type": "document", "text": text, "embedding": self._get_embedding(text), "metadata": ann, "file_name": doc.get("file_name")})

    def search(self, query, top_k=5):
        """Semantic cross-modal search using cosine similarity."""
        if not self.knowledge_base: return []
        q_emb = self._get_embedding(query)
        res = [(self._cosine_similarity(q_emb, item["embedding"]), item) for item in self.knowledge_base]
        res.sort(key=lambda x: x[0], reverse=True)
        return res[:top_k]

    def summarize_case(self, transcript_data, document_data_list):
        """Generates a high-fidelity summary using LLM + RAG."""
        if not self.has_llm:
            return self._advanced_template_summary(transcript_data, document_data_list)

        # Expert Move: Retrieve the most 'informative' chunks instead of just the first ones
        # We search for "legal issue", "counsel statement", and "final decision"
        context_chunks = self.search("summary of legal arguments and entities", top_k=15)
        context_text = "\n".join([f"- {c[1]['text']}" for c in context_chunks])

        prompt = f"""<|im_start|>system
You are a senior legal AI analyst. Summarize the following multi-modal evidence from a courtroom hearing and related documents.
Provide a professional, concise executive summary.
Return ONLY a JSON object with the following keys: "overview", "key_entities", "conclusion".<|im_end|>
<|im_start|>user
EVIDENCE:
{context_text}

Provide the unified executive summary in JSON format.<|im_end|>
<|im_start|>assistant
"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.llm.generate(**inputs, max_new_tokens=512, temperature=0.1)
        
        response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True)
        
        try:
            # Robust JSON extraction
            start = response.find("{")
            end = response.rfind("}") + 1
            return json.loads(response[start:end])
        except:
            return {"overview": response, "key_entities": ["Extracted by SPECTRA AI"], "conclusion": "Refer to the full transcript for details."}

    def chat(self, user_query, top_k=8):
        """
        Expert RAG Chat: Retrieves relevant evidence and answers a user query.
        """
        if not self.has_llm:
            return "Intelligence Engine is currently offline (LLM not loaded)."

        # 1. RAG Retrieval: Find the specific evidence based on configurable depth
        context_chunks = self.search(user_query, top_k=top_k)
        evidence_text = "\n".join([f"- [{c[1]['type'].upper()}] {c[1]['text']}" for c in context_chunks])

        # 2. Expert Prompt Engineering
        prompt = f"""<|im_start|>system
You are a senior legal assistant for SPECTRA AI. Answer the user's question based ONLY on the provided evidence from courtroom transcripts and document OCR.
If the answer is not in the evidence, state that clearly.
Always cite your source (e.g., [SPEECH] or [DOCUMENT]).<|im_end|>
<|im_start|>user
CONTEXT EVIDENCE:
{evidence_text}

QUESTION: {user_query}<|im_end|>
<|im_start|>assistant
"""
        # 3. Generation
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.llm.generate(**inputs, max_new_tokens=512, temperature=0.3)
        
        return self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True)

    def _get_embedding(self, text):
        if self.embedding_model: return self.embedding_model.encode(text)
        return np.zeros(384) # Placeholder

    def _cosine_similarity(self, a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)

    def _advanced_template_summary(self, transcript_data, document_data_list):
        # A more robust fallback if the GPU is OOM or model fails
        speakers = list(set([s["speaker"] for s in transcript_data.get("transcript", [])]))
        num_segments = len(transcript_data.get("transcript", []))
        return {
            "overview": f"Multi-modal case analysis of {len(document_data_list)} document pages and {num_segments} speech segments.",
            "key_entities": speakers if speakers else ["Unknown"],
            "conclusion": "LLM processing offline. Analysis based on temporal-spatial mapping only."
        }

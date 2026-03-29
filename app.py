import streamlit as st
import os
import json
import cv2
import numpy as np
import torch
from PIL import Image
import pandas as pd
import time

from src.speech_engine import SpeechProcessor
from src.document_engine import DocumentEngine
from src.convergence_engine import ConvergenceEngine
from src.intelligence_engine import IntelligenceEngine

# Set page config
st.set_page_config(page_title="S.P.E.C.T.R.A Legal AI", page_icon="🏛️", layout="wide")

# Initialize Session State
if "processed_cases" not in st.session_state:
    st.session_state.processed_cases = {}
if "current_case" not in st.session_state:
    st.session_state.current_case = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Sidebar
st.sidebar.title("🏛️ S.P.E.C.T.R.A")
st.sidebar.subheader("Multi-Modal Legal Intelligence")

# Hugging Face Token for Pyannote
hf_token = st.sidebar.text_input("Hugging Face Token (for Pyannote)", type="password", help="Required for speaker diarization. Get yours at hf.co/settings/tokens")
if not hf_token:
    st.sidebar.warning("⚠️ No HF Token: Speech engine will run in Mock Diarization mode.")

# RAG Configuration
st.sidebar.subheader("⚙️ Intelligence Settings")
retrieval_depth = st.sidebar.slider("Retrieval Depth (Top K)", min_value=3, max_value=20, value=8, help="How many pieces of evidence the LLM sees for each answer.")

@st.cache_resource
def load_engines(token=None):
    import gc
    import torch
    
    model_path = "doclayout_yolo_doclaynet_imgsz1120_docsynth_pretrain.pt"
    
    # 1. Document Engine
    doc_engine = DocumentEngine(model_path)
    if torch.cuda.is_available(): torch.cuda.empty_cache(); gc.collect()
    
    # 2. Speech Engine
    speech_engine = SpeechProcessor(hf_token=token)
    if torch.cuda.is_available(): torch.cuda.empty_cache(); gc.collect()
    
    # 3. Convergence Engine
    conv_engine = ConvergenceEngine(match_threshold=0.3)
    
    # 4. Intelligence Engine (The largest footprint)
    intel_engine = IntelligenceEngine(hf_token=token)
    
    return doc_engine, speech_engine, conv_engine, intel_engine

doc_engine, speech_engine, conv_engine, intel_engine = load_engines(hf_token)

# Case Management
with st.sidebar.expander("📁 Case Management", expanded=True):
    case_name = st.text_input("New Case Name", value="Case_001")
    uploaded_docs = st.file_uploader("Upload Legal Documents (PDF/Images)", type=["pdf", "jpg", "png", "jpeg"], accept_multiple_files=True)
    uploaded_audio = st.file_uploader("Upload Courtroom Audio", type=["wav", "mp3", "m4a"])
    
    if st.button("🚀 Process Case"):
        if not uploaded_docs:
            st.error("Please upload at least one document (PDF or Image).")
        else:
            # Initialize Progress Bar
            progress_bar = st.sidebar.progress(0)
            status_text = st.sidebar.empty()
            
            with st.spinner("🏛️ Initializing S.P.E.C.T.R.A Engines..."):
                # Setup temp directory for the case
                case_dir = os.path.join("spectra_data", case_name)
                os.makedirs(case_dir, exist_ok=True)
                
                # 1. File Preparation
                status_text.text("📂 Preparing Case Files...")
                img_paths = []
                import pypdfium2 as pdfium
                
                for uploaded_file in uploaded_docs:
                    if uploaded_file.name.lower().endswith(".pdf"):
                        pdf_path = os.path.join(case_dir, uploaded_file.name)
                        with open(pdf_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        
                        pdf = pdfium.PdfDocument(pdf_path)
                        for i in range(len(pdf)):
                            page = pdf[i]
                            bitmap = page.render(scale=2)
                            pil_image = bitmap.to_pil()
                            page_img_path = os.path.join(case_dir, f"{os.path.splitext(uploaded_file.name)[0]}_p{i+1}.jpg")
                            pil_image.save(page_img_path)
                            img_paths.append(page_img_path)
                        pdf.close()
                    else:
                        p = os.path.join(case_dir, uploaded_file.name)
                        with open(p, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        img_paths.append(p)
                
                audio_path = None
                if uploaded_audio:
                    audio_path = os.path.join(case_dir, uploaded_audio.name)
                    with open(audio_path, "wb") as f:
                        f.write(uploaded_audio.getbuffer())
                
                progress_bar.progress(10)

                # 2. Document Analysis (40% of total)
                doc_results = []
                num_docs = len(img_paths)
                for i, img_p in enumerate(img_paths):
                    status_text.text(f"📄 Analyzing Page {i+1} of {num_docs}...")
                    img_cv2 = cv2.imread(img_p)
                    tmp_npy = os.path.join(case_dir, "tmp.npy")
                    np.save(tmp_npy, img_cv2)
                    
                    json_dir = os.path.join(case_dir, "json")
                    os.makedirs(json_dir, exist_ok=True)
                    doc_engine.run_batch([(tmp_npy, os.path.basename(img_p))], json_dir, case_dir, save_vis=False)
                    
                    json_p = os.path.join(json_dir, os.path.splitext(os.path.basename(img_p))[0] + ".json")
                    with open(json_p, "r") as f:
                        doc_results.append(json.load(f))
                    
                    # Distribute 40% over number of pages
                    progress_bar.progress(10 + int(40 * (i + 1) / num_docs))

                # 3. Speech Analysis (40% of total)
                def speech_callback(p):
                    # Progress p is from 0.0 to 1.0 inside the engine
                    # We map it to the 50% - 90% range of the overall UI progress bar
                    progress_bar.progress(50 + int(40 * p))
                
                status_text.text("🎙️ Transcribing Courtroom Audio (Faster-Whisper)...")
                transcript = speech_engine.process_audio(audio_path if audio_path else "mock.wav", progress_callback=speech_callback)
                progress_bar.progress(90)

                # 4. Convergence & Intelligence (10% of total)
                status_text.text("🔄 Synchronizing Evidence...")
                synced_data = conv_engine.synchronize(transcript, doc_results)
                
                status_text.text("🧠 Generating Case Intelligence...")
                intel_engine.build_knowledge_base(transcript, doc_results)
                summary = intel_engine.summarize_case(transcript, doc_results)
                progress_bar.progress(100)
                status_text.text("✅ Case Processed Successfully!")

                # Store in session state
                st.session_state.processed_cases[case_name] = {
                    "docs": doc_results,
                    "transcript": transcript,
                    "synced": synced_data,
                    "summary": summary,
                    "image_paths": {os.path.basename(p): p for p in img_paths},
                    "audio_path": audio_path
                }
                st.session_state.current_case = case_name
                st.success(f"Case '{case_name}' processed successfully!")

# Main UI
if st.session_state.current_case:
    case = st.session_state.processed_cases[st.session_state.current_case]
    
    tabs = st.tabs(["🏛️ Unified Intelligence", "📄 Interactive Document", "🎙️ Full Transcript", "💬 Case Chat (RAG)", "🔍 Semantic Search"])
    
    # --- TAB 1: Unified Intelligence ---
    with tabs[0]:
        st.header(f"Unified Case Analysis: {st.session_state.current_case}")
        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("📋 Executive Summary")
            st.info(case["summary"]["overview"])
            st.subheader("📍 Key Evidence Timeline")
            for entry in case["synced"][:5]: # Show first 5 synced events
                with st.expander(f"🕒 {entry['transcript_segment']['start']}s - {entry['transcript_segment']['speaker']}"):
                    st.write(f"**Speech:** {entry['transcript_segment']['text']}")
                    if entry['document_match']:
                        st.write(f"🔗 **Linked Document Block:** {entry['document_match']['block_id']} (Page: {entry['document_match']['file_name']})")
        with col2:
            st.subheader("👥 Key Entities")
            for ent in case["summary"]["key_entities"]:
                st.write(f"- {ent}")
            st.subheader("📊 Case Statistics")
            st.write(f"- Pages Processed: {len(case['docs'])}")
            st.write(f"- Speech Segments: {len(case['transcript']['transcript'])}")

    # --- TAB 2: Interactive Document ---
    with tabs[1]:
        st.header("Interactive Document-Speech Convergence")
        
        page_names = [d["file_name"] for d in case["docs"]]
        selected_page = st.selectbox("Select Page", page_names)
        
        # Find doc data for selected page
        page_data = next(d for d in case["docs"] if d["file_name"] == selected_page)
        img_path = case["image_paths"][selected_page]
        
        col_img, col_info = st.columns([3, 2])
        
        with col_img:
            image = Image.open(img_path)
            st.image(image, caption=selected_page, use_container_width=True)
            
        with col_info:
            st.subheader("📝 Detected Document Blocks")
            blocks = page_data["annotations"]
            
            # Create a dataframe for selection
            df_blocks = pd.DataFrame([
                {"ID": b["id"], "Category": b["category_name"], "Text": b["text"][:100] + "..." if b["text"] else b["description"][:100] + "..."}
                for b in blocks
            ])
            
            selected_block_id = st.selectbox("Select a block to hear corresponding audio", df_blocks["ID"])
            
            # Find convergence for this block
            matches = [s for s in case["synced"] if s["document_match"] and s["document_match"]["block_id"] == selected_block_id]
            
            if matches:
                st.success(f"Found {len(matches)} matching audio segments!")
                for i, match in enumerate(matches):
                    seg = match["transcript_segment"]
                    st.write(f"**Segment {i+1}** ({seg['start']}s - {seg['end']}s)")
                    st.write(f"🗣️ *{seg['speaker']}:* {seg['text']}")
                    if case["audio_path"] and os.path.exists(case["audio_path"]):
                        st.audio(case["audio_path"], start_time=int(seg["start"]))
                    else:
                        st.info("Audio file not found or mock audio used.")
            else:
                st.warning("No direct audio match found for this specific block.")

    # --- TAB 3: Full Transcript ---
    with tabs[2]:
        st.header("Courtroom Transcript")
        for seg in case["transcript"]["transcript"]:
            with st.chat_message("human" if "PLAINTIFF" in seg["speaker"].upper() else "assistant"):
                st.write(f"**{seg['speaker']}** [{seg['start']}s - {seg['end']}s]")
                st.write(seg["text"])

    # --- TAB 4: Case Chat (RAG) ---
    with tabs[3]:
        st.header("💬 Multi-Modal Case Chat")
        st.info("Ask questions about the case. The LLM will answer based on evidence from transcripts and documents.")
        
        # Display chat history
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat Input
        if prompt := st.chat_input("Ask about the legal arguments..."):
            # Add user message to history
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Generate RAG Response
            with st.chat_message("assistant"):
                with st.spinner("⚖️ Searching evidence and thinking..."):
                    # Ensure knowledge base is built for search
                    intel_engine.build_knowledge_base(case["transcript"], case["docs"])
                    response = intel_engine.chat(prompt, top_k=retrieval_depth)
                    st.markdown(response)
            
            # Add assistant message to history
            st.session_state.chat_history.append({"role": "assistant", "content": response})

    # --- TAB 5: Semantic Search ---
    with tabs[4]:
        st.header("🔍 Cross-Modal Semantic Search")
        query = st.text_input("Enter your query (e.g., 'What was said about the financial evidence?')")
        if query:
            # We need to rebuild knowledge base if engines were reloaded or just use the local one
            # For simplicity, we'll re-run search if needed
            intel_engine.build_knowledge_base(case["transcript"], case["docs"])
            results = intel_engine.search(query, top_k=5)
            
            for sim, res in results:
                with st.expander(f"🎯 Match Score: {sim:.2f} | Type: {res['type'].capitalize()}"):
                    st.write(res["text"])
                    if res["type"] == "speech":
                        st.write(f"🕒 Timestamp: {res['metadata']['start']}s")
                    else:
                        st.write(f"📄 Document: {res['file_name']}")

else:
    st.info("🏛️ Welcome to S.P.E.C.T.R.A. Please process a case using the sidebar to begin analysis.")
    
    # Show a professional overview if no case is loaded
    st.subheader("How S.P.E.C.T.R.A Works")
    st.markdown("""
    🏛️ **S.P.E.C.T.R.A** (Speech Processing & Electronic Document Convergence for Legal Records & Analysis) is a next-generation AI system designed for legal professionals.

    ### 🛠️ The Pipeline
    1.  **📄 Document Processing**: We use **YOLOv10** for layout detection and **Surya OCR** for high-accuracy text extraction from legal filings and evidence.
    2.  **🎙️ Speech Intelligence**: Courtroom audio is processed using **OpenAI Whisper** for transcription and **Pyannote** for speaker diarization (identifying judges, counsel, and witnesses).
    3.  **🔄 Multi-Modal Convergence**: Our proprietary engine synchronizes audio timestamps with specific document blocks, allowing for interactive "click-to-hear" navigation.
    4.  **🧠 Semantic Search & RAG**: A cross-modal knowledge base enables you to search across both speech and text using natural language queries.

    *Upload your case files in the sidebar to generate a unified intelligence report.*
    """)

# 🏛️ S.P.E.C.T.R.A: Multi-Modal Legal Intelligence

**Speech Processing & Electronic Document Convergence for Legal Records & Analysis**

SPECTRA is a next-generation AI system designed to bridge the gap between spoken courtroom arguments and written legal documents. It synchronizes audio transcripts with document analysis to provide a unified, searchable, and interactive intelligence report.

---

## 🚀 Key Features

*   **📄 Document Intelligence**: High-accuracy layout detection (YOLOv10) and OCR (Surya) for complex legal filings.
*   **🎙️ Speech Intelligence**: Professional transcription (OpenAI Whisper) and speaker diarization (Pyannote) for courtroom recordings.
*   **🔄 Multi-Modal Convergence**: Interactive "Click-to-Hear" functionality that maps document paragraphs to specific timestamps in audio.
*   **🧠 Semantic Search & RAG**: Cross-modal knowledge retrieval allowing natural language queries across both text and speech.
*   **📊 Unified Reports**: Automated executive summaries and evidence timelines.

---

## 🛠️ System Architecture

SPECTRA operates through four specialized engines:

1.  **Document Engine**: Extracts structure and text from PDFs/Images.
2.  **Speech Engine**: Transcribes and identifies speakers in audio recordings.
3.  **Convergence Engine**: Performs fuzzy temporal-spatial mapping between modalities.
4.  **Intelligence Engine**: Builds a semantic vector store for RAG and summarization.

---

## 📦 Installation

### 1. Environment Setup
We recommend using **Conda** to manage dependencies:

```bash
conda create -n spectra python=3.12
conda activate spectra
pip install -r requirements.txt
```

### 2. System Dependencies
Ensure `ffmpeg` is installed for audio processing:
```bash
# Ubuntu/Debian
sudo apt-get update && sudo apt-get install -y ffmpeg
```

---

## 🖥️ Usage

### Web UI (Recommended)
Start the interactive Streamlit application to upload and process cases visually:

```bash
streamlit run app.py
```

### CLI Pipeline
Process a case directory via the command line:

```bash
python cli.py --case-dir ./my_case --output-dir ./results
```

### Document Analysis Only
Run layout analysis and OCR on a folder of images:

```bash
python dla_cli.py --source ./images --output ./results
```

---

## 🔑 Configuration

### Hugging Face Token
To enable **Speaker Diarization** (Pyannote), you must provide an HF Token:
1.  Accept terms for [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) and [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0).
2.  Generate a token at [hf.co/settings/tokens](https://huggingface.co/settings/tokens).
3.  Enter the token in the SPECTRA Web UI sidebar or set it as an environment variable:
    ```bash
    export HF_TOKEN="your_token_here"
    ```

---

## 🧪 Testing & Evaluation

Run the verification suite to ensure all engines are correctly integrated:

```bash
# Test environment and imports
python tests/test_env.py

# Test full pipeline logic (with mock data)
python tests/test_pipeline.py

# Run benchmark evaluation
python evaluate.py
```

---

## 📂 Project Structure

```
.
├── app.py                 # Streamlit Web UI Entry
├── cli.py                 # Full Pipeline CLI
├── dla_cli.py             # Document Layout CLI
├── evaluate.py            # Model Evaluation Script
├── visualize.py           # Rendering Utilities
├── src/                   # Core Logic
│   ├── document_engine.py # Layout & OCR
│   ├── speech_engine.py   # Transcription & Diarization
│   ├── convergence_engine.py # Sync Logic
│   ├── intelligence_engine.py # RAG & Summary
│   └── utils.py           # Common Helpers
├── tests/                 # Testing Suite
└── requirements.txt       # Dependencies
```

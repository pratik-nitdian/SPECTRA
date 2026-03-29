import os
import json
import argparse
from tqdm import tqdm

from src.speech_engine import SpeechProcessor
from src.document_engine import DocumentEngine
from src.convergence_engine import ConvergenceEngine
from src.intelligence_engine import IntelligenceEngine

class SpectraPipeline:
    def __init__(self, model_path, hf_token=None):
        print("\n" + "="*50)
        print("🏛️ INITIALIZING S.P.E.C.T.R.A MULTI-MODAL PIPELINE")
        print("="*50)
        
        # Initialize Engines
        self.doc_engine = DocumentEngine(model_path)
        self.speech_engine = SpeechProcessor(hf_token=hf_token)
        self.conv_engine = ConvergenceEngine()
        self.intel_engine = IntelligenceEngine()

    def process_case(self, case_dir, output_dir):
        """
        Processes all files in a case directory (PDFs, Images, Audio).
        """
        os.makedirs(output_dir, exist_ok=True)
        json_dir = os.path.join(output_dir, "document_analysis")
        os.makedirs(json_dir, exist_ok=True)

        # 1. Gather files
        audio_files = [f for f in os.listdir(case_dir) if f.lower().endswith((".wav", ".mp3", ".m4a"))]
        image_files = [f for f in os.listdir(case_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
        
        # 2. Process Documents (OCR & Layout)
        print(f"\n📄 Step 1: Processing {len(image_files)} document images...")
        doc_data_list = []
        for img_file in tqdm(image_files, desc="Documents"):
            img_path = os.path.join(case_dir, img_file)
            # For simplicity, we process one by one here; in production, use batching
            # We mock the npy_path for now as inference.py expects npy if called via run_batch
            # Actually, let's use a simpler way if possible, or just call run_batch with one item
            import cv2
            import numpy as np
            img = cv2.imread(img_path)
            tmp_npy = os.path.join(output_dir, "tmp.npy")
            np.save(tmp_npy, img)
            
            self.doc_engine.run_batch([(tmp_npy, img_file)], json_dir, output_dir, save_vis=True)
            
            # Load the generated JSON
            json_path = os.path.join(json_dir, os.path.splitext(img_file)[0] + ".json")
            with open(json_path, "r") as f:
                doc_data_list.append(json.load(f))

        # 3. Process Speech
        print(f"\n🎙️ Step 2: Processing {len(audio_files)} audio recordings...")
        all_transcripts = []
        for audio_file in tqdm(audio_files, desc="Speech"):
            audio_path = os.path.join(case_dir, audio_file)
            transcript = self.speech_engine.process_audio(audio_path)
            all_transcripts.append(transcript)

        # 4. Convergence & Intelligence
        if all_transcripts and doc_data_list:
            print("\n🔄 Step 3: Synchronizing speech and documents...")
            # Sync the first transcript with all documents for this demo
            synced_report = self.conv_engine.synchronize(all_transcripts[0], doc_data_list)
            
            print("\n🧠 Step 4: Building Intelligence Knowledge Base...")
            self.intel_engine.build_knowledge_base(all_transcripts[0], doc_data_list)
            summary = self.intel_engine.summarize_case(all_transcripts[0], doc_data_list)
            
            # 5. Save Final Unified Report
            final_report = {
                "case_summary": summary,
                "synced_timeline": synced_report,
                "metadata": {
                    "total_pages": len(doc_data_list),
                    "total_audio_recordings": len(all_transcripts)
                }
            }
            
            report_path = os.path.join(output_dir, "SPECTRA_Final_Report.json")
            with open(report_path, "w") as f:
                json.dump(final_report, f, indent=4)
            
            print(f"\n✅ Pipeline Complete! Final report saved to: {report_path}")
        else:
            print("\n⚠️ Insufficient data (missing audio or documents) for full convergence.")

def get_args():
    parser = argparse.ArgumentParser(description="SPECTRA Multi-Modal Pipeline")
    parser.add_argument("--case-dir", type=str, required=True, help="Directory containing audio and images")
    parser.add_argument("--output-dir", type=str, default="spectra_output", help="Output directory")
    parser.add_argument("--model", type=str, default="doclayout_yolo_doclaynet_imgsz1120_docsynth_pretrain.pt", help="Path to layout model")
    parser.add_argument("--hf-token", type=str, help="Hugging Face token for Pyannote")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    pipeline = SpectraPipeline(args.model, hf_token=args.hf_token)
    pipeline.process_case(args.case_dir, args.output_dir)

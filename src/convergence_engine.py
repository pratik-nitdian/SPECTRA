import json
import os
import re
from difflib import SequenceMatcher

class ConvergenceEngine:
    """
    S.P.E.C.T.R.A Convergence Engine
    Synchronizes audio transcripts with document analysis outputs.
    """
    def __init__(self, match_threshold=0.6):
        self.match_threshold = match_threshold

    def synchronize(self, transcript_data, document_data_list):
        """
        Syncs a transcript with a list of document analysis outputs (one per page).
        Returns a mapping of transcript segments to document blocks.
        """
        synced_results = []
        
        # Flatten all document blocks for searching
        all_blocks = []
        for doc in document_data_list:
            file_name = doc.get("file_name", "unknown")
            for ann in doc.get("annotations", []):
                if ann.get("text"):
                    all_blocks.append({
                        "id": ann["id"],
                        "text": self._preprocess_text(ann["text"]),
                        "original_text": ann["text"],
                        "file_name": file_name,
                        "bbox": ann["bbox"]
                    })

        for segment in transcript_data.get("transcript", []):
            seg_text = self._preprocess_text(segment["text"])
            if not seg_text: continue

            best_match = None
            max_ratio = 0

            # Find the best matching document block for this transcript segment
            for block in all_blocks:
                ratio = self._get_match_ratio(seg_text, block["text"])
                if ratio > max_ratio:
                    max_ratio = ratio
                    best_match = block

            sync_entry = {
                "transcript_segment": segment,
                "document_match": None
            }

            if max_ratio >= self.match_threshold:
                sync_entry["document_match"] = {
                    "block_id": best_match["id"],
                    "file_name": best_match["file_name"],
                    "bbox": best_match["bbox"],
                    "match_confidence": round(max_ratio, 4)
                }
            
            synced_results.append(sync_entry)

        return synced_results

    def _preprocess_text(self, text):
        # Remove non-alphanumeric characters and lowercase
        return re.sub(r'\W+', ' ', text).lower().strip()

    def _get_match_ratio(self, text1, text2):
        # Using SequenceMatcher for fuzzy matching
        # In a production system, we'd use more sophisticated NLP (embeddings)
        return SequenceMatcher(None, text1, text2).ratio()

if __name__ == "__main__":
    # Test with dummy data
    ce = ConvergenceEngine()
    
    dummy_transcript = {
        "transcript": [
            {"speaker": "SPEAKER_01", "start": 0.0, "end": 5.0, "text": "This is a legal document about SPECTRA."}
        ]
    }
    
    dummy_doc = {
        "file_name": "page1.jpg",
        "annotations": [
            {"id": "block_0_1", "text": "Legal Document: SPECTRA Analysis", "bbox": [10, 10, 100, 50]}
        ]
    }
    
    res = ce.synchronize(dummy_transcript, [dummy_doc])
    print(json.dumps(res, indent=4))

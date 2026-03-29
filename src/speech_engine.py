import os
import json
import torch
import numpy as np

# Try Faster Whisper first for performance
try:
    from faster_whisper import WhisperModel
    HAS_FASTER_WHISPER = True
except ImportError:
    HAS_FASTER_WHISPER = False
    try:
        import whisper
        HAS_WHISPER = True
    except ImportError:
        HAS_WHISPER = False

try:
    from pyannote.audio import Pipeline
    import torchaudio
    HAS_PYANNOTE = True
except ImportError:
    HAS_PYANNOTE = False

class SpeechProcessor:
    """
    S.P.E.C.T.R.A Speech Processing Engine
    Handles ASR (Faster-Whisper) and Speaker Diarization (Pyannote).
    """
    def __init__(self, whisper_model="base", hf_token=None, device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.whisper_model_name = whisper_model
        self.hf_token = hf_token
        self.asr_model = None
        self.diarization_pipeline = None

        # 1. Load ASR Model
        if HAS_FASTER_WHISPER:
            print(f"🎙️ Loading Faster-Whisper ASR ({whisper_model}) on {self.device}...")
            # compute_type "float16" is great for RTX 4070
            self.asr_model = WhisperModel(whisper_model, device=self.device, compute_type="float16" if self.device == "cuda" else "int8")
        elif HAS_WHISPER:
            print(f"🎙️ Loading Vanilla Whisper ASR ({whisper_model}) on {self.device}...")
            self.asr_model = whisper.load_model(whisper_model, device=self.device)
        
        # 2. Load Diarization Pipeline
        if HAS_PYANNOTE and hf_token:
            print(f"🎙️ Loading Pyannote Diarization...")
            try:
                self.diarization_pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token=hf_token
                )
                if self.diarization_pipeline:
                    self.diarization_pipeline.to(torch.device(self.device))
            except Exception as e:
                print(f"⚠️ Failed to load Pyannote: {e}")

    def process_audio(self, audio_path, output_json=None, progress_callback=None):
        """
        Runs ASR and Diarization on an audio file.
        Returns a structured transcript with speaker labels.
        """
        # 1. Run ASR
        segments_data = []
        if self.asr_model and os.path.exists(audio_path):
            print(f"✨ Transcribing {audio_path}...")
            if HAS_FASTER_WHISPER:
                segments, info = self.asr_model.transcribe(audio_path, beam_size=5)
                duration = info.duration
                for s in segments:
                    segments_data.append({
                        "start": s.start,
                        "end": s.end,
                        "text": s.text.strip(),
                        "avg_logprob": s.avg_logprob
                    })
                    if progress_callback and duration > 0:
                        # Report progress based on audio timestamp vs total duration
                        progress_callback(min(s.end / duration, 1.0))
            else:
                # Vanilla Whisper doesn't easily yield progress during a single call, 
                # but we'll mock a "start" and "end" for the callback
                if progress_callback: progress_callback(0.1)
                asr_result = self.asr_model.transcribe(audio_path, verbose=False)
                segments_data = asr_result.get("segments", [])
                if progress_callback: progress_callback(1.0)
        else:
            reason = "ASR model not loaded" if not self.asr_model else f"file {audio_path} missing"
            print(f"⚠️ {reason}. Using mock transcripts.")
            segments_data = self._get_mock_segments()

        # 2. Run Diarization (if available)
        diarization = None
        if self.diarization_pipeline and os.path.exists(audio_path):
            print(f"✨ Diarizing {audio_path}...")
            try:
                # Pre-load audio to ensure consistency and fix chunking errors
                waveform, sample_rate = torchaudio.load(audio_path)
                
                # Convert to mono if necessary
                if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)
                
                # Ensure 16kHz
                if sample_rate != 16000:
                    resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                    waveform = resampler(waveform)
                    sample_rate = 16000
                
                # Pass waveform dict to fix the 'expected size' RuntimeError
                audio_input = {"waveform": waveform, "sample_rate": sample_rate}
                diarization = self.diarization_pipeline(audio_input)
            except Exception as e:
                print(f"⚠️ Diarization failed: {e}. Proceeding with transcription only.")

        # 3. Merge ASR and Diarization
        transcript = []
        for seg in segments_data:
            speaker = "UNKNOWN"
            if diarization:
                speaker = self._get_speaker_for_segment(seg["start"], seg["end"], diarization)
            
            transcript.append({
                "speaker": speaker,
                "start": round(seg["start"], 2),
                "end": round(seg["end"], 2),
                "text": seg["text"].strip(),
                "confidence": round(seg.get("avg_logprob", 0), 4)
            })

        output = {
            "audio_file": os.path.basename(audio_path),
            "transcript": transcript
        }

        if output_json:
            with open(output_json, "w") as f:
                json.dump(output, f, indent=4)
        
        return output

    def _get_speaker_for_segment(self, start, end, diarization):
        from pyannote.core import Segment
        from collections import Counter
        
        try:
            seg = Segment(start, end)
            # Find speakers in this segment
            speakers = diarization.crop(seg).labels()
            if not speakers:
                return "UNKNOWN"
            # Return most frequent speaker in segment
            return Counter(speakers).most_common(1)[0][0]
        except:
            return "UNKNOWN"

    def _get_mock_segments(self):
        """Mock transcripts for testing without ASR."""
        return [
            {"start": 0.0, "end": 5.0, "text": "Good morning, Your Honor. Counsel for the plaintiff, SPECTRA Legal AI.", "avg_logprob": -0.1},
            {"start": 5.5, "end": 12.0, "text": "We are here today to discuss the convergence of speech and document intelligence.", "avg_logprob": -0.05},
            {"start": 12.5, "end": 20.0, "text": "As stated in the proposal, our system bridges the gap between spoken and written evidence.", "avg_logprob": -0.12}
        ]

if __name__ == "__main__":
    # Test with mock data
    sp = SpeechProcessor()
    res = sp.process_audio("dummy.wav")
    print(json.dumps(res, indent=4))

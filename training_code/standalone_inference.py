#!/usr/bin/env python3
"""
Standalone VAD Inference
Use trained model without any pyannote.audio dependency
"""

import sys
from pathlib import Path
import torch
import torchaudio
import numpy as np
from typing import List, Tuple

# Import model from training script
from standalone_train_vad import VADModel


class StandaloneVADInference:
    """
    VAD inference without pyannote.audio
    Implements hysteresis thresholding and segment filtering
    """
    
    def __init__(
        self,
        model_path: str,
        onset_threshold: float = 0.5,
        offset_threshold: float = 0.3,
        min_duration_on: float = 0.5,
        min_duration_off: float = 0.3,
        sample_rate: int = 16000
    ):
        """
        Initialize VAD inference
        
        Parameters
        ----------
        model_path : str
            Path to trained model (.pt file)
        onset_threshold : float
            Threshold for speech start (0.0-1.0)
        offset_threshold : float
            Threshold for speech end (0.0-1.0)
        min_duration_on : float
            Minimum speech segment duration (seconds)
        min_duration_off : float
            Minimum silence duration between segments (seconds)
        sample_rate : int
            Audio sample rate
        """
        self.onset_threshold = onset_threshold
        self.offset_threshold = offset_threshold
        self.min_duration_on = min_duration_on
        self.min_duration_off = min_duration_off
        self.sample_rate = sample_rate
        self.frame_shift = 0.01  # 10ms
        
        # Load model
        print(f"📥 Loading model from: {model_path}")
        # Use weights_only=False for PyTorch Lightning checkpoints (trusted source)
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        # Create model
        self.model = VADModel(**checkpoint['hyperparameters'])
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Move to GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        print(f"✅ Model loaded on {self.device}")
        print(f"   Onset: {onset_threshold}, Offset: {offset_threshold}")
    
    def process_audio(self, audio_path: str) -> List[Tuple[float, float]]:
        """
        Process audio file and detect speech segments
        
        Parameters
        ----------
        audio_path : str
            Path to audio file
        
        Returns
        -------
        segments : list of tuples
            List of (start_time, end_time) in seconds
        """
        # Load audio using soundfile to avoid torchcodec/FFmpeg DLL issues
        try:
            import soundfile as sf
            # Read audio with soundfile (avoids torchcodec)
            data, sr = sf.read(audio_path, dtype='float32', always_2d=True)
            # Convert from (samples, channels) to (channels, samples)
            data = data.T
            waveform = torch.from_numpy(data).float()
        except Exception as e:
            print(f"⚠️  soundfile failed, trying torchaudio fallback: {e}")
            # Fallback to torchaudio (may fail with torchcodec error)
            waveform, sr = torchaudio.load(audio_path)
        
        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Resample if needed
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        
        # Process in chunks (5 seconds with 2.5s overlap)
        chunk_duration = 5.0
        chunk_samples = int(chunk_duration * self.sample_rate)
        hop_samples = chunk_samples // 2
        
        total_samples = waveform.shape[1]
        all_probs = []
        
        with torch.no_grad():
            for start in range(0, total_samples, hop_samples):
                end = min(start + chunk_samples, total_samples)
                chunk = waveform[:, start:end]
                
                # Pad if needed
                if chunk.shape[1] < chunk_samples:
                    padding = chunk_samples - chunk.shape[1]
                    chunk = torch.nn.functional.pad(chunk, (0, padding))
                
                # Forward pass - model expects (batch, samples) so squeeze channel dim
                chunk = chunk.squeeze(0)  # (channels, samples) -> (samples,)
                chunk = chunk.to(self.device)
                logits = self.model(chunk.unsqueeze(0))  # (1, samples) -> (1, frames, 1)
                probs = torch.sigmoid(logits).squeeze().cpu().numpy()
                
                all_probs.append(probs)
        
        # Merge overlapping predictions (average)
        num_frames = int(total_samples / (self.frame_shift * self.sample_rate))
        merged_probs = np.zeros(num_frames)
        counts = np.zeros(num_frames)
        
        for i, probs in enumerate(all_probs):
            start_frame = int(i * hop_samples / (self.frame_shift * self.sample_rate))
            end_frame = min(start_frame + len(probs), num_frames)
            
            merged_probs[start_frame:end_frame] += probs[:end_frame - start_frame]
            counts[start_frame:end_frame] += 1
        
        merged_probs /= np.maximum(counts, 1)
        
        # Apply hysteresis thresholding
        segments = self._apply_hysteresis(merged_probs)
        
        # Filter short segments
        segments = self._filter_segments(segments)
        
        return segments
    
    def _apply_hysteresis(self, probs: np.ndarray) -> List[Tuple[float, float]]:
        """
        Apply hysteresis thresholding
        Uses different thresholds for speech start (onset) and end (offset)
        """
        segments = []
        is_speech = False
        start_frame = 0
        
        for i, prob in enumerate(probs):
            if not is_speech and prob > self.onset_threshold:
                # Speech starts
                is_speech = True
                start_frame = i
            elif is_speech and prob < self.offset_threshold:
                # Speech ends
                is_speech = False
                start_time = start_frame * self.frame_shift
                end_time = i * self.frame_shift
                segments.append((start_time, end_time))
        
        # Handle case where speech continues to end
        if is_speech:
            start_time = start_frame * self.frame_shift
            end_time = len(probs) * self.frame_shift
            segments.append((start_time, end_time))
        
        return segments
    
    def _filter_segments(self, segments: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """
        Filter segments by duration and merge close segments
        """
        if not segments:
            return segments
        
        # Filter short segments
        filtered = []
        for start, end in segments:
            duration = end - start
            if duration >= self.min_duration_on:
                filtered.append((start, end))
        
        if not filtered:
            return filtered
        
        # Merge segments with short gaps
        merged = [filtered[0]]
        for start, end in filtered[1:]:
            last_start, last_end = merged[-1]
            gap = start - last_end
            
            if gap < self.min_duration_off:
                # Merge segments
                merged[-1] = (last_start, end)
            else:
                merged.append((start, end))
        
        return merged


def main():
    """Test VAD inference on audio file"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Standalone VAD Inference")
    parser.add_argument(
        "--audio",
        type=str,
        required=True,
        help="Path to audio file"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="experiments_standalone/vad_standalone_v1/final_model/model.pt",
        help="Path to trained model"
    )
    parser.add_argument(
        "--onset",
        type=float,
        default=0.5,
        help="Onset threshold (0.0-1.0)"
    )
    parser.add_argument(
        "--offset",
        type=float,
        default=0.3,
        help="Offset threshold (0.0-1.0)"
    )
    
    args = parser.parse_args()
    
    # Check files exist
    if not Path(args.model).exists():
        print(f"❌ Error: Model not found at {args.model}")
        print("\n💡 Train a model first: python standalone_train_vad.py")
        sys.exit(1)
    
    if not Path(args.audio).exists():
        print(f"❌ Error: Audio file not found: {args.audio}")
        sys.exit(1)
    
    # Run VAD
    print("\n" + "="*80)
    print("STANDALONE VAD INFERENCE")
    print("="*80 + "\n")
    
    vad = StandaloneVADInference(
        model_path=args.model,
        onset_threshold=args.onset,
        offset_threshold=args.offset,
        min_duration_on=0.5,
        min_duration_off=0.3
    )
    
    print(f"\n🎵 Processing: {args.audio}")
    segments = vad.process_audio(args.audio)
    
    print(f"\n✅ Detected {len(segments)} speech segments:\n")
    
    total_speech = 0
    for i, (start, end) in enumerate(segments, 1):
        duration = end - start
        total_speech += duration
        print(f"   {i:3d}. {start:8.2f}s → {end:8.2f}s  (duration: {duration:6.2f}s)")
    
    print(f"\n📊 Summary:")
    print(f"   • Total segments: {len(segments)}")
    print(f"   • Total speech time: {total_speech:.2f}s")
    
    if len(segments) > 0:
        avg_duration = total_speech / len(segments)
        print(f"   • Average segment: {avg_duration:.2f}s")


if __name__ == "__main__":
    main()

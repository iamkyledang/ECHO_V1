#!/usr/bin/env python3
"""
Standalone VAD Training Pipeline
Inspired by pyannote.audio architecture but using only common libraries:
- PyTorch for deep learning
- PyTorch Lightning for training
- torchaudio for audio processing
- No pyannote.audio dependency!

Architecture: SincNet -> LSTM -> Linear (PyanNet-inspired)
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger


# ============================================================================
# 1. DATA LOADING AND PREPROCESSING
# ============================================================================

class RTTMParser:
    """Parse RTTM annotation files"""
    
    @staticmethod
    def parse_rttm_file(rttm_path: str) -> Dict[str, List[Tuple[float, float]]]:
        """
        Parse RTTM file and return speech segments per file
        
        Returns:
            dict: {file_id: [(start_time, end_time), ...]}
        """
        segments = {}
        
        with open(rttm_path, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                
                file_id = parts[1]
                start_time = float(parts[3])
                duration = float(parts[4])
                end_time = start_time + duration
                
                if file_id not in segments:
                    segments[file_id] = []
                
                segments[file_id].append((start_time, end_time))
        
        return segments


class VADDataset(Dataset):
    """
    Voice Activity Detection Dataset
    Loads audio chunks and creates frame-level binary labels
    """
    
    def __init__(
        self,
        audio_dir: str,
        file_list: List[str],
        annotations: Dict[str, List[Tuple[float, float]]],
        chunk_duration: float = 5.0,
        sample_rate: int = 16000,
        frame_shift: float = 0.01,  # 10ms
        augment: bool = False
    ):
        self.audio_dir = Path(audio_dir)
        self.file_list = file_list
        self.annotations = annotations
        self.chunk_duration = chunk_duration
        self.sample_rate = sample_rate
        self.frame_shift = frame_shift
        self.augment = augment
        
        self.chunk_samples = int(chunk_duration * sample_rate)
        self.frame_samples = int(frame_shift * sample_rate)
        
        # Generate all possible chunks
        self.chunks = self._generate_chunks()
        
        print(f"   Dataset created: {len(self.file_list)} files, {len(self.chunks)} chunks")
    
    def _generate_chunks(self) -> List[Tuple[str, float]]:
        """Generate all possible training chunks"""
        chunks = []
        missing_files = []
        
        for file_id in self.file_list:
            audio_path = self.audio_dir / f"{file_id}.wav"
            
            if not audio_path.exists():
                missing_files.append(file_id)
                continue
            
            # Get audio duration (compatible with different torchaudio versions)
            try:
                # Try new API (torchaudio >= 0.10)
                import soundfile as sf
                info = sf.info(str(audio_path))
                duration = info.duration
            except:
                # Fallback: load audio to get duration
                try:
                    waveform, sr = torchaudio.load(str(audio_path))
                    duration = waveform.shape[1] / sr
                except Exception as e:
                    print(f"   ⚠️  Warning: Could not load {file_id}: {e}")
                    continue
            
            # Generate chunks (with 50% overlap)
            start = 0.0
            while start + self.chunk_duration <= duration:
                chunks.append((file_id, start))
                start += self.chunk_duration / 2  # 50% overlap
        
        # Report missing files
        if missing_files:
            print(f"   ⚠️  Warning: {len(missing_files)} audio files not found in {self.audio_dir}")
            print(f"   Missing: {missing_files[:5]}{'...' if len(missing_files) > 5 else ''}")
        
        return chunks
    
    def __len__(self) -> int:
        return len(self.chunks)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        file_id, start_time = self.chunks[idx]
        
        # Load audio chunk (use soundfile to avoid torchcodec/FFmpeg issues)
        audio_path = self.audio_dir / f"{file_id}.wav"
        file_sr = None
        try:
            import soundfile as sf
            # Use file's samplerate to compute start/frame offsets
            info = sf.info(str(audio_path))
            file_sr = info.samplerate
            start_frame = int(start_time * file_sr)
            frames = int(self.chunk_duration * file_sr)
            data = sf.read(str(audio_path), start=start_frame, frames=frames, dtype='float32', always_2d=True)[0]
            # data: (frames, channels) -> convert to (channels, frames)
            data = data.T
            waveform = torch.from_numpy(data).float()
        except Exception:
            # Fallback to torchaudio.load (may trigger torchcodec if backend set to torchcodec)
            # Use previous approach as a fallback to keep compatibility
            waveform, sr = torchaudio.load(
                str(audio_path),
                frame_offset=int(start_time * self.sample_rate),
                num_frames=self.chunk_samples
            )
            file_sr = sr
        
        # Convert to mono
        if waveform.ndim == 2 and waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Resample if needed
        if file_sr is not None and file_sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(file_sr, self.sample_rate)
            waveform = resampler(waveform)
        
        # Pad if too short
        if waveform.shape[1] < self.chunk_samples:
            padding = self.chunk_samples - waveform.shape[1]
            waveform = F.pad(waveform, (0, padding))
        
        # Apply augmentation
        if self.augment:
            waveform = self._augment(waveform)
        
        # Create frame-level labels
        num_frames = int(self.chunk_duration / self.frame_shift)
        labels = torch.zeros(num_frames, dtype=torch.float32)
        
        # Get speech segments for this file
        if file_id in self.annotations:
            for seg_start, seg_end in self.annotations[file_id]:
                # Convert to chunk-relative time
                rel_start = seg_start - start_time
                rel_end = seg_end - start_time
                
                # Convert to frame indices
                frame_start = int(rel_start / self.frame_shift)
                frame_end = int(rel_end / self.frame_shift)
                
                # Clip to valid range
                frame_start = max(0, frame_start)
                frame_end = min(num_frames, frame_end)
                
                # Mark as speech
                if frame_start < frame_end:
                    labels[frame_start:frame_end] = 1.0
        
        return waveform.squeeze(0), labels
    
    def _augment(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply simple audio augmentation"""
        
        # Random gain (±6 dB)
        if np.random.random() < 0.5:
            gain_db = np.random.uniform(-6, 6)
            gain = 10 ** (gain_db / 20)
            waveform = waveform * gain
        
        # Add white noise (SNR 20-40 dB)
        if np.random.random() < 0.5:
            snr_db = np.random.uniform(20, 40)
            signal_power = waveform.pow(2).mean()
            noise_power = signal_power / (10 ** (snr_db / 10))
            noise = torch.randn_like(waveform) * torch.sqrt(noise_power)
            waveform = waveform + noise
        
        return waveform


# ============================================================================
# 2. MODEL ARCHITECTURE (Inspired by PyanNet)
# ============================================================================

class SincConv1d(nn.Module):
    """
    SincNet convolutional layer
    Learns bandpass filters directly from raw waveform
    Based on: "Speaker Recognition from Raw Waveform with SincNet" (Ravanelli & Bengio, 2018)
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 60,
        kernel_size: int = 251,
        sample_rate: int = 16000,
        stride: int = 1,
        padding: int = 0,
        min_low_hz: float = 50,
        min_band_hz: float = 50
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate
        self.stride = stride
        self.padding = padding
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz
        
        # Initialize frequency parameters
        low_hz = 30
        high_hz = sample_rate / 2 - (min_low_hz + min_band_hz)
        
        # Learnable parameters for filter frequencies
        mel_scale = torch.linspace(
            self._hz_to_mel(low_hz),
            self._hz_to_mel(high_hz),
            out_channels + 1
        )
        hz_scale = self._mel_to_hz(mel_scale)
        
        self.low_hz_ = nn.Parameter(hz_scale[:-1])
        self.band_hz_ = nn.Parameter(torch.diff(hz_scale))
        
        # Hamming window
        n = torch.arange(0, kernel_size).float()
        self.window_ = 0.54 - 0.46 * torch.cos(2 * np.pi * n / kernel_size)
        
        # Time axis
        self.n_ = 2 * np.pi * torch.arange(-(kernel_size // 2), (kernel_size // 2) + 1).float() / sample_rate
    
    @staticmethod
    def _hz_to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)
    
    @staticmethod
    def _mel_to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure positive frequencies
        low = self.min_low_hz + torch.abs(self.low_hz_)
        high = torch.clamp(low + self.min_band_hz + torch.abs(self.band_hz_), self.min_low_hz, self.sample_rate / 2)
        
        # Create bandpass filters
        band = (high - low)[:, None]
        
        # Move tensors to same device
        n_ = self.n_.to(x.device)
        window_ = self.window_.to(x.device)
        
        f_times_t_low = torch.matmul(low[:, None], n_[None, :])
        f_times_t_high = torch.matmul(high[:, None], n_[None, :])
        
        # Bandpass filter
        band_pass_left = ((torch.sin(f_times_t_high) - torch.sin(f_times_t_low)) / (n_ / 2)) * window_
        band_pass_center = 2 * band
        band_pass_right = ((torch.sin(f_times_t_high) - torch.sin(f_times_t_low)) / (n_ / 2)) * window_
        
        filters = torch.cat([band_pass_left[:, :self.kernel_size // 2],
                            band_pass_center,
                            band_pass_right[:, self.kernel_size // 2 + 1:]], dim=1)
        
        # Normalize
        filters = filters / (2 * band[:, 0, None])
        
        # Apply convolution
        filters = filters.view(self.out_channels, 1, self.kernel_size)
        return F.conv1d(x.unsqueeze(1), filters, stride=self.stride, padding=self.padding)


class VADModel(pl.LightningModule):
    """
    Voice Activity Detection Model
    Architecture: SincNet -> LSTM -> Linear (inspired by PyanNet)
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        sincnet_out_channels: int = 60,
        sincnet_kernel_size: int = 251,
        sincnet_stride: int = 160,  # 10ms with 16kHz
        lstm_hidden_size: int = 128,
        lstm_num_layers: int = 2,
        lstm_dropout: float = 0.1,
        linear_hidden_size: int = 128,
        learning_rate: float = 1e-3
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.sample_rate = sample_rate
        self.learning_rate = learning_rate
        
        # SincNet frontend
        self.sincnet = SincConv1d(
            in_channels=1,
            out_channels=sincnet_out_channels,
            kernel_size=sincnet_kernel_size,
            sample_rate=sample_rate,
            stride=sincnet_stride,
            padding=sincnet_kernel_size // 2
        )
        
        # Batch normalization
        self.bn1 = nn.BatchNorm1d(sincnet_out_channels)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=sincnet_out_channels,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            dropout=lstm_dropout if lstm_num_layers > 1 else 0,
            bidirectional=True,
            batch_first=True
        )
        
        # Linear layers
        lstm_output_size = lstm_hidden_size * 2  # bidirectional
        self.linear1 = nn.Linear(lstm_output_size, linear_hidden_size)
        self.bn2 = nn.BatchNorm1d(linear_hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.linear2 = nn.Linear(linear_hidden_size, linear_hidden_size)
        self.bn3 = nn.BatchNorm1d(linear_hidden_size)
        
        # Output layer
        self.output = nn.Linear(linear_hidden_size, 1)
        
        # Loss function
        self.criterion = nn.BCEWithLogitsLoss()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        Args:
            x: (batch, samples) raw waveform
        Returns:
            (batch, frames, 1) frame-level predictions
        """
        # SincNet convolution
        x = self.sincnet(x)  # (batch, channels, frames)
        x = F.relu(self.bn1(x))
        x = F.max_pool1d(x, kernel_size=3, stride=1, padding=1)
        
        # Transpose for LSTM (batch, frames, channels)
        x = x.transpose(1, 2)
        
        # LSTM
        x, _ = self.lstm(x)  # (batch, frames, lstm_hidden*2)
        
        # Linear layers (process each frame)
        batch, frames, features = x.shape
        x = x.reshape(batch * frames, features)
        
        x = F.relu(self.bn2(self.linear1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.linear2(x)))
        
        # Output
        x = self.output(x)  # (batch*frames, 1)
        x = x.reshape(batch, frames, 1)
        
        return x
    
    def training_step(self, batch, batch_idx):
        waveforms, labels = batch
        
        # Forward pass
        logits = self(waveforms).squeeze(-1)  # (batch, frames)
        
        # Compute loss
        loss = self.criterion(logits, labels)
        
        # Compute accuracy
        preds = torch.sigmoid(logits) > 0.5
        acc = (preds == labels).float().mean()
        
        # Log metrics
        self.log('train/loss', loss, prog_bar=True)
        self.log('train/acc', acc, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        waveforms, labels = batch
        
        # Forward pass
        logits = self(waveforms).squeeze(-1)
        
        # Compute loss
        loss = self.criterion(logits, labels)
        
        # Compute accuracy
        preds = torch.sigmoid(logits) > 0.5
        acc = (preds == labels).float().mean()
        
        # Log metrics
        self.log('val/loss', loss, prog_bar=True)
        self.log('val/acc', acc, prog_bar=True)
        
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val/loss'
            }
        }


# ============================================================================
# 3. TRAINING PIPELINE
# ============================================================================

class StandaloneVADTrainer:
    """Complete VAD training pipeline"""
    
    def __init__(
        self,
        audio_base_dir: str = "D:/ECHO_V3/AI_subtitle_generator/voxconverse",
        data_dir: str = "D:/ECHO_V3/AI_subtitle_generator/nemo_vad_movie",
        output_dir: str = "experiments_standalone",
        experiment_name: str = "vad_standalone_v1"
    ):
        self.audio_base_dir = Path(audio_base_dir)
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.experiment_name = experiment_name
        
        self.output_dir.mkdir(exist_ok=True)
        
        print("="*80)
        print("STANDALONE VAD TRAINING PIPELINE")
        print("Using: PyTorch + PyTorch Lightning + torchaudio")
        print("="*80)
    
    def load_dataset(self) -> Tuple[VADDataset, VADDataset]:
        """Load training and validation datasets"""
        print("\n📚 Loading datasets...")
        
        # Load file lists
        train_lst = self.data_dir / "train.lst"
        dev_lst = self.data_dir / "dev.lst"
        
        with open(train_lst, 'r') as f:
            train_files = [line.strip() for line in f if line.strip()]
        
        with open(dev_lst, 'r') as f:
            dev_files = [line.strip() for line in f if line.strip()]
        
        print(f"   Train files: {len(train_files)}")
        print(f"   Val files:   {len(dev_files)}")
        
        # Parse RTTM annotations
        train_rttm = self.data_dir / "train.rttm"
        dev_rttm = self.data_dir / "dev.rttm"
        
        train_annotations = RTTMParser.parse_rttm_file(str(train_rttm))
        dev_annotations = RTTMParser.parse_rttm_file(str(dev_rttm))
        
        print(f"   Train annotations: {len(train_annotations)} files")
        print(f"   Val annotations:   {len(dev_annotations)} files")
        
        # Verify audio directory exists
        audio_dir = self.audio_base_dir / "dev"
        if not audio_dir.exists():
            print(f"\n❌ Error: Audio directory not found: {audio_dir}")
            print(f"\n💡 Expected structure:")
            print(f"   {self.audio_base_dir}/")
            print(f"   └── dev/")
            print(f"       ├── file1.wav")
            print(f"       ├── file2.wav")
            print(f"       └── ...")
            print(f"\n💡 Available directories in {self.audio_base_dir}:")
            if self.audio_base_dir.exists():
                subdirs = [d.name for d in self.audio_base_dir.iterdir() if d.is_dir()]
                if subdirs:
                    for subdir in subdirs:
                        print(f"   • {subdir}")
                else:
                    print(f"   (no directories found)")
            else:
                print(f"   ❌ Base directory does not exist!")
            sys.exit(1)
        
        # Count available audio files
        audio_files = list(audio_dir.glob("*.wav"))
        print(f"   Audio files in {audio_dir.name}/: {len(audio_files)} .wav files")
        
        if len(audio_files) == 0:
            print(f"\n❌ Error: No .wav files found in {audio_dir}")
            print(f"\n💡 Make sure audio files are in: {audio_dir}")
            sys.exit(1)
        
        # Create datasets
        print("\n📦 Creating training dataset...")
        train_dataset = VADDataset(
            audio_dir=str(self.audio_base_dir / "dev"),
            file_list=train_files,
            annotations=train_annotations,
            chunk_duration=5.0,
            augment=True
        )
        
        print("📦 Creating validation dataset...")
        val_dataset = VADDataset(
            audio_dir=str(self.audio_base_dir / "dev"),
            file_list=dev_files,
            annotations=dev_annotations,
            chunk_duration=5.0,
            augment=False
        )
        
        # Verify we have data
        if len(train_dataset) == 0:
            print(f"\n❌ Error: Training dataset is empty!")
            print(f"\n💡 Possible causes:")
            print(f"   1. Audio files don't match the file IDs in train.lst")
            print(f"   2. Audio files are too short (< {5.0}s)")
            print(f"   3. Audio files couldn't be loaded (wrong format/corrupted)")
            print(f"\n💡 Debug info:")
            print(f"   • Expected audio dir: {self.audio_base_dir / 'dev'}")
            print(f"   • Train files in train.lst: {len(train_files)}")
            print(f"   • First 5 file IDs: {train_files[:5]}")
            sys.exit(1)
        
        if len(val_dataset) == 0:
            print(f"\n❌ Error: Validation dataset is empty!")
            sys.exit(1)
        
        return train_dataset, val_dataset
    
    def create_dataloaders(
        self,
        train_dataset: VADDataset,
        val_dataset: VADDataset,
        batch_size: int = 32,
        num_workers: int = 4
    ) -> Tuple[DataLoader, DataLoader]:
        """Create data loaders"""
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True if num_workers > 0 else False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True if num_workers > 0 else False
        )
        
        print(f"\n✅ Data loaders created")
        print(f"   Batch size: {batch_size}")
        print(f"   Train batches: {len(train_loader)}")
        print(f"   Val batches: {len(val_loader)}")
        
        return train_loader, val_loader
    
    def train(
        self,
        max_epochs: int = 100,
        batch_size: int = 32,
        num_workers: int = 4,
        learning_rate: float = 1e-3
    ):
        """Execute training"""
        
        print("\n" + "="*80)
        print("STARTING TRAINING")
        print("="*80)
        
        # Load datasets
        train_dataset, val_dataset = self.load_dataset()
        train_loader, val_loader = self.create_dataloaders(
            train_dataset, val_dataset, batch_size, num_workers
        )
        
        # Create model
        print("\n🧠 Creating VAD model...")
        model = VADModel(
            sample_rate=16000,
            sincnet_out_channels=60,
            sincnet_stride=160,  # 10ms
            lstm_hidden_size=128,
            lstm_num_layers=2,
            linear_hidden_size=128,
            learning_rate=learning_rate
        )
        
        print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Setup callbacks
        print("\n💾 Setting up callbacks...")
        
        checkpoint_dir = self.output_dir / self.experiment_name / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_callback = ModelCheckpoint(
            dirpath=str(checkpoint_dir),
            filename="vad-{epoch:03d}-{val/loss:.4f}",
            monitor="val/loss",
            mode="min",
            save_top_k=3,
            save_last=True
        )
        
        early_stop_callback = EarlyStopping(
            monitor="val/loss",
            patience=20,
            mode="min",
            verbose=True
        )
        
        lr_monitor = LearningRateMonitor(logging_interval='epoch')
        
        # Setup logger
        logger = TensorBoardLogger(
            save_dir=str(self.output_dir),
            name=self.experiment_name
        )
        
        # Create trainer
        print("\n⚡ Creating PyTorch Lightning trainer...")
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            logger=logger,
            callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
            precision=16 if torch.cuda.is_available() else 32,
            log_every_n_steps=10,
            deterministic=True
        )
        
        device = "GPU" if torch.cuda.is_available() else "CPU"
        print(f"   Device: {device}")
        print(f"   Max epochs: {max_epochs}")
        print(f"   Precision: {16 if torch.cuda.is_available() else 32}-bit")
        
        # Train!
        print("\n🚀 Starting training loop...\n")
        trainer.fit(model, train_loader, val_loader)
        
        # Training complete
        print("\n" + "="*80)
        print("TRAINING COMPLETE!")
        print("="*80)
        
        best_model_path = checkpoint_callback.best_model_path
        print(f"\n✅ Best model: {best_model_path}")
        print(f"✅ Best val loss: {checkpoint_callback.best_model_score:.6f}")
        
        # Save final model
        final_model_dir = self.output_dir / self.experiment_name / "final_model"
        final_model_dir.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'model_state_dict': model.state_dict(),
            'hyperparameters': model.hparams
        }, final_model_dir / "model.pt")
        
        print(f"✅ Final model saved to: {final_model_dir}")
        print(f"\nTo view training logs:")
        print(f"  tensorboard --logdir {self.output_dir}/{self.experiment_name}")
        
        return best_model_path


# ============================================================================
# 4. MAIN ENTRY POINT
# ============================================================================

def main():
    """Main training script"""
    
    # Check prerequisites
    required_files = ["train.lst", "dev.lst", "train.rttm", "dev.rttm"]
    base_path = Path("D:/ECHO_V3/AI_subtitle_generator/nemo_vad_movie")
    
    missing = [f for f in required_files if not (base_path / f).exists()]
    if missing:
        print("❌ Error: Missing required files!")
        print("\nMissing:")
        for f in missing:
            print(f"  • {f}")
        print("\n💡 Run: python prepare_voxconverse.py")
        sys.exit(1)
    
    # Check GPU
    if torch.cuda.is_available():
        print(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA version: {torch.version.cuda}")
    else:
        print("⚠️  No GPU detected, training will use CPU (slower)")
    
    # Create trainer
    trainer = StandaloneVADTrainer()
    
    # Train
    trainer.train(
        max_epochs=100,
        batch_size=32,
        num_workers=4,
        learning_rate=1e-3
    )


if __name__ == "__main__":
    main()

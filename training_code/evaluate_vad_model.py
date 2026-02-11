#!/usr/bin/env python3
"""
Comprehensive VAD Model Evaluation
Evaluates trained model on test set with detailed metrics and visualizations
"""

import os
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict
import numpy as np
import torch
import torchaudio
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict
import json
import argparse

# Suppress scipy/torchmetrics warnings
import warnings
warnings.filterwarnings('ignore')

import sys
sys.modules['pytorch_lightning'] = None
sys.modules['torchmetrics'] = None

# Import ONLY inference (NO pytorch-lightning dependency!)
from inference_pure import StandaloneVADInference


class RTTMParser:
    """Parse RTTM annotation files"""
    
    @staticmethod
    def parse_rttm_file(rttm_path: str) -> Dict[str, List[Tuple[float, float]]]:
        """
        Parse RTTM file and extract speech segments per file
        
        Returns
        -------
        annotations : dict
            {file_id: [(start, end), ...]}
        """
        annotations = defaultdict(list)
        
        with open(rttm_path, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                
                parts = line.strip().split()
                if len(parts) < 4:
                    continue
                
                file_id = parts[1]
                start_time = float(parts[3])
                duration = float(parts[4])
                end_time = start_time + duration
                
                annotations[file_id].append((start_time, end_time))
        
        # Merge overlapping segments
        for file_id in annotations:
            annotations[file_id] = RTTMParser._merge_segments(annotations[file_id])
        
        return dict(annotations)
    
    @staticmethod
    def _merge_segments(segments: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Merge overlapping segments"""
        if not segments:
            return []
        
        sorted_segs = sorted(segments)
        merged = [sorted_segs[0]]
        
        for start, end in sorted_segs[1:]:
            last_start, last_end = merged[-1]
            if start <= last_end:
                merged[-1] = (last_start, max(last_end, end))
            else:
                merged.append((start, end))
        
        return merged


class VADEvaluator:
    """Comprehensive VAD evaluation with metrics and visualizations"""
    
    def __init__(
        self,
        model_path: str,
        test_audio_dir: str,
        test_rttm_path: str,
        test_lst_path: str,
        output_dir: str = "evaluation_results"
    ):
        self.model_path = model_path
        self.test_audio_dir = Path(test_audio_dir)
        self.test_rttm_path = test_rttm_path
        self.test_lst_path = test_lst_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Results storage
        self.results = []
        self.metrics = {}
        
        # Load test files
        with open(test_lst_path, 'r') as f:
            self.test_files = [line.strip() for line in f if line.strip()]
        
        # Load annotations
        print(f"📋 Loading annotations from {test_rttm_path}")
        self.annotations = RTTMParser.parse_rttm_file(test_rttm_path)
        
        # Initialize VAD model
        print(f"🔍 Loading VAD model from {model_path}")
        self.vad = StandaloneVADInference(
            model_path=model_path,
            onset_threshold=0.5,
            offset_threshold=0.3,
            min_duration_on=0.5,
            min_duration_off=0.3
        )
        
        print(f"✅ Evaluator initialized")
        print(f"   Test files: {len(self.test_files)}")
        print(f"   Annotations: {len(self.annotations)} files")
    
    def evaluate_all(self, max_files: int = None):
        """
        Evaluate model on test files
        
        Args:
            max_files: Maximum number of files to evaluate (None = all files)
        """
        # Limit test files if requested
        test_files = self.test_files[:max_files] if max_files else self.test_files
        
        print("\n" + "="*80)
        print("EVALUATING VAD MODEL ON TEST SET")
        if max_files:
            print(f"(QUICK MODE: Processing first {max_files} files only)")
        print("="*80 + "\n")
        
        for idx, file_id in enumerate(test_files, 1):
            print(f"[{idx}/{len(test_files)}] Processing {file_id}...", end=" ")
            
            audio_path = self.test_audio_dir / f"{file_id}.wav"
            
            if not audio_path.exists():
                print(f"❌ Audio not found")
                continue
            
            if file_id not in self.annotations:
                print(f"⚠️  No annotations")
                continue
            
            try:
                # Get predictions (use __call__ method)
                predicted = self.vad(str(audio_path))
                reference = self.annotations[file_id]
                
                # Compute metrics
                metrics = self._compute_metrics(predicted, reference, file_id)
                self.results.append(metrics)
                
                print(f"✓ (F1: {metrics['f1']:.3f}, DER: {metrics['der']:.3f})")
                
            except Exception as e:
                print(f"❌ Error: {e}")
        
        # Check if we have any results
        if not self.results:
            print("\n❌ ERROR: No results collected! All evaluations failed.")
            print("   Please check:")
            print("   1. Audio files exist in test directory")
            print("   2. Model is working correctly")
            print("   3. RTTM annotations are valid")
            return
        
        # Aggregate metrics
        self._aggregate_metrics()
        
        # Print results to console (no visualizations)
        self._print_results()
        
        # Save results (optional)
        # Uncomment the line below to save detailed log and JSON files
        # self._save_results()
    
    def _compute_metrics(
        self,
        predicted: List[Tuple[float, float]],
        reference: List[Tuple[float, float]],
        file_id: str
    ) -> Dict:
        """
        Compute evaluation metrics for a single file
        
        Metrics:
        - Precision, Recall, F1
        - Detection Error Rate (DER)
        - False Alarm Rate (FA)
        - Miss Rate (MISS)
        """
        # Get audio duration
        audio_path = self.test_audio_dir / f"{file_id}.wav"
        waveform, sr = torchaudio.load(str(audio_path))
        total_duration = waveform.shape[1] / sr
        
        # Frame-level evaluation (10ms frames)
        frame_duration = 0.01
        num_frames = int(total_duration / frame_duration)
        
        ref_frames = np.zeros(num_frames, dtype=bool)
        pred_frames = np.zeros(num_frames, dtype=bool)
        
        # Mark reference frames
        for start, end in reference:
            start_frame = int(start / frame_duration)
            end_frame = int(end / frame_duration)
            ref_frames[start_frame:end_frame] = True
        
        # Mark predicted frames
        for start, end in predicted:
            start_frame = int(start / frame_duration)
            end_frame = int(end / frame_duration)
            pred_frames[start_frame:end_frame] = True
        
        # Compute confusion matrix
        tp = np.sum(ref_frames & pred_frames)
        fp = np.sum(~ref_frames & pred_frames)
        fn = np.sum(ref_frames & ~pred_frames)
        tn = np.sum(~ref_frames & ~pred_frames)
        
        # Compute metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # DER components (as percentage of total duration)
        speech_duration = np.sum(ref_frames) * frame_duration
        fa_rate = fp * frame_duration / total_duration
        miss_rate = fn * frame_duration / total_duration
        der = (fp + fn) * frame_duration / total_duration
        
        return {
            'file_id': file_id,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'der': der,
            'fa_rate': fa_rate,
            'miss_rate': miss_rate,
            'tp': int(tp),
            'fp': int(fp),
            'fn': int(fn),
            'tn': int(tn),
            'total_duration': total_duration,
            'speech_duration': speech_duration,
            'num_pred_segments': len(predicted),
            'num_ref_segments': len(reference)
        }
    
    def _aggregate_metrics(self):
        """Aggregate metrics across all files"""
        if not self.results:
            print("⚠️  No results to aggregate")
            return
        
        # Compute averages
        self.metrics = {
            'num_files': len(self.results),
            'avg_precision': np.mean([r['precision'] for r in self.results]),
            'avg_recall': np.mean([r['recall'] for r in self.results]),
            'avg_f1': np.mean([r['f1'] for r in self.results]),
            'avg_der': np.mean([r['der'] for r in self.results]),
            'avg_fa_rate': np.mean([r['fa_rate'] for r in self.results]),
            'avg_miss_rate': np.mean([r['miss_rate'] for r in self.results]),
            'total_duration': sum(r['total_duration'] for r in self.results),
            'total_speech_duration': sum(r['speech_duration'] for r in self.results),
            'total_tp': sum(r['tp'] for r in self.results),
            'total_fp': sum(r['fp'] for r in self.results),
            'total_fn': sum(r['fn'] for r in self.results),
            'total_tn': sum(r['tn'] for r in self.results)
        }
        
        # Compute standard deviations
        self.metrics['std_precision'] = np.std([r['precision'] for r in self.results])
        self.metrics['std_recall'] = np.std([r['recall'] for r in self.results])
        self.metrics['std_f1'] = np.std([r['f1'] for r in self.results])
        self.metrics['std_der'] = np.std([r['der'] for r in self.results])
        
        print("\n" + "="*80)
        print("EVALUATION RESULTS")
        print("="*80)
        print(f"\n📊 Overall Metrics:\n")
        print(f"   Precision:  {self.metrics['avg_precision']:.4f} ± {self.metrics['std_precision']:.4f}")
        print(f"   Recall:     {self.metrics['avg_recall']:.4f} ± {self.metrics['std_recall']:.4f}")
        print(f"   F1-Score:   {self.metrics['avg_f1']:.4f} ± {self.metrics['std_f1']:.4f}")
        print(f"   DER:        {self.metrics['avg_der']:.4f} ± {self.metrics['std_der']:.4f}")
        print(f"   FA Rate:    {self.metrics['avg_fa_rate']:.4f}")
        print(f"   Miss Rate:  {self.metrics['avg_miss_rate']:.4f}")
        print(f"\n   Total audio: {self.metrics['total_duration']/3600:.2f} hours")
        print(f"   Total speech: {self.metrics['total_speech_duration']/3600:.2f} hours")
    
    def _print_results(self):
        """Print evaluation results to console (no files or visualizations)"""
        print("\n" + "="*80)
        print("EVALUATION RESULTS")
        print("="*80)
        
        # Overall metrics
        print("\n📊 OVERALL PERFORMANCE METRICS")
        print("-" * 80)
        print(f"Precision:                {self.metrics['avg_precision']:.4f} ± {self.metrics['std_precision']:.4f}")
        print(f"Recall:                   {self.metrics['avg_recall']:.4f} ± {self.metrics['std_recall']:.4f}")
        print(f"F1-Score:                 {self.metrics['avg_f1']:.4f} ± {self.metrics['std_f1']:.4f}")
        print(f"Detection Error Rate:     {self.metrics['avg_der']:.4f} ± {self.metrics['std_der']:.4f}")
        print(f"False Alarm Rate:         {self.metrics['avg_fa_rate']:.4f}")
        print(f"Miss Rate:                {self.metrics['avg_miss_rate']:.4f}")
        
        # Dataset statistics
        print("\n📊 DATASET STATISTICS")
        print("-" * 80)
        print(f"Number of files:          {len(self.results)}")
        print(f"Total audio duration:     {self.metrics['total_duration']/3600:.2f} hours")
        print(f"Total speech duration:    {self.metrics['total_speech_duration']/3600:.2f} hours")
        print(f"Speech ratio:             {(self.metrics['total_speech_duration']/self.metrics['total_duration']*100):.1f}%")
        
        # Confusion matrix
        print("\n📊 CONFUSION MATRIX (Total Frames)")
        print("-" * 80)
        print(f"True Positives (TP):      {self.metrics['total_tp']:,}")
        print(f"False Positives (FP):     {self.metrics['total_fp']:,}")
        print(f"False Negatives (FN):     {self.metrics['total_fn']:,}")
        print(f"True Negatives (TN):      {self.metrics['total_tn']:,}")
        
        # Best and worst files
        sorted_results = sorted(self.results, key=lambda x: x['f1'], reverse=True)
        
        print("\n📊 TOP 5 BEST FILES")
        print("-" * 80)
        print(f"{'File ID':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'DER':>10}")
        print("-" * 80)
        for r in sorted_results[:5]:
            print(f"{r['file_id']:<20} {r['precision']:>10.4f} {r['recall']:>10.4f} {r['f1']:>10.4f} {r['der']:>10.4f}")
        
        print("\n📊 TOP 5 WORST FILES")
        print("-" * 80)
        print(f"{'File ID':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'DER':>10}")
        print("-" * 80)
        for r in sorted_results[-5:]:
            print(f"{r['file_id']:<20} {r['precision']:>10.4f} {r['recall']:>10.4f} {r['f1']:>10.4f} {r['der']:>10.4f}")
        
        print("\n" + "="*80)
        print("✅ EVALUATION COMPLETE - Results printed above")
        print("="*80)
    
    def _generate_visualizations(self):
        """Generate evaluation visualizations"""
        print(f"\n📊 Generating visualizations...")
        
        # 1. Performance metrics bar chart
        self._plot_metrics_overview()
        
        # 2. Confusion matrix
        self._plot_confusion_matrix()
        
        # 3. Per-file F1 scores
        self._plot_per_file_scores()
        
        # 4. DER distribution
        self._plot_der_distribution()
        
        # 5. Sample predictions (first 5 files)
        self._plot_sample_predictions(num_samples=5)
        
        print(f"   ✅ Saved {5} visualizations to {self.output_dir}/")
    
    def _plot_metrics_overview(self):
        """Bar chart of main metrics"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        metrics = ['Precision', 'Recall', 'F1-Score']
        values = [
            self.metrics['avg_precision'],
            self.metrics['avg_recall'],
            self.metrics['avg_f1']
        ]
        errors = [
            self.metrics['std_precision'],
            self.metrics['std_recall'],
            self.metrics['std_f1']
        ]
        
        bars = ax.bar(metrics, values, yerr=errors, capsize=10, 
                     color=['#3498db', '#2ecc71', '#e74c3c'], alpha=0.8)
        
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('VAD Model Performance Metrics', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1.0)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'metrics_overview.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_confusion_matrix(self):
        """Normalized confusion matrix"""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        tp = self.metrics['total_tp']
        fp = self.metrics['total_fp']
        fn = self.metrics['total_fn']
        tn = self.metrics['total_tn']
        
        # Normalize by row (actual class)
        total_speech = tp + fn
        total_nonspeech = fp + tn
        
        cm_norm = np.array([
            [tp / total_speech, fn / total_speech],
            [fp / total_nonspeech, tn / total_nonspeech]
        ])
        
        im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
        
        # Labels
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Speech', 'Non-Speech'])
        ax.set_yticklabels(['Speech', 'Non-Speech'])
        ax.set_xlabel('Predicted', fontsize=12)
        ax.set_ylabel('Actual', fontsize=12)
        ax.set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
        
        # Add text annotations
        for i in range(2):
            for j in range(2):
                text = ax.text(j, i, f'{cm_norm[i, j]:.3f}',
                             ha="center", va="center", color="black", fontweight='bold')
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Proportion', rotation=270, labelpad=20)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_per_file_scores(self):
        """F1 scores for each file"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        file_ids = [r['file_id'] for r in self.results]
        f1_scores = [r['f1'] for r in self.results]
        
        # Sort by F1 score
        sorted_indices = np.argsort(f1_scores)
        file_ids = [file_ids[i] for i in sorted_indices]
        f1_scores = [f1_scores[i] for i in sorted_indices]
        
        colors = ['#e74c3c' if f1 < 0.7 else '#f39c12' if f1 < 0.85 else '#2ecc71' 
                 for f1 in f1_scores]
        
        ax.barh(range(len(file_ids)), f1_scores, color=colors, alpha=0.8)
        ax.set_yticks(range(len(file_ids)))
        ax.set_yticklabels([f[:15] for f in file_ids], fontsize=8)
        ax.set_xlabel('F1-Score', fontsize=12)
        ax.set_title('Per-File F1 Scores', fontsize=14, fontweight='bold')
        ax.set_xlim(0, 1.0)
        ax.grid(axis='x', alpha=0.3)
        
        # Add legend
        red_patch = mpatches.Patch(color='#e74c3c', label='Poor (< 0.70)')
        orange_patch = mpatches.Patch(color='#f39c12', label='Fair (0.70-0.85)')
        green_patch = mpatches.Patch(color='#2ecc71', label='Good (> 0.85)')
        ax.legend(handles=[red_patch, orange_patch, green_patch], loc='lower right')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'per_file_scores.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_der_distribution(self):
        """DER distribution histogram"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        der_values = [r['der'] for r in self.results]
        
        ax.hist(der_values, bins=20, color='#3498db', alpha=0.7, edgecolor='black')
        ax.axvline(self.metrics['avg_der'], color='red', linestyle='--', linewidth=2,
                  label=f"Mean: {self.metrics['avg_der']:.3f}")
        ax.set_xlabel('Detection Error Rate (DER)', fontsize=12)
        ax.set_ylabel('Number of Files', fontsize=12)
        ax.set_title('DER Distribution Across Test Files', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'der_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_sample_predictions(self, num_samples=5):
        """Plot predicted vs reference segments for sample files"""
        samples = self.results[:num_samples]
        
        fig, axes = plt.subplots(num_samples, 1, figsize=(14, 3*num_samples))
        if num_samples == 1:
            axes = [axes]
        
        for idx, result in enumerate(samples):
            ax = axes[idx]
            file_id = result['file_id']
            
            # Get predictions and reference (use __call__ method)
            audio_path = self.test_audio_dir / f"{file_id}.wav"
            predicted = self.vad(str(audio_path))
            reference = self.annotations[file_id]
            
            # Plot reference segments
            for start, end in reference:
                ax.add_patch(plt.Rectangle((start, 0.5), end-start, 0.4,
                                          facecolor='#2ecc71', alpha=0.6, label='Reference'))
            
            # Plot predicted segments
            for start, end in predicted:
                ax.add_patch(plt.Rectangle((start, 0.0), end-start, 0.4,
                                          facecolor='#3498db', alpha=0.6, label='Predicted'))
            
            ax.set_ylim(-0.1, 1.0)
            ax.set_xlim(0, result['total_duration'])
            ax.set_xlabel('Time (seconds)', fontsize=10)
            ax.set_yticks([0.2, 0.7])
            ax.set_yticklabels(['Predicted', 'Reference'])
            ax.set_title(f"{file_id} (F1: {result['f1']:.3f}, DER: {result['der']:.3f})",
                        fontsize=11, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            
            # Remove duplicate legend entries
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), loc='upper right')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'sample_predictions.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _save_results(self):
        """Save detailed results to log file and JSON"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Create detailed log file
        log_path = self.output_dir / 'evaluation_results.log'
        
        with open(log_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("STANDALONE VAD MODEL - EVALUATION REPORT\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Evaluation Date: {timestamp}\n")
            f.write(f"Model Path: {self.model_path}\n")
            f.write(f"Test Set: {self.test_audio_dir}\n")
            f.write(f"Number of Test Files: {self.metrics['num_files']}\n\n")
            
            f.write("="*80 + "\n")
            f.write("OVERALL PERFORMANCE METRICS\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Precision:       {self.metrics['avg_precision']:.4f} ± {self.metrics['std_precision']:.4f}\n")
            f.write(f"Recall:          {self.metrics['avg_recall']:.4f} ± {self.metrics['std_recall']:.4f}\n")
            f.write(f"F1-Score:        {self.metrics['avg_f1']:.4f} ± {self.metrics['std_f1']:.4f}\n")
            f.write(f"Detection Error Rate (DER): {self.metrics['avg_der']:.4f} ± {self.metrics['std_der']:.4f}\n")
            f.write(f"False Alarm Rate:     {self.metrics['avg_fa_rate']:.4f}\n")
            f.write(f"Miss Rate:            {self.metrics['avg_miss_rate']:.4f}\n\n")
            
            f.write("="*80 + "\n")
            f.write("DATASET STATISTICS\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Total Audio Duration:  {self.metrics['total_duration']/3600:.2f} hours\n")
            f.write(f"Total Speech Duration: {self.metrics['total_speech_duration']/3600:.2f} hours\n")
            f.write(f"Speech Ratio:          {self.metrics['total_speech_duration']/self.metrics['total_duration']*100:.1f}%\n\n")
            
            f.write("="*80 + "\n")
            f.write("CONFUSION MATRIX (Total Frames @ 10ms)\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"True Positives (TP):  {self.metrics['total_tp']:,}\n")
            f.write(f"False Positives (FP): {self.metrics['total_fp']:,}\n")
            f.write(f"False Negatives (FN): {self.metrics['total_fn']:,}\n")
            f.write(f"True Negatives (TN):  {self.metrics['total_tn']:,}\n\n")
            
            f.write("="*80 + "\n")
            f.write("PER-FILE RESULTS\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"{'File ID':<30} {'Precision':>10} {'Recall':>10} {'F1':>10} {'DER':>10}\n")
            f.write("-"*80 + "\n")
            
            for result in sorted(self.results, key=lambda x: x['f1'], reverse=True):
                f.write(f"{result['file_id']:<30} "
                       f"{result['precision']:>10.4f} "
                       f"{result['recall']:>10.4f} "
                       f"{result['f1']:>10.4f} "
                       f"{result['der']:>10.4f}\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("EVALUATION COMPLETE\n")
            f.write("="*80 + "\n")
        
        print(f"\n✅ Detailed log saved to: {log_path}")
        
        # Save JSON for programmatic access
        json_path = self.output_dir / 'evaluation_results.json'
        output_data = {
            'timestamp': timestamp,
            'model_path': str(self.model_path),
            'test_set': str(self.test_audio_dir),
            'overall_metrics': self.metrics,
            'per_file_results': self.results
        }
        
        with open(json_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"✅ JSON results saved to: {json_path}")


def main():
    """Main evaluation script"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate VAD Model on Test Set")
    parser.add_argument(
        "--model",
        type=str,
        default="experiments_standalone/vad_standalone_v1/final_model/model.pt",
        help="Path to trained model"
    )
    parser.add_argument(
        "--test-audio-dir",
        type=str,
        default="D:/ECHO_V3/AI_subtitle_generator/voxconverse/test",
        help="Directory containing test audio files"
    )
    parser.add_argument(
        "--test-rttm",
        type=str,
        default="D:/ECHO_V3/AI_subtitle_generator/nemo_vad_movie/test.rttm",
        help="Path to test RTTM annotations"
    )
    parser.add_argument(
        "--test-lst",
        type=str,
        default="D:/ECHO_V3/AI_subtitle_generator/nemo_vad_movie/test.lst",
        help="Path to test file list"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation_results",
        help="Output directory for results"
    )
    
    args = parser.parse_args()
    
    # Check files exist
    if not Path(args.model).exists():
        print(f"❌ Error: Model not found at {args.model}")
        sys.exit(1)
    
    if not Path(args.test_audio_dir).exists():
        print(f"❌ Error: Test audio directory not found: {args.test_audio_dir}")
        sys.exit(1)
    
    if not Path(args.test_rttm).exists():
        print(f"❌ Error: Test RTTM not found: {args.test_rttm}")
        sys.exit(1)
    
    if not Path(args.test_lst).exists():
        print(f"❌ Error: Test LST not found: {args.test_lst}")
        sys.exit(1)
    
    # Run evaluation
    evaluator = VADEvaluator(
        model_path=args.model,
        test_audio_dir=args.test_audio_dir,
        test_rttm_path=args.test_rttm,
        test_lst_path=args.test_lst,
        output_dir=args.output_dir
    )
    
    # Quick evaluation: only first 10 files
    evaluator.evaluate_all(max_files=3)
    
    # Results already printed by _print_results()
    print("\n💡 Tip: To save results to files, uncomment self._save_results() in the code")


if __name__ == "__main__":
    main()

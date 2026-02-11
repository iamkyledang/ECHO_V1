"""
SRT Evaluation Tool
Compares generated subtitles (result_data/) with ground truth (srt_data/)
Calculates accuracy metrics: WER, timing precision, and quality scores

Requirements: pip install jiwer python-Levenshtein
"""

import os
import re
from typing import List, Tuple, Dict
from dataclasses import dataclass
import statistics

try:
    from jiwer import wer, cer
    HAS_JIWER = True
except ImportError:
    HAS_JIWER = False
    print("⚠️  Warning: jiwer not installed. Install with: pip install jiwer")

@dataclass
class SubtitleEntry:
    """Represents a single subtitle entry"""
    index: int
    start_time: float
    end_time: float
    text: str
    
    @property
    def duration(self):
        return self.end_time - self.start_time

def parse_srt_time(time_str: str) -> float:
    """
    Convert SRT timestamp to seconds.
    Format: HH:MM:SS,mmm
    """
    try:
        time_part, ms_part = time_str.strip().split(',')
        h, m, s = map(int, time_part.split(':'))
        ms = int(ms_part)
        return h * 3600 + m * 60 + s + ms / 1000.0
    except Exception as e:
        print(f"⚠️  Error parsing time '{time_str}': {e}")
        return 0.0

def parse_srt_file(filepath: str) -> List[SubtitleEntry]:
    """Parse an SRT file and return list of subtitle entries"""
    entries = []
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split by double newline to get individual entries
        blocks = re.split(r'\n\s*\n', content.strip())
        
        for block in blocks:
            lines = block.strip().split('\n')
            if len(lines) < 3:
                continue
            
            try:
                # Parse index
                index = int(lines[0].strip())
                
                # Parse timestamp
                time_match = re.match(r'(\S+)\s*-->\s*(\S+)', lines[1])
                if not time_match:
                    continue
                
                start_time = parse_srt_time(time_match.group(1))
                end_time = parse_srt_time(time_match.group(2))
                
                # Parse text (may span multiple lines)
                text = ' '.join(lines[2:]).strip()
                
                entries.append(SubtitleEntry(index, start_time, end_time, text))
            except Exception as e:
                continue
        
        return entries
    except FileNotFoundError:
        print(f"❌ File not found: {filepath}")
        return []
    except Exception as e:
        print(f"❌ Error parsing {filepath}: {e}")
        return []

def calculate_timing_offset(ref_entries: List[SubtitleEntry], 
                            gen_entries: List[SubtitleEntry]) -> Tuple[float, float]:
    """
    Calculate average timing offset between reference and generated subtitles.
    Returns: (mean_offset, std_deviation)
    """
    if not ref_entries or not gen_entries:
        return 0.0, 0.0
    
    offsets = []
    
    # Compare entries that overlap
    for ref in ref_entries:
        for gen in gen_entries:
            # Check if entries overlap in time
            overlap_start = max(ref.start_time, gen.start_time)
            overlap_end = min(ref.end_time, gen.end_time)
            
            if overlap_start < overlap_end:
                # Calculate start time difference
                offset = abs(ref.start_time - gen.start_time)
                offsets.append(offset)
                break
    
    if not offsets:
        return 0.0, 0.0
    
    mean_offset = statistics.mean(offsets)
    std_offset = statistics.stdev(offsets) if len(offsets) > 1 else 0.0
    
    return mean_offset, std_offset

def evaluate_srt_pair(reference_path: str, generated_path: str) -> Dict:
    """
    Evaluate a single SRT file pair.
    Returns metrics dictionary.
    """
    ref_entries = parse_srt_file(reference_path)
    gen_entries = parse_srt_file(generated_path)
    
    # Extract all text
    ref_text = ' '.join(entry.text for entry in ref_entries)
    gen_text = ' '.join(entry.text for entry in gen_entries)
    
    # Calculate metrics
    metrics = {
        'reference_entries': len(ref_entries),
        'generated_entries': len(gen_entries),
        'reference_chars': len(ref_text),
        'generated_chars': len(gen_text),
        'reference_words': len(ref_text.split()),
        'generated_words': len(gen_text.split()),
    }
    
    # Calculate WER and CER if jiwer is available
    if HAS_JIWER and ref_text and gen_text:
        try:
            metrics['wer'] = wer(ref_text, gen_text) * 100  # Convert to percentage
            metrics['cer'] = cer(ref_text, gen_text) * 100
        except Exception as e:
            metrics['wer'] = None
            metrics['cer'] = None
    else:
        metrics['wer'] = None
        metrics['cer'] = None
    
    # Calculate timing metrics
    mean_offset, std_offset = calculate_timing_offset(ref_entries, gen_entries)
    metrics['timing_offset_mean'] = mean_offset
    metrics['timing_offset_std'] = std_offset
    
    # Calculate average entry duration
    if ref_entries:
        metrics['ref_avg_duration'] = statistics.mean(e.duration for e in ref_entries)
    else:
        metrics['ref_avg_duration'] = 0.0
    
    if gen_entries:
        metrics['gen_avg_duration'] = statistics.mean(e.duration for e in gen_entries)
    else:
        metrics['gen_avg_duration'] = 0.0
    
    return metrics

def print_table_header():
    """Print evaluation results table header"""
    print("\n" + "="*150)
    print("SRT EVALUATION RESULTS")
    print("="*150)
    print(f"{'File Name':<35} | {'Entries':<12} | {'Words':<12} | {'WER %':<8} | {'CER %':<8} | {'Timing Δ':<12} | {'Quality':<8}")
    print(f"{'':35} | {'Ref → Gen':<12} | {'Ref → Gen':<12} | {'':<8} | {'':<8} | {'(seconds)':<12} | {'':<8}")
    print("-"*150)

def print_table_row(filename: str, metrics: Dict):
    """Print a single row of evaluation results"""
    # Format entries
    entries_str = f"{metrics['reference_entries']:>3} → {metrics['generated_entries']:<3}"
    
    # Format words
    words_str = f"{metrics['reference_words']:>4} → {metrics['generated_words']:<4}"
    
    # Format WER
    wer_str = f"{metrics['wer']:.1f}" if metrics['wer'] is not None else "N/A"
    
    # Format CER
    cer_str = f"{metrics['cer']:.1f}" if metrics['cer'] is not None else "N/A"
    
    # Format timing
    timing_str = f"{metrics['timing_offset_mean']:.3f}±{metrics['timing_offset_std']:.3f}"
    
    # Calculate quality score (lower WER = better)
    if metrics['wer'] is not None:
        if metrics['wer'] < 10:
            quality = "Excellent"
        elif metrics['wer'] < 20:
            quality = "Good"
        elif metrics['wer'] < 40:
            quality = "Fair"
        else:
            quality = "Poor"
    else:
        quality = "N/A"
    
    print(f"{filename:<35} | {entries_str:<12} | {words_str:<12} | {wer_str:<8} | {cer_str:<8} | {timing_str:<12} | {quality:<8}")

def print_summary(all_metrics: List[Tuple[str, Dict]]):
    """Print summary statistics"""
    print("-"*150)
    
    # Calculate averages
    valid_wers = [m['wer'] for _, m in all_metrics if m['wer'] is not None]
    valid_cers = [m['cer'] for _, m in all_metrics if m['cer'] is not None]
    valid_timings = [m['timing_offset_mean'] for _, m in all_metrics]
    
    total_ref_entries = sum(m['reference_entries'] for _, m in all_metrics)
    total_gen_entries = sum(m['generated_entries'] for _, m in all_metrics)
    total_ref_words = sum(m['reference_words'] for _, m in all_metrics)
    total_gen_words = sum(m['generated_words'] for _, m in all_metrics)
    
    print(f"\n{'SUMMARY STATISTICS':<35} | {'':<12} | {'':<12} | {'':<8} | {'':<8} | {'':<12} | {'':<8}")
    print("-"*150)
    
    if valid_wers:
        avg_wer = statistics.mean(valid_wers)
        avg_cer = statistics.mean(valid_cers) if valid_cers else 0.0
        avg_timing = statistics.mean(valid_timings) if valid_timings else 0.0
        
        entries_str = f"{total_ref_entries:>3} → {total_gen_entries:<3}"
        words_str = f"{total_ref_words:>4} → {total_gen_words:<4}"
        
        print(f"{'Average':<35} | {entries_str:<12} | {words_str:<12} | {avg_wer:<8.1f} | {avg_cer:<8.1f} | {avg_timing:<12.3f} | {'':<8}")
        
        # Overall quality assessment
        if avg_wer < 15:
            overall = "✅ Excellent"
        elif avg_wer < 25:
            overall = "✅ Good"
        elif avg_wer < 40:
            overall = "⚠️  Fair"
        else:
            overall = "❌ Needs Work"
        
        print(f"\n{'Overall Quality:':<35} {overall}")
    else:
        print("No valid WER calculations (install jiwer: pip install jiwer)")
    
    print("="*150)

def main():
    """Main evaluation function"""
    # Define directories
    reference_dir = "srt_data"  # Ground truth subtitles
    generated_dir = "result_data"  # Your generated subtitles
    
    print("\n" + "="*150)
    print("🔍 SRT SUBTITLE EVALUATION TOOL")
    print("="*150)
    print(f"📁 Reference (Ground Truth): {os.path.abspath(reference_dir)}")
    print(f"📁 Generated (Your Output):  {os.path.abspath(generated_dir)}")
    
    # Check if directories exist
    if not os.path.exists(reference_dir):
        print(f"\n❌ ERROR: Reference directory not found: {reference_dir}")
        print(f"   Run 'python download_ted_talk_srt.py' first to download ground truth subtitles")
        return
    
    if not os.path.exists(generated_dir):
        print(f"\n❌ ERROR: Generated directory not found: {generated_dir}")
        print(f"   Generate subtitles using your software first")
        return
    
    # Get list of SRT files in reference directory
    # Handle both .srt and .srt.en extensions (YouTube downloads often have .en suffix)
    ref_files_raw = [f for f in os.listdir(reference_dir) if f.endswith('.srt') or f.endswith('.srt.en')]
    
    if not ref_files_raw:
        print(f"\n❌ ERROR: No SRT files found in {reference_dir}")
        return
    
    print(f"\n✅ Found {len(ref_files_raw)} reference subtitle files")
    
    if not HAS_JIWER:
        print("\n⚠️  WARNING: jiwer not installed - WER/CER metrics will be unavailable")
        print("   Install with: pip install jiwer")
    
    # Build file pairs mapping
    # Extract base names to match ref files with generated files
    file_pairs = []
    for ref_file in sorted(ref_files_raw):
        # Extract base name (remove .srt.en or .srt)
        if ref_file.endswith('.srt.en'):
            base_name = ref_file[:-7]  # Remove '.srt.en'
        elif ref_file.endswith('.en.srt'):
            base_name = ref_file[:-7]  # Remove '.en.srt'
        else:
            base_name = ref_file[:-4]  # Remove '.srt'
        
        # Look for matching generated file (should be base_name.srt)
        gen_file = base_name
        file_pairs.append((ref_file, gen_file, base_name))
    
    # Evaluate each file pair
    all_metrics = []
    print_table_header()
    
    for ref_file, gen_file, base_name in file_pairs:
        ref_path = os.path.join(reference_dir, ref_file)
        gen_path = os.path.join(generated_dir, gen_file)
        
        # Display base name for clarity
        display_name = gen_file  # Show the expected generated filename
        
        if not os.path.exists(gen_path):
            print(f"{display_name:<35} | {'NOT FOUND':<12} | {'-':<12} | {'-':<8} | {'-':<8} | {'-':<12} | {'N/A':<8}")
            continue
        
        # Evaluate this pair
        metrics = evaluate_srt_pair(ref_path, gen_path)
        all_metrics.append((display_name, metrics))
        
        # Print results
        print_table_row(display_name, metrics)
    
    # Print summary
    if all_metrics:
        print_summary(all_metrics)
    
    # Print legend
    print("\n📊 METRICS LEGEND:")
    print("   • Entries: Number of subtitle segments")
    print("   • Words: Total word count")
    print("   • WER: Word Error Rate (lower is better, <15% is excellent)")
    print("   • CER: Character Error Rate (lower is better)")
    print("   • Timing Δ: Average timing offset ± standard deviation (seconds)")
    print("   • Quality: Overall assessment (Excellent/Good/Fair/Poor)")
    
    print("\n💡 TIPS:")
    print("   • WER < 10%: Excellent transcription accuracy")
    print("   • WER 10-20%: Good quality, minor errors")
    print("   • WER 20-40%: Fair quality, needs improvement")
    print("   • WER > 40%: Poor quality, major issues")
    print("   • Timing Δ < 0.5s: Excellent synchronization")
    print("   • Timing Δ 0.5-1.0s: Good synchronization")
    print("   • Timing Δ > 1.0s: Poor synchronization")
    
    print("\n" + "="*150)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Evaluation interrupted by user. Exiting...")
    except Exception as e:
        print(f"\n\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()

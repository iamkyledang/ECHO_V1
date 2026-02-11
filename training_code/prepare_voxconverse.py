#!/usr/bin/env python3
"""
Prepare VoxConverse dataset for Pyannote.audio training
This script generates .lst and .rttm files from the voxconverse dataset structure
"""

import os
from pathlib import Path
import shutil

def prepare_voxconverse_dataset():
    """
    Prepares VoxConverse dataset by:
    1. Creating file lists (.lst files)
    2. Organizing RTTM annotations
    3. Verifying data integrity
    """
    
    # Define paths
    base_path = Path("D:/ECHO_V3/AI_subtitle_generator/voxconverse")
    nemo_path = Path("D:/ECHO_V3/AI_subtitle_generator/nemo_vad_movie")
    
    subsets = ["dev", "test"]
    
    print("="*80)
    print("PREPARING VOXCONVERSE DATASET FOR PYANNOTE.AUDIO")
    print("="*80)
    
    for subset in subsets:
        subset_path = base_path / subset
        
        if not subset_path.exists():
            print(f"\n⚠️  Warning: {subset_path} does not exist, skipping...")
            continue
        
        print(f"\n📁 Processing {subset} set...")
        
        # Get all .rttm files
        rttm_files = list(subset_path.glob("*.rttm"))
        print(f"   Found {len(rttm_files)} RTTM files")
        
        # Create .lst file (list of file IDs without extension)
        lst_file = nemo_path / f"{subset}.lst"
        file_ids = []
        
        for rttm_file in sorted(rttm_files):
            file_id = rttm_file.stem  # Get filename without extension
            
            # Check if corresponding .wav file exists
            wav_file = subset_path / f"{file_id}.wav"
            
            if not wav_file.exists():
                print(f"   ⚠️  Warning: {wav_file.name} not found for {rttm_file.name}")
                continue
            
            file_ids.append(file_id)
        
        # Write .lst file
        with open(lst_file, 'w') as f:
            for file_id in file_ids:
                f.write(f"{file_id}\n")
        
        print(f"   ✅ Created {lst_file.name} with {len(file_ids)} entries")
        
        # Create combined .rttm file for the entire subset
        combined_rttm = nemo_path / f"{subset}.rttm"
        with open(combined_rttm, 'w') as out_f:
            for rttm_file in sorted(rttm_files):
                file_id = rttm_file.stem
                if file_id in file_ids:
                    with open(rttm_file, 'r') as in_f:
                        out_f.write(in_f.read())
        
        print(f"   ✅ Created {combined_rttm.name}")
    
    # Create train.lst from dev set (we'll use dev for both train and validation)
    dev_lst = nemo_path / "dev.lst"
    train_lst = nemo_path / "train.lst"
    train_rttm = nemo_path / "train.rttm"
    dev_rttm = nemo_path / "dev.rttm"
    
    if dev_lst.exists():
        # Split dev set into train (80%) and validation (20%)
        with open(dev_lst, 'r') as f:
            all_files = [line.strip() for line in f if line.strip()]
        
        split_point = int(len(all_files) * 0.8)
        train_files = all_files[:split_point]
        dev_files = all_files[split_point:]
        
        # Write train.lst
        with open(train_lst, 'w') as f:
            for file_id in train_files:
                f.write(f"{file_id}\n")
        
        print(f"\n✅ Created {train_lst.name} with {len(train_files)} training samples")
        
        # Update dev.lst with only validation files
        with open(dev_lst, 'w') as f:
            for file_id in dev_files:
                f.write(f"{file_id}\n")
        
        print(f"✅ Updated {dev_lst.name} with {len(dev_files)} validation samples")
        
        # Create train.rttm and update dev.rttm
        base_path_dev = base_path / "dev"
        
        with open(train_rttm, 'w') as train_f, open(dev_rttm, 'w') as dev_f:
            for file_id in train_files:
                rttm_file = base_path_dev / f"{file_id}.rttm"
                if rttm_file.exists():
                    with open(rttm_file, 'r') as in_f:
                        train_f.write(in_f.read())
            
            for file_id in dev_files:
                rttm_file = base_path_dev / f"{file_id}.rttm"
                if rttm_file.exists():
                    with open(rttm_file, 'r') as in_f:
                        dev_f.write(in_f.read())
        
        print(f"✅ Created train.rttm and updated dev.rttm")
    
    print("\n" + "="*80)
    print("DATASET PREPARATION COMPLETE!")
    print("="*80)
    print("\nDataset Summary:")
    
    for lst_file in [train_lst, dev_lst, nemo_path / "test.lst"]:
        if lst_file.exists():
            with open(lst_file, 'r') as f:
                count = len([line for line in f if line.strip()])
            print(f"  • {lst_file.name:15} : {count:4} files")
    
    print("\n✅ Ready to train! Run: python train_vad_pyannote.py")

if __name__ == "__main__":
    prepare_voxconverse_dataset()

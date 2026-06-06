#!/usr/bin/env python3
"""Download and preview HuggingFace datasets directly."""

from loguru import logger
from pathlib import Path
from datasets import load_dataset
from huggingface_hub import dataset_info
import json
import sys
import os

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add("logs/preview.log", rotation="30 MB", level="DEBUG")

@logger.catch(reraise=True)
def download_dataset(dataset_id: str, output_dir: str = "temp/datasets"):
    """Download a dataset and save preview/mini/full versions."""
    logger.info(f"Downloading dataset: {dataset_id}")
    
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Get dataset info
        try:
            info = dataset_info(dataset_id)
            logger.info(f"Downloads: {info.downloads}, Likes: {info.likes}")
        except Exception as e:
            logger.warning(f"Could not fetch dataset info: {e}")
        
        # Try to load dataset
        logger.info(f"Loading dataset...")
        try:
            dataset = load_dataset(dataset_id, trust_remote_code=False)
        except Exception as e:
            logger.warning(f"Failed without config, trying default: {e}")
            try:
                dataset = load_dataset(dataset_id, split="train", trust_remote_code=False)
            except Exception as e2:
                logger.error(f"Could not load dataset: {e2}")
                return False
        
        # Handle both dataset dict (multiple splits) and dataset (single split)
        if isinstance(dataset, dict):
            splits = list(dataset.keys())
            logger.info(f"Dataset has splits: {splits}")
            # Use train split if available, otherwise first split
            split_name = "train" if "train" in splits else splits[0]
            data = dataset[split_name]
        else:
            data = dataset
            split_name = "train"
        
        logger.info(f"Loaded {len(data)} examples from {split_name} split")
        logger.info(f"Columns: {list(data.column_names)}")
        
        # Save preview (3 examples)
        preview = data.select(range(min(3, len(data))))
        preview_path = os.path.join(output_dir, f"preview_{dataset_id.replace('/', '_')}_{split_name}.json")
        preview.to_json(preview_path, orient="records")
        logger.info(f"Saved preview to {preview_path}")
        
        # Save mini (50 examples or all if less)
        mini_size = min(50, len(data))
        mini = data.select(range(mini_size))
        mini_path = os.path.join(output_dir, f"mini_{dataset_id.replace('/', '_')}_{split_name}.json")
        mini.to_json(mini_path, orient="records")
        logger.info(f"Saved mini ({mini_size} examples) to {mini_path}")
        
        # Save full
        full_path = os.path.join(output_dir, f"full_{dataset_id.replace('/', '_')}_{split_name}.json")
        data.to_json(full_path, orient="records")
        logger.info(f"Saved full ({len(data)} examples) to {full_path}")
        
        # Show sample
        logger.info(f"\n--- Sample Rows ---")
        for i in range(min(2, len(data))):
            logger.info(f"\nRow {i+1}:")
            row = data[i]
            for key, value in row.items():
                if isinstance(value, str) and len(value) > 150:
                    value = value[:150] + "..."
                logger.info(f"  {key}: {value}")
        
        logger.info(f"\nSuccessfully downloaded {dataset_id}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to download {dataset_id}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

@logger.catch(reraise=True)
def main():
    # Create output directory
    output_dir = "temp/datasets"
    os.makedirs(output_dir, exist_ok=True)
    
    # These are the datasets that successfully loaded in preview
    datasets_to_download = [
        'tasksource/ruletaker',      # Logical reasoning - RuleTaker
        'tasksource/proofwriter',     # Logical reasoning - ProofWriter  
        'suzakuteam/entailment_bank', # Entailment bank for abductive reasoning
        'flaitenberger/LogicalReasoning-hard-v5',  # Logical reasoning
    ]
    
    results = {}
    
    # Download the known working datasets
    logger.info("Downloading known working datasets...")
    for dataset_id in datasets_to_download:
        logger.info(f"\n{'='*60}")
        success = download_dataset(dataset_id, output_dir)
        results[dataset_id] = "success" if success else "failed"
    
    logger.info(f"\n{'='*60}")
    logger.info("Download Summary:")
    for dataset_id, status in results.items():
        logger.info(f"  {dataset_id}: {status}")

if __name__ == "__main__":
    main()

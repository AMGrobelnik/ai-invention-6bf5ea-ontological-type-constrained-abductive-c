#!/usr/bin/env python3
"""Create final standardized output for the dataset artifact."""

import json
import os
from pathlib import Path
from loguru import logger

logger.remove()
logger.add("logs/final_output.log", rotation="30 MB", level="DEBUG")
logger.add(lambda msg: print(msg, end=""), level="INFO", format="{message}")

@logger.catch(reraise=True)
def main():
    output_dir = Path("temp/datasets")
    
    # Define the datasets we successfully downloaded with their standardized info
    datasets = []
    
    # 1. ROCStories (narrative reasoning)
    filepath = output_dir / "full_mintujupally_ROCStories_train.json"
    if filepath.exists() and filepath.stat().st_size < 300 * 1024 * 1024:
        datasets.append({
            "id": "mintujupally/ROCStories",
            "domain": "narrative",
            "task_type": "narrative reasoning",
            "description": "ROCStories dataset for narrative understanding and commonsense reasoning",
            "file_path": str(filepath),
            "size_mb": round(filepath.stat().st_size / (1024 * 1024), 2),
            "example_count": "5000 (sampled)",
            "suitable_for": "abductive reasoning over narrative contexts"
        })
    
    # 2. SQuAD (reading comprehension)
    filepath = output_dir / "full_rajpurkar_squad_train.json"
    if filepath.exists() and filepath.stat().st_size < 300 * 1024 * 1024:
        datasets.append({
            "id": "rajpurkar/squad",
            "domain": "reading_comprehension",
            "task_type": "question answering",
            "description": "SQuAD v2 for reading comprehension with implicit facts",
            "file_path": str(filepath),
            "size_mb": round(filepath.stat().st_size / (1024 * 1024), 2),
            "example_count": "5000 (sampled)",
            "suitable_for": "abductive reasoning to infer missing context facts"
        })
    
    # 3. SciQ (science Q&A)
    filepath = output_dir / "full_allenai_sciq_train.json"
    if filepath.exists() and filepath.stat().st_size < 300 * 1024 * 1024:
        datasets.append({
            "id": "allenai/sciq",
            "domain": "science_qa",
            "task_type": "multi-hop reasoning",
            "description": "SciQ dataset for science question answering requiring multi-hop reasoning",
            "file_path": str(filepath),
            "size_mb": round(filepath.stat().st_size / (1024 * 1024), 2),
            "example_count": "5000 (sampled)",
            "suitable_for": "abductive reasoning over scientific facts"
        })
    
    # 4. Entailment Bank (abductive reasoning)
    filepath = output_dir / "full_suzakuteam_entailment_bank_train.json"
    if filepath.exists() and filepath.stat().st_size < 300 * 1024 * 1024:
        datasets.append({
            "id": "suzakuteam/entailment_bank",
            "domain": "abductive_reasoning",
            "task_type": "textual entailment",
            "description": "Entailment Bank for abductive reasoning with chain-of-thought proofs",
            "file_path": str(filepath),
            "size_mb": round(filepath.stat().st_size / (1024 * 1024), 2),
            "example_count": 1840,
            "suitable_for": "abductive reasoning with explicit proof chains"
        })
    
    logger.info(f"\n\n{'='*60}")
    logger.info(f"FINAL DATASET INVENTORY: {len(datasets)} datasets")
    logger.info(f"{'='*60}\n")
    
    total_size = 0
    for i, ds in enumerate(datasets, 1):
        logger.info(f"{i}. {ds['id']}")
        logger.info(f"   Domain: {ds['domain']}")
        logger.info(f"   Size: {ds['size_mb']} MB")
        logger.info(f"   Examples: {ds['example_count']}")
        logger.info(f"   Suitable for: {ds['suitable_for']}")
        logger.info("")
        total_size += ds['size_mb']
    
    logger.info(f"Total size: {round(total_size, 2)} MB")
    logger.info(f"Total datasets: {len(datasets)}")
    
    # Create data_out.json
    output = {
        "datasets": datasets,
        "summary": {
            "total_datasets": len(datasets),
            "total_size_mb": round(total_size, 2),
            "domains_covered": list(set([ds["domain"] for ds in datasets])),
            "all_under_300mb": all(ds["size_mb"] < 300 for ds in datasets),
        },
        "methodology": {
            "search_strategy": "HuggingFace Hub search with keyword queries",
            "selection_criteria": "Downloads >100, has documentation, suitable for abductive reasoning evaluation",
            "standardization": "Unified JSON format with input/output structure",
            "limitations": "Some datasets (ROCStories, CLUTRR, CaseHOLD) have unsupported loader scripts; used alternatives"
        }
    }
    
    output_path = Path("data_out.json")
    output_path.write_text(json.dumps(output, indent=2))
    logger.info(f"\nSaved final output to {output_path}")
    
    # Create preview of each dataset
    logger.info(f"\n{'='*60}")
    logger.info("DATASET PREVIEWS")
    logger.info(f"{'='*60}\n")
    
    for ds in datasets:
        filepath = Path(ds["file_path"])
        if filepath.exists():
            data = json.loads(filepath.read_text())
            if isinstance(data, list):
                logger.info(f"{ds['id']}: {len(data)} examples")
                if len(data) > 0:
                    logger.info(f"  Sample: {json.dumps(data[0], indent=2)[:300]}...")
                logger.info("")

if __name__ == "__main__":
    main()

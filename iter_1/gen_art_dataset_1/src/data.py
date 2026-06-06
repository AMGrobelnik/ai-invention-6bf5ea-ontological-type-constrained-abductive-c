#!/usr/bin/env python3
"""Standardize 4 datasets for neuro-symbolic reasoning evaluation."""
import json
from pathlib import Path
from loguru import logger
import sys

logger.remove()
logger.add(sys.stdout, level='INFO', format='{time:HH:mm:ss}|{level:<7}|{message}')
logger.add('logs/run.log', rotation='30 MB', level='DEBUG')

@logger.catch(reraise=True)
def main():
    output_dir = Path('temp/datasets')
    result = {'datasets': []}
    
    # Dataset 1: ROCStories - narrative commonsense reasoning
    fp = output_dir / 'full_mintujupally_ROCStories_train.json'
    if fp.exists():
        logger.info('Processing ROCStories...')
        data = []
        with open(fp, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        
        examples = []
        for i, item in enumerate(data[:1000]):
            text = item.get('text', '')
            sentences = [s.strip() for s in text.split('.') if s.strip()]
            if len(sentences) >= 2:
                inp = '. '.join(sentences[:-1]) + '.'
                out = sentences[-1] + '.'
                examples.append({
                    'input': inp,
                    'output': out,
                    'metadata_row_index': i,
                    'metadata_domain': 'narrative',
                    'metadata_task_type': 'abductive'
                })
        
        result['datasets'].append({'dataset': 'mintujupally/ROCStories', 'examples': examples})
        logger.info(f'ROCStories: {len(examples)} examples')
    
    # Dataset 2: RuleTaker - logical reasoning
    fp = output_dir / 'full_tasksource_ruletaker_train_sampled.json'
    if fp.exists():
        logger.info('Processing RuleTaker...')
        data = []
        with open(fp, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        
        examples = []
        for i, item in enumerate(data):
            context = item.get('context', '')
            question = item.get('question', '')
            label = item.get('label', '')
            config = item.get('config', '')
            
            examples.append({
                'input': f'Context: {context}\nQuestion: {question}',
                'output': label,
                'metadata_row_index': i,
                'metadata_domain': 'logical',
                'metadata_task_type': 'entailment',
                'metadata_config': config
            })
        
        result['datasets'].append({'dataset': 'tasksource/ruletaker', 'examples': examples})
        logger.info(f'RuleTaker: {len(examples)} examples')
    
    # Dataset 3: Entailment Bank - abductive reasoning with implicit facts
    fp = output_dir / 'full_suzakuteam_entailment_bank_train.json'
    if fp.exists():
        logger.info('Processing Entailment Bank...')
        data = []
        with open(fp, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        
        examples = []
        for i, item in enumerate(data[:500]):
            question = item.get('question', '')
            answer = item.get('answer', '')
            cot = item.get('cot', [])
            
            # Use the chain of thought as implicit facts to abduce
            implicit_facts = ' '.join(cot) if isinstance(cot, list) else str(cot)
            
            examples.append({
                'input': question,
                'output': answer,
                'metadata_row_index': i,
                'metadata_domain': 'science',
                'metadata_task_type': 'abductive',
                'metadata_implicit_facts': implicit_facts[:500]
            })
        
        result['datasets'].append({'dataset': 'suzakuteam/entailment_bank', 'examples': examples})
        logger.info(f'Entailment Bank: {len(examples)} examples')
    
    # Dataset 4: ProofWriter - logical reasoning with proofs
    fp = output_dir / 'full_tasksource_proofwriter_train_sampled.json'
    if fp.exists():
        logger.info('Processing ProofWriter...')
        data = []
        with open(fp, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        
        examples = []
        for i, item in enumerate(data):
            context = item.get('context', '')
            question = item.get('question', '')
            answer = item.get('answer', '')
            
            examples.append({
                'input': f'Context: {context}\nQuestion: {question}',
                'output': answer,
                'metadata_row_index': i,
                'metadata_domain': 'logical',
                'metadata_task_type': 'proof'
            })
        
        result['datasets'].append({'dataset': 'tasksource/proofwriter', 'examples': examples})
        logger.info(f'ProofWriter: {len(examples)} examples')
    
    # Save output
    output_path = Path('full_data_out.json')
    output_path.write_text(json.dumps(result, indent=2))
    logger.info(f'Saved to {output_path}')
    
    # Summary
    total = sum(len(ds['examples']) for ds in result['datasets'])
    logger.info(f'Total: {len(result["datasets"])} datasets, {total} examples')

if __name__ == '__main__':
    main()

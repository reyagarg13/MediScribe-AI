#!/usr/bin/env python3
"""
Comprehensive evaluation for TrOCR prescription OCR models.
Includes CER, WER, and entity-level metrics.
"""
import argparse
import json
import os
import torch
from typing import List, Dict, Any, Tuple
import numpy as np
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, TrainerCallback
import jiwer
from tqdm import tqdm
import re


def compute_cer(predicted: str, ground_truth: str) -> float:
    """Compute Character Error Rate."""
    if len(ground_truth) == 0:
        return 1.0 if len(predicted) > 0 else 0.0
    
    # Simple edit distance for CER
    def edit_distance(s1, s2):
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
            
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i-1] == s2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
        
        return dp[m][n]
    
    return edit_distance(predicted, ground_truth) / len(ground_truth)


def compute_wer(predicted: str, ground_truth: str) -> float:
    """Compute Word Error Rate using jiwer."""
    try:
        return jiwer.wer(ground_truth, predicted)
    except:
        return 1.0


def extract_medical_entities(text: str) -> Dict[str, List[str]]:
    """Extract medical entities from prescription text."""
    entities = {
        'medications': [],
        'dosages': [],
        'frequencies': [],
        'durations': []
    }
    
    # Simple regex patterns for entity extraction
    patterns = {
        'dosages': r'\b\d+(?:\.\d+)?\s*(?:mg|ml|g|mcg|iu|units?)\b',
        'frequencies': r'\b(?:once|twice|thrice|\d+\s*times?)\s*(?:daily|a\s*day|per\s*day|bid|tid|qid|q\d+h)\b',
        'durations': r'\b(?:for\s+)?\d+\s*(?:days?|weeks?|months?|hours?)\b'
    }
    
    for entity_type, pattern in patterns.items():
        matches = re.findall(pattern, text.lower())
        entities[entity_type] = matches
    
    # Extract potential medication names (simplified)
    # Remove dosages, frequencies, durations from text, then extract remaining medical-looking words
    clean_text = text.lower()
    for pattern in patterns.values():
        clean_text = re.sub(pattern, '', clean_text)
    
    # Look for capitalized words or common drug patterns
    med_patterns = r'\b[a-z]{3,}(?:ol|in|ine|ate|ide|ium|cillin)\b'
    med_matches = re.findall(med_patterns, clean_text)
    entities['medications'] = med_matches
    
    return entities


def compute_entity_f1(pred_entities: List[str], gt_entities: List[str]) -> Tuple[float, float, float]:
    """Compute precision, recall, and F1 for entity lists."""
    if not gt_entities and not pred_entities:
        return 1.0, 1.0, 1.0
    if not gt_entities:
        return 0.0, 1.0, 0.0
    if not pred_entities:
        return 1.0, 0.0, 0.0
    
    pred_set = set(pred_entities)
    gt_set = set(gt_entities)
    
    intersection = len(pred_set & gt_set)
    precision = intersection / len(pred_set)
    recall = intersection / len(gt_set)
    
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return precision, recall, f1


def compute_ocr_metrics(predictions: List[str], ground_truths: List[str]) -> Dict[str, float]:
    """Compute comprehensive OCR metrics."""
    if len(predictions) != len(ground_truths):
        raise ValueError("Predictions and ground truths must have same length")
    
    # Character and word error rates
    cers = [compute_cer(pred, gt) for pred, gt in zip(predictions, ground_truths)]
    wers = [compute_wer(pred, gt) for pred, gt in zip(predictions, ground_truths)]
    
    # Exact match accuracy
    exact_matches = [pred.strip().lower() == gt.strip().lower() for pred, gt in zip(predictions, ground_truths)]
    
    # Entity-level metrics
    entity_metrics = {'medications': [], 'dosages': [], 'frequencies': [], 'durations': []}
    
    for pred, gt in zip(predictions, ground_truths):
        pred_entities = extract_medical_entities(pred)
        gt_entities = extract_medical_entities(gt)
        
        for entity_type in entity_metrics:
            _, _, f1 = compute_entity_f1(pred_entities[entity_type], gt_entities[entity_type])
            entity_metrics[entity_type].append(f1)
    
    return {
        'cer': np.mean(cers),
        'wer': np.mean(wers),
        'exact_match': np.mean(exact_matches),
        'medication_f1': np.mean(entity_metrics['medications']),
        'dosage_f1': np.mean(entity_metrics['dosages']),
        'frequency_f1': np.mean(entity_metrics['frequencies']),
        'duration_f1': np.mean(entity_metrics['durations'])
    }


class OCRMetricsCallback(TrainerCallback):
    """Callback to compute OCR metrics during training."""
    
    def __init__(self, processor, max_eval_samples: int = 100):
        self.processor = processor
        self.max_eval_samples = max_eval_samples
    
    def on_evaluate(self, args, state, control, model, eval_dataloader, **kwargs):
        """Compute metrics on evaluation set."""
        model.eval()
        predictions = []
        ground_truths = []
        
        print("Computing OCR metrics...")
        
        with torch.no_grad():
            for i, batch in enumerate(eval_dataloader):
                if i >= self.max_eval_samples // args.per_device_eval_batch_size:
                    break
                
                # Generate predictions
                pixel_values = batch['pixel_values'].to(model.device)
                generated_ids = model.generate(
                    pixel_values,
                    max_length=256,
                    num_beams=4,
                    early_stopping=True
                )
                
                # Decode predictions
                batch_predictions = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
                predictions.extend(batch_predictions)
                
                # Decode ground truth (handle label tensor)
                labels = batch['labels'].cpu().numpy()
                labels[labels == -100] = self.processor.tokenizer.pad_token_id
                batch_ground_truths = self.processor.batch_decode(labels, skip_special_tokens=True)
                ground_truths.extend(batch_ground_truths)
        
        # Compute metrics
        metrics = compute_ocr_metrics(predictions, ground_truths)
        
        # Log metrics
        for key, value in metrics.items():
            control.log[f"eval_{key}"] = value
        
        model.train()


def evaluate_model(model_dir: str, test_file: str, output_file: str = None, verbose: bool = True):
    """Evaluate trained model on test set."""
    print(f"Loading model from {model_dir}")
    processor = TrOCRProcessor.from_pretrained(model_dir)
    model = VisionEncoderDecoderModel.from_pretrained(model_dir)
    model.eval()
    
    if torch.cuda.is_available():
        model = model.cuda()
        print("Using GPU for evaluation")
    
    # Load test data
    test_data = []
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            test_data.append(json.loads(line.strip()))
    
    print(f"Evaluating on {len(test_data)} examples...")
    
    predictions = []
    ground_truths = []
    results = []
    
    for item in tqdm(test_data):
        try:
            # Load and process image
            image = Image.open(item['image_path']).convert('RGB')
            inputs = processor(images=image, return_tensors="pt")
            
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # Generate prediction
            with torch.no_grad():
                generated_ids = model.generate(
                    inputs['pixel_values'],
                    max_length=256,
                    num_beams=4,
                    early_stopping=True
                )
            
            predicted_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            ground_truth = item['text']
            
            predictions.append(predicted_text)
            ground_truths.append(ground_truth)
            
            if verbose:
                print(f"\nImage: {item['image_path']}")
                print(f"Ground Truth: {ground_truth}")
                print(f"Prediction:   {predicted_text}")
                print(f"CER: {compute_cer(predicted_text, ground_truth):.3f}")
                print("-" * 50)
            
            results.append({
                'image_path': item['image_path'],
                'ground_truth': ground_truth,
                'prediction': predicted_text,
                'cer': compute_cer(predicted_text, ground_truth),
                'wer': compute_wer(predicted_text, ground_truth)
            })
            
        except Exception as e:
            print(f"Error processing {item['image_path']}: {e}")
            continue
    
    # Compute overall metrics
    metrics = compute_ocr_metrics(predictions, ground_truths)
    
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    for key, value in metrics.items():
        print(f"{key.upper()}: {value:.4f}")
    
    # Save detailed results
    if output_file:
        detailed_results = {
            'metrics': metrics,
            'detailed_results': results
        }
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, indent=2, ensure_ascii=False)
        print(f"\nDetailed results saved to {output_file}")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate TrOCR prescription OCR model")
    parser.add_argument("--model_dir", required=True, help="Directory containing trained model")
    parser.add_argument("--test_file", required=True, help="JSONL file with test data")
    parser.add_argument("--output_file", help="Output file for detailed results")
    parser.add_argument("--quiet", action="store_true", help="Quiet mode (less verbose)")
    
    args = parser.parse_args()
    
    metrics = evaluate_model(
        args.model_dir, 
        args.test_file, 
        args.output_file,
        verbose=not args.quiet
    )


if __name__ == "__main__":
    main()

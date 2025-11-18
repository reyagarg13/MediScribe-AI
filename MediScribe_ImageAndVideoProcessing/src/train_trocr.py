#!/usr/bin/env python3
"""
Enhanced TrOCR fine-tuning for prescription OCR.
Optimized for RTX 4060 with comprehensive evaluation and monitoring.
"""
import argparse
import os
import json
import torch
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available, training without experiment tracking")
from torch.utils.data import DataLoader
from transformers import (
    TrOCRProcessor, 
    VisionEncoderDecoderModel, 
    TrainingArguments, 
    Trainer,
    EarlyStoppingCallback
)
from dataset import PrescriptionDataset, PrescriptionDataCollator, split_dataset
from evaluate import compute_ocr_metrics, OCRMetricsCallback
import numpy as np
from typing import Dict, Any


class WeightedTrainer(Trainer):
    """Custom trainer with confidence-weighted loss for noisy pseudo-labels."""
    
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        confidences = inputs.get("confidences", None)
        
        # Forward pass
        outputs = model(**{k: v for k, v in inputs.items() if k != "confidences"})
        
        if confidences is not None:
            # Weight loss by confidence scores
            loss = outputs.loss
            confidence_weights = confidences.to(loss.device)
            # Normalize weights to maintain scale
            confidence_weights = confidence_weights / (confidence_weights.mean() + 1e-8)
            weighted_loss = loss * confidence_weights.mean()
            
            return (weighted_loss, outputs) if return_outputs else weighted_loss
        
        return (outputs.loss, outputs) if return_outputs else outputs.loss


def setup_model_and_processor(model_name: str = "microsoft/trocr-base-handwritten"):
    """Initialize TrOCR model and processor."""
    print(f"Loading model: {model_name}")
    
    processor = TrOCRProcessor.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name)
    
    # Configure generation parameters for better prescription OCR
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    
    # Generation parameters for inference
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.config.max_length = 256
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3
    model.config.length_penalty = 2.0
    model.config.num_beams = 4
    
    return processor, model


def create_training_args(
    output_dir: str,
    batch_size: int = 2,
    epochs: int = 5,
    learning_rate: float = 5e-5,
    fp16: bool = True,
    save_steps: int = 500,
    eval_steps: int = 500
) -> TrainingArguments:
    """Create optimized training arguments for RTX 4060."""
    
    return TrainingArguments(
        output_dir=output_dir,
        
        # Batch and memory optimization
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=8 // batch_size,  # Effective batch size = 8
        dataloader_pin_memory=True,
        dataloader_num_workers=2,
        
        # Training parameters
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        warmup_steps=100,
        weight_decay=0.01,
        
        # Mixed precision and optimization
        fp16=fp16 and torch.cuda.is_available(),
        gradient_checkpointing=True,  # Save VRAM at cost of speed
        remove_unused_columns=False,
        
        # Force GPU usage
        no_cuda=False,
    
        # Evaluation and saving
        evaluation_strategy="steps",
        eval_steps=eval_steps,
        save_steps=save_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_cer",
        greater_is_better=False,
        
        # Logging
        logging_steps=50,
        logging_dir=f"{output_dir}/logs",
        report_to=["tensorboard", "wandb"] if WANDB_AVAILABLE and wandb.run else ["tensorboard"],
        
        # Reproducibility
        seed=42,
        data_seed=42,
    )


def main():
    parser = argparse.ArgumentParser(description="Train TrOCR for prescription OCR")
    
    # Data arguments
    parser.add_argument("--train_data", required=True, help="Path to training JSONL file")
    parser.add_argument("--val_data", help="Path to validation JSONL file (optional)")
    parser.add_argument("--auto_split", action="store_true", help="Auto-split train_data if no val_data")
    
    # Model arguments
    parser.add_argument("--model_name", default="microsoft/trocr-base-handwritten")
    parser.add_argument("--output_dir", default="./models/trocr_prescription")
    parser.add_argument("--resume_from_checkpoint", help="Path to checkpoint to resume from")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size (RTX 4060 optimal: 2)")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--max_length", type=int, default=256, help="Max sequence length")
    parser.add_argument("--image_size", type=int, default=384, help="Input image size")
    
    # Optimization arguments
    parser.add_argument("--no_fp16", action="store_true", help="Disable mixed precision")
    parser.add_argument("--weighted_loss", action="store_true", help="Use confidence-weighted loss")
    parser.add_argument("--early_stopping_patience", type=int, default=3)
    
    # Logging and monitoring
    parser.add_argument("--wandb_project", help="Weights & Biases project name")
    parser.add_argument("--experiment_name", help="Experiment name for logging")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    # Setup logging
    if args.wandb_project and WANDB_AVAILABLE:
        wandb.init(
            project=args.wandb_project,
            name=args.experiment_name or "trocr-prescription",
            config=vars(args)
        )
    elif args.wandb_project and not WANDB_AVAILABLE:
        print("Warning: wandb_project specified but wandb not available")
    
    # Check GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔥 Using device: {device}")
    if torch.cuda.is_available():
        print(f"🚀 GPU: {torch.cuda.get_device_name()}")
        print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup data
    if args.val_data is None and args.auto_split:
        print("Auto-splitting dataset...")
        args.train_data, args.val_data = split_dataset(args.train_data)
    
    # Initialize model and processor
    processor, model = setup_model_and_processor(args.model_name)
    
    # Force model to GPU
    if torch.cuda.is_available():
        model = model.to(device)
        print(f"✅ Model moved to {device}")
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = PrescriptionDataset(
        args.train_data, 
        processor,
        max_length=args.max_length,
        image_size=args.image_size,
        augment=True,
        debug=args.debug
    )
    
    eval_dataset = None
    if args.val_data and os.path.exists(args.val_data):
        eval_dataset = PrescriptionDataset(
            args.val_data,
            processor,
            max_length=args.max_length,
            image_size=args.image_size,
            augment=False,
            debug=args.debug
        )
    
    # Create data collator
    data_collator = PrescriptionDataCollator(processor)
    
    # Create training arguments
    training_args = create_training_args(
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        fp16=not args.no_fp16
    )
    
    # Setup callbacks
    callbacks = [OCRMetricsCallback(processor)]
    if args.early_stopping_patience > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience))
    
    # Initialize trainer
    trainer_class = WeightedTrainer if args.weighted_loss else Trainer
    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=callbacks,
    )
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Start training
    print("Starting training...")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    
    # Save final model
    print("Saving final model...")
    trainer.save_model()
    processor.save_pretrained(args.output_dir)
    
    # Final evaluation
    if eval_dataset:
        print("Running final evaluation...")
        eval_results = trainer.evaluate()
        print("Final evaluation results:")
        for key, value in eval_results.items():
            print(f"  {key}: {value:.4f}")
    
    # Save training info
    training_info = {
        "args": vars(args),
        "model_name": args.model_name,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "final_eval": eval_results if eval_dataset else None
    }
    
    with open(os.path.join(args.output_dir, "training_info.json"), "w") as f:
        json.dump(training_info, f, indent=2)
    
    print(f"Training completed! Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()

"""Evaluation script for ScreenSpot TRM model."""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import yaml

from models import ScreenBBoxTRMModel
from models.screen_trm_model import ModelConfig
from data import ScreenSpotDataset, collate_fn
from utils import compute_metrics


def get_device(requested: str = "auto") -> torch.device:
    """Get the best available device.
    
    Args:
        requested: Device string ("auto", "cuda", "mps", "cpu")
        
    Returns:
        torch.device for the selected device
    """
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    elif requested == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            print("Warning: CUDA not available, falling back to CPU")
            return torch.device("cpu")
    elif requested == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            print("Warning: MPS not available, falling back to CPU")
            return torch.device("cpu")
    else:
        return torch.device("cpu")


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    save_predictions: bool = False,
) -> Tuple[Dict[str, float], List[Dict]]:
    """Evaluate model on dataset.
    
    Args:
        model: Model to evaluate
        dataloader: DataLoader for evaluation
        device: Device to use
        save_predictions: Whether to return individual predictions
        
    Returns:
        Tuple of (metrics dict, predictions list)
    """
    model.eval()
    
    all_preds = []
    all_targets = []
    all_image_sizes = []
    predictions = []
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
        images = batch["images"].to(device)
        tasks = batch["tasks"]
        bboxes = batch["bboxes"].to(device)
        image_sizes = batch["image_sizes"]
        
        # Forward
        pred_bbox, _ = model(images, tasks, return_intermediates=False)
        
        # Collect
        all_preds.append(pred_bbox.cpu())
        all_targets.append(bboxes.cpu())
        all_image_sizes.extend(image_sizes)
        
        # Save individual predictions if requested
        if save_predictions:
            for i in range(len(tasks)):
                predictions.append({
                    "task": tasks[i],
                    "pred_bbox": pred_bbox[i].cpu().tolist(),
                    "gt_bbox": bboxes[i].cpu().tolist(),
                    "image_size": image_sizes[i],
                })
    
    # Aggregate
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # Compute metrics
    metrics = compute_metrics(all_preds, all_targets, all_image_sizes)
    
    return metrics, predictions


def main():
    parser = argparse.ArgumentParser(description="Evaluate ScreenSpot TRM model")
    
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--data-path", type=str,
                        default="dataset/screenspot_training.parquet",
                        help="Path to evaluation data")
    parser.add_argument("--split", type=str, default="val",
                        choices=["train", "val"],
                        help="Dataset split to evaluate")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device to use (auto, cuda, mps, cpu)")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save predictions JSON")
    parser.add_argument("--save-predictions", action="store_true",
                        help="Save individual predictions")
    
    args = parser.parse_args()
    
    device = get_device(args.device)
    print(f"Using device: {device}")
    
    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Create model
    model_config_dict = checkpoint.get("model_config", {})
    model_config = ModelConfig(**model_config_dict)
    
    model = ScreenBBoxTRMModel(config=model_config, device=str(device))
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded from epoch {checkpoint.get('epoch', 'unknown')}")
    
    # Create dataset
    dataset = ScreenSpotDataset(
        args.data_path,
        model.preprocess,
        split=args.split,
        val_split=0.1,
        seed=42,
    )
    print(f"Evaluating on {len(dataset)} samples ({args.split} split)")
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    
    # Evaluate
    metrics, predictions = evaluate(
        model, dataloader, device,
        save_predictions=args.save_predictions
    )
    
    # Print results
    print("\n" + "="*50)
    print("Evaluation Results")
    print("="*50)
    for k, v in sorted(metrics.items()):
        print(f"  {k}: {v:.4f}")
    
    # Save predictions if requested
    if args.output and args.save_predictions:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w") as f:
            json.dump({
                "metrics": metrics,
                "predictions": predictions,
            }, f, indent=2)
        print(f"\nPredictions saved to: {output_path}")


if __name__ == "__main__":
    main()

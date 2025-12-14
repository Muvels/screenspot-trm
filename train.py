"""
Training Script for UI Grounding with Enhanced TRM

Supports:
- Dual-state TRM architecture (z_H, z_L)
- Configurable H_cycles and L_cycles
- Optional Adaptive Computation Time (ACT)
- Supervised + RL training phases
"""

import argparse
import torch
import logging
from torch.utils.data import DataLoader
from transformers import AutoProcessor

from data_loader.screenspot import ScreenspotDataset, RowGroupBatchSampler
from models.agent import InfoMaxAgent
from training.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description="Train UI Grounding TRM")
    
    # Data arguments
    parser.add_argument("--data_path", type=str, default="dataset/screenspot_training.parquet", 
                        help="Path to parquet dataset")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0, 
                        help="Number of workers for data loading")
    parser.add_argument("--max_samples", type=int, default=None, 
                        help="Limit dataset size for debugging")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate for TRM and heads")
    parser.add_argument("--backbone_lr", type=float, default=None,
                        help="Learning rate for vision backbone (default: lr/10 when unfreezing)")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "sample"], 
                        help="train or sample mode")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, default="google/siglip-so400m-patch14-384",
                        help="Vision-language model to use")
    parser.add_argument("--hidden_size", type=int, default=512,
                        help="Hidden dimension for TRM")
    
    # TRM Architecture arguments
    parser.add_argument("--trm_layers", type=int, default=2,
                        help="Number of layers in L_level")
    parser.add_argument("--H_cycles", type=int, default=3,
                        help="Number of high-level reasoning cycles (outer loop)")
    parser.add_argument("--L_cycles", type=int, default=6,
                        help="Number of low-level refinement cycles (inner loop)")
    
    # Adaptive Computation Time (ACT) arguments
    parser.add_argument("--use_act", action="store_true",
                        help="Enable adaptive halting (ACT)")
    parser.add_argument("--max_steps", type=int, default=10,
                        help="Maximum steps for ACT")
    parser.add_argument("--act_loss_weight", type=float, default=0.5,
                        help="Weight for ACT loss")
    
    # WandB & Logging
    parser.add_argument("--use_wandb", action="store_true", 
                        help="Enable W&B logging")
    parser.add_argument("--project_name", type=str, default="screenspot-trm",
                        help="W&B project name")
    parser.add_argument("--run_name", type=str, default=None,
                        help="W&B run name")
    
    # Misc arguments
    parser.add_argument("--sample_interval", type=int, default=100, 
                        help="Steps between sampling")
    parser.add_argument("--num_samples", type=int, default=5, 
                        help="Number of samples to generate")
    parser.add_argument("--resume_from", type=str, default=None, 
                        help="Path to checkpoint to resume from")
    parser.add_argument("--unfreeze_backbone", action="store_true", 
                        help="Unfreeze vision encoder")
    parser.add_argument("--skip_rl", action="store_true",
                        help="Skip RL training phase")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Log configuration
    logger.info("=" * 60)
    logger.info("UI Grounding TRM Training")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model_name}")
    logger.info(f"TRM Config: H_cycles={args.H_cycles}, L_cycles={args.L_cycles}, layers={args.trm_layers}")
    logger.info(f"ACT: {'Enabled' if args.use_act else 'Disabled'}")
    if args.use_act:
        logger.info(f"  Max steps: {args.max_steps}, Loss weight: {args.act_loss_weight}")
    logger.info("=" * 60)
    
    # =========================================================================
    # 1. Setup Data
    # =========================================================================
    logger.info("Initializing Dataset...")
    
    processor = AutoProcessor.from_pretrained(args.model_name)
    
    def transform(image):
        inputs = processor(images=image, return_tensors="pt")
        return inputs['pixel_values'].squeeze(0)
    
    def collate_fn(batch):
        """Custom collate function that handles None values for input_ids/attention_mask."""
        # Stack tensors
        pixel_values = torch.stack([item["pixel_values"] for item in batch])
        ground_truth_bbox = torch.stack([item["ground_truth_bbox"] for item in batch])
        original_size = torch.stack([item["original_size"] for item in batch])
        
        # Handle instructions (list of strings)
        instructions = [item["instruction"] for item in batch]
        
        # Handle optional input_ids/attention_mask (might be None if not pre-tokenized)
        input_ids = batch[0].get("input_ids")
        if input_ids is not None:
            input_ids = torch.stack([item["input_ids"] for item in batch])
            attention_mask = torch.stack([item["attention_mask"] for item in batch])
        else:
            input_ids = None
            attention_mask = None
        
        return {
            "pixel_values": pixel_values,
            "instruction": instructions,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "ground_truth_bbox": ground_truth_bbox,
            "original_size": original_size
        }
    
    try:
        dataset = ScreenspotDataset(args.data_path, transform=transform)
    except FileNotFoundError:
        logger.error(f"Dataset not found at {args.data_path}. Please check path.")
        return

    # Get row group offsets for efficient batch sampling
    row_group_offsets = dataset.get_row_group_offsets()
    num_row_groups = len(row_group_offsets)
    
    # Split by row groups (not individual samples) to maintain locality
    # 90% train, 10% val
    train_rg_count = int(0.9 * num_row_groups)
    
    # Shuffle row groups for split
    import random
    rg_indices = list(range(num_row_groups))
    random.shuffle(rg_indices)
    
    train_rg_indices = set(rg_indices[:train_rg_count])
    val_rg_indices = set(rg_indices[train_rg_count:])
    
    train_offsets = [row_group_offsets[i] for i in train_rg_indices]
    val_offsets = [row_group_offsets[i] for i in val_rg_indices]
    
    train_samples = sum(num_rows for _, num_rows in train_offsets)
    val_samples = sum(num_rows for _, num_rows in val_offsets)
    
    # Limit dataset if max_samples is set
    if args.max_samples and args.max_samples < train_samples:
        logger.info(f"Limiting to ~{args.max_samples} samples for debugging.")
        # Limit by reducing row groups
        limited_offsets = []
        total = 0
        for offset, num_rows in train_offsets:
            if total >= args.max_samples:
                break
            limited_offsets.append((offset, num_rows))
            total += num_rows
        train_offsets = limited_offsets
        train_samples = sum(num_rows for _, num_rows in train_offsets)
    
    logger.info(f"Dataset: {train_samples:,} train, {val_samples:,} val samples")
    logger.info(f"Row groups: {len(train_offsets)} train, {len(val_offsets)} val")
    
    # Create batch samplers for efficient row group-aware loading
    train_batch_sampler = RowGroupBatchSampler(
        train_offsets, 
        batch_size=args.batch_size, 
        shuffle=True
    )
    val_batch_sampler = RowGroupBatchSampler(
        val_offsets, 
        batch_size=args.batch_size, 
        shuffle=False
    )
    
    # DataLoaders with batch samplers (no shuffle - sampler handles it)
    train_loader = DataLoader(
        dataset,  # Use full dataset, sampler selects indices
        batch_sampler=train_batch_sampler,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    ) 
    val_loader = DataLoader(
        dataset,
        batch_sampler=val_batch_sampler,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    # =========================================================================
    # 2. Setup Agent
    # =========================================================================
    logger.info("Initializing Agent...")
    
    agent = InfoMaxAgent(
        vision_text_model=args.model_name,
        hidden_size=args.hidden_size,
        trm_layers=args.trm_layers,
        H_cycles=args.H_cycles,
        L_cycles=args.L_cycles,
        use_act=args.use_act,
        max_steps=args.max_steps,
        freeze_backbone=not args.unfreeze_backbone
    )
    
    # Log architecture info
    total_depth = args.H_cycles * (args.L_cycles + 1)  # L_cycles per H + 1 H update
    logger.info(f"Agent initialized:")
    logger.info(f"  - Effective recursion depth: {total_depth} blocks")
    logger.info(f"  - Output steps: {args.H_cycles}")
    logger.info(f"  - ACT enabled: {args.use_act}")
    
    # Count parameters
    total_params = sum(p.numel() for p in agent.parameters())
    trainable_params = sum(p.numel() for p in agent.parameters() if p.requires_grad)
    logger.info(f"  - Total params: {total_params:,}")
    logger.info(f"  - Trainable params: {trainable_params:,}")
    
    if args.resume_from:
        logger.info(f"Resuming from checkpoint: {args.resume_from}")
        try:
            checkpoint = torch.load(args.resume_from, map_location="cpu")
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                agent.load_state_dict(checkpoint['model_state_dict'])
            else:
                agent.load_state_dict(checkpoint)
            logger.info("Checkpoint loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return
    
    # =========================================================================
    # 3. Setup Trainer
    # =========================================================================
    # Determine backbone LR (use lr/10 by default when unfreezing)
    backbone_lr = args.backbone_lr
    if args.unfreeze_backbone and backbone_lr is None:
        backbone_lr = args.lr / 10
        logger.info(f"Using default backbone_lr={backbone_lr:.2e} (lr/10)")
    
    trainer = Trainer(
        agent, 
        train_loader, 
        val_loader, 
        lr=args.lr, 
        backbone_lr=backbone_lr,
        num_epochs=args.epochs, 
        use_wandb=args.use_wandb,
        act_loss_weight=args.act_loss_weight
    )
    
    if args.mode == "train":
        logger.info("Starting Training...")
        
        if args.use_wandb:
            import wandb
            
            # Create run name if not provided
            run_name = args.run_name
            if run_name is None:
                act_suffix = "_act" if args.use_act else ""
                run_name = f"trm_H{args.H_cycles}_L{args.L_cycles}{act_suffix}"
            
            wandb.init(
                project=args.project_name, 
                name=run_name,
                config=vars(args)
            )
            
        for epoch in range(args.epochs):
            logger.info(f"\n{'='*60}")
            logger.info(f"Epoch {epoch + 1}/{args.epochs}")
            logger.info(f"{'='*60}")
            
            # 1. Supervised Phase
            logger.info("Phase 1: Supervised Training")
            trainer.train_supervised_epoch(epoch)
            
            # 2. RL Phase (optional)
            if not args.skip_rl:
                logger.info("Phase 2: RL Training")
                trainer.train_rl_epoch(epoch)
            
            # 3. Validation
            logger.info("Phase 3: Validation")
            trainer.validate_epoch(epoch)
            
            # 4. Checkpoint
            checkpoint_name = f"checkpoint_ep{epoch}"
            if args.use_act:
                checkpoint_name += "_act"
            checkpoint_name += ".pt"
            
            trainer.save_checkpoint(checkpoint_name)
            logger.info(f"Saved checkpoint: {checkpoint_name}")
            
        if args.use_wandb:
            wandb.finish()
            
    elif args.mode == "sample":
        logger.info(f"Sampling {args.num_samples} predictions...")
        agent.eval()
        device = trainer.device
        
        examples = []
        for batch in val_loader:
            pixel_values = batch["pixel_values"].to(device)
            instructions = batch["instruction"]
            gt_bbox = batch["ground_truth_bbox"]
            
            # Tokenize
            text_inputs = processor(
                text=list(instructions), 
                return_tensors="pt", 
                padding=True, 
                truncation=True
            ).to(device)
            
            with torch.no_grad():
                pred_bbox, val_pred, _, halting_info = agent(
                    pixel_values, 
                    text_inputs.input_ids, 
                    text_inputs.attention_mask
                )
            
            for i in range(len(instructions)):
                if len(examples) >= args.num_samples:
                    break
                    
                example = {
                    "instruction": instructions[i],
                    "pred_all_steps": pred_bbox[i].cpu().tolist(),
                    "pred_final": pred_bbox[i, -1].cpu().tolist(),
                    "gt": gt_bbox[i].tolist(),
                    "value": val_pred[i, -1].item()
                }
                
                if halting_info is not None:
                    example["steps_taken"] = halting_info.steps_taken[i].item()
                    
                examples.append(example)
            
            if len(examples) >= args.num_samples:
                break
        
        for i, ex in enumerate(examples):
            logger.info(f"\nSample {i+1}:")
            logger.info(f"  Task: {ex['instruction']}")
            logger.info(f"  Pred (final): {ex['pred_final']}")
            logger.info(f"  GT:   {ex['gt']}")
            logger.info(f"  Val:  {ex['value']:.4f}")
            if 'steps_taken' in ex:
                logger.info(f"  Steps: {ex['steps_taken']}")


if __name__ == "__main__":
    main()

import argparse
import torch
import logging
from torch.utils.data import DataLoader
from transformers import CLIPProcessor

from data_loader.screenspot import ScreenspotDataset
from models.agent import InfoMaxAgent
from training.trainer import Trainer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="dataset/screenspot_training.parquet", help="Path to parquet dataset")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--mode", type=str, default="train", choices=["train", "sample"], help="train or sample mode")
    parser.add_argument("--model_name", type=str, default="google/siglip-so400m-patch14-384")
    
    # WandB & Sampling flags
    parser.add_argument("--use_wandb", action="store_true", help="Enable W&B logging")
    parser.add_argument("--sample_interval", type=int, default=100, help="Steps between sampling")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples to generate")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit dataset size for debugging")
    parser.add_argument("--resume_from", type=str, default=None, help="Path to checkpoint to resume from")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # 1. Setup Data
    logger.info("Initializing Dataset...")
    
    # Switch to AutoProcessor
    from transformers import AutoProcessor
    try:
        processor = AutoProcessor.from_pretrained(args.model_name, trust_remote_code=True)
    except Exception as e:
        logger.warning(f"Could not load AutoProcessor for {args.model_name}, falling back to CLIPProcessor.")
        processor = CLIPProcessor.from_pretrained(args.model_name)

    def transform(image):
        # Handle Qwen-VL vs CLIP
        # Qwen-VL processor returns 'pixel_values' and 'image_grid_thw'
        try:
            inputs = processor(images=image, return_tensors="pt")
            # Return dict directly to be handled in collate
            return {k: v.squeeze(0) for k, v in inputs.items()} 
        except Exception:
            # Fallback
            return processor(images=image, return_tensors="pt")['pixel_values'].squeeze(0)
    
    try:
        dataset = ScreenspotDataset(args.data_path, transform=transform)
    except FileNotFoundError:
        logger.error(f"Dataset not found at {args.data_path}. Please check path.")
        return

    # Limit dataset if max_samples is set
    if args.max_samples and args.max_samples < len(dataset):
        logger.info(f"Limiting dataset to {args.max_samples} samples for debugging.")
        indices = torch.arange(args.max_samples)
        dataset = torch.utils.data.Subset(dataset, indices)
    # Split (Mock split for now)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    def qwen_collate_fn(batch):
        # Custom collate to handle Qwen specific fields if present
        # Batch is list of dicts
        
        # Check if first sample has dict as pixel_values (our wrapper logic might have put it there? 
        # No, ScreenspotDataset puts the return of transform into 'pixel_values' key)
        
        pixel_val_sample = batch[0]["pixel_values"]
        is_qwen_dict = isinstance(pixel_val_sample, dict) and "image_grid_thw" in pixel_val_sample
        
        collated = {}
        
        # Handle pixel_values
        if is_qwen_dict:
            # Qwen Batching: 
            # pixel_values -> Concatenate all (Sum_T, D)
            # image_grid_thw -> Concatenate (B, 3)
            pvs = [b["pixel_values"]["pixel_values"] for b in batch]
            grids = [b["pixel_values"]["image_grid_thw"] for b in batch]
            
            collated["pixel_values"] = torch.cat(pvs, dim=0)
            collated["image_grid_thw"] = torch.cat(grids, dim=0) # Pass as kwarg
        else:
            # Standard stacking
            collated["pixel_values"] = torch.stack([b["pixel_values"] for b in batch])
            
        # Handle others
        # Instruction / input_ids (Padding needed?)
        # For simplicity, assume standard collate for others or manual padding
        # Let's use default collate for simple fields, but input_ids might be variable length?
        # Dataset currently returns (L,) tensor for input_ids.
        
        if batch[0]["input_ids"] is not None:
             # Pad sequences
             input_ids = [b["input_ids"] for b in batch]
             attention_mask = [b["attention_mask"] for b in batch]
             collated["input_ids"] = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=processor.tokenizer.pad_token_id if hasattr(processor, "tokenizer") else 0)
             collated["attention_mask"] = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
        else:
            # Raw text instructions
            collated["instruction"] = [b["instruction"] for b in batch]
            
        collated["ground_truth_bbox"] = torch.stack([b["ground_truth_bbox"] for b in batch])
        
        return collated

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=qwen_collate_fn) 
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=qwen_collate_fn)
    
    # 2. Setup Agent
    logger.info("Initializing Agent...")
    logger.info("Initializing Agent...")
    agent = InfoMaxAgent(vision_text_model=args.model_name)
    
    if args.resume_from:
        logger.info(f"Resuming from checkpoint: {args.resume_from}")
        try:
            state_dict = torch.load(args.resume_from, map_location="cpu") # Load to CPU first
            agent.load_state_dict(state_dict)
            logger.info("Checkpoint loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return
    
    # 3. Setup Trainer
    trainer = Trainer(agent, train_loader, val_loader, lr=args.lr, use_wandb=args.use_wandb)
    
    if args.mode == "train":
        logger.info("Starting Training...")
        if args.use_wandb:
            import wandb
            wandb.init(project="screenspot-trm", config=vars(args))
            
        for epoch in range(args.epochs):
            # 1. Supervised Phase (Warmup)
            trainer.train_supervised_epoch(epoch)
            # 2. RL Phase (Continual)
            trainer.train_rl_epoch(epoch)
            
            trainer.save_checkpoint(f"checkpoint_ep{epoch}.pt")
            
    elif args.mode == "sample":
        logger.info(f"Sampling {args.num_samples} predictions...")
        agent.eval()
        device = trainer.device
        
        # Take a batch from validation
        examples = []
        for batch in val_loader:
            pixel_values = batch["pixel_values"].to(device)
            instructions = batch["instruction"]
            gt_bbox = batch["ground_truth_bbox"]
            
            # Tokenize
            text_inputs = processor(text=list(instructions), return_tensors="pt", padding=True, truncation=True).to(device)
            
            with torch.no_grad():
                pred_bbox, val_pred, _ = agent(pixel_values, text_inputs.input_ids, text_inputs.attention_mask)
            
            for i in range(len(instructions)):
                if len(examples) >= args.num_samples:
                    break
                examples.append({
                    "instruction": instructions[i],
                    "pred": pred_bbox[i].cpu().tolist(),
                    "gt": gt_bbox[i].tolist(),
                    "value": val_pred[i].item()
                })
            
            if len(examples) >= args.num_samples:
                break
        
        for i, ex in enumerate(examples):
            logger.info(f"Sample {i+1}:")
            logger.info(f"  Task: {ex['instruction']}")
            logger.info(f"  Pred: {ex['pred']}")
            logger.info(f"  GT:   {ex['gt']}")
            logger.info(f"  Val:  {ex['value']}")

if __name__ == "__main__":
    main()

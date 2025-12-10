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
    parser.add_argument("--model_name", type=str, default="openai/clip-vit-base-patch32")
    
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
    # Using CLIPProcessor for transform if we want consistent image norm, 
    # but Dataset currently uses default ToTensor if no transform passed.
    # To be precise, we should use CLIP's image processor.
    processor = CLIPProcessor.from_pretrained(args.model_name)
    def transform(image):
        # CLIP inputs: pixel_values
        inputs = processor(images=image, return_tensors="pt")
        return inputs['pixel_values'].squeeze(0) # (C, H, W)
    
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
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0) # workers=0 for simplicity
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    
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

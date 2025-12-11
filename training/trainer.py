import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
import tqdm
import logging
from transformers import AutoProcessor
from typing import Optional, Dict

from models.agent import InfoMaxAgent
from training.rewards import compute_iou, compute_reward, compute_giou

class Trainer:
    """
    Handles training logic:
    - Supervised Pretraining (Regression)
    - RL Fine-tuning (PPO-style or simple Policy Gradient)
    """
    def __init__(self, 
                 agent: InfoMaxAgent, 
                 train_loader: DataLoader, 
                 val_loader: Optional[DataLoader] = None,
                 lr: float = 1e-4,
                 device: str = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"),
                 use_wandb: bool = False):
                 
        self.agent = agent.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.use_wandb = use_wandb
        
        self.optimizer = optim.AdamW(self.agent.parameters(), lr=lr)
        
        # Scheduler: Use length of loader * default epochs (e.g. 1)
        # Better: T_max = steps per epoch. We step scheduler every batch.
        # If we want it to reset every epoch, T_max = len(train_loader).
        steps_per_epoch = len(train_loader) if train_loader is not None else 1000
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=steps_per_epoch)
        
        self.step = 0
        
    def train_supervised_epoch(self, epoch: int):
        """
        Train one epoch using Supervised Regression (MSE/L1 + IoU Loss).
        """
        self.agent.train()
        pbar = tqdm.tqdm(self.train_loader, desc=f"Epoch {epoch} (Sup)")
        
        for batch in pbar:
            self.optimizer.zero_grad()
            
            # Prepare data
            pixel_values = batch["pixel_values"].to(self.device)
            gt_bbox = batch["ground_truth_bbox"].to(self.device)
            
            # Check for pre-tokenized data
            if batch.get("input_ids") is not None and batch["input_ids"][0] is not None:
                # Assuming collation handled stacking, but Parquet list -> Tensor in Dataset might mean we have (B, L)
                # If dataset returned valid tensors, DataLoader stacks them.
                input_ids = batch["input_ids"].to(self.device)
                attn_mask = batch["attention_mask"].to(self.device)
            else:
                # Fallback: Tokenize on fly (CACHE THE PROCESSOR!)
                if not hasattr(self, "_processor"):
                    self._processor = AutoProcessor.from_pretrained(self.agent.encoder.model_name)
                    
                instructions = batch["instruction"]
                text_inputs = self._processor(text=list(instructions), return_tensors="pt", padding=True, truncation=True).to(self.device)
                input_ids = text_inputs.input_ids
                attn_mask = text_inputs.attention_mask
            
            # Forward
            pred_bbox, value_pred, _ = self.agent(pixel_values, input_ids, attn_mask)
            
            # Loss: Deep Supervision
            # pred_bbox shape: (B, Steps, 4)
            # gt_bbox shape: (B, 4) -> Expand to (B, Steps, 4)
            Steps = pred_bbox.shape[1]
            gt_bbox_expanded = gt_bbox.unsqueeze(1).expand(-1, Steps, -1)
            
            # Compute loss for all steps
            # L1
            loss_l1 = nn.functional.l1_loss(pred_bbox, gt_bbox_expanded)
            
            # IoU / GIoU
            # Flatten to (B*Steps, 4) for reward func
            pred_flat = pred_bbox.view(-1, 4)
            gt_flat = gt_bbox_expanded.reshape(-1, 4)
            
            iou_flat = compute_iou(pred_flat, gt_flat)
            giou_flat = compute_giou(pred_flat, gt_flat)
            
            loss_iou = 1.0 - giou_flat.mean()
            
            loss = loss_l1 + loss_iou
            
            loss.backward()
            
            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(self.agent.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            self.step += 1
            
            # Logging (Use Final Step for metrics)
            final_iou = iou_flat.view(-1, Steps)[:, -1].mean()
            
            metrics = {
                "train/loss": loss.item(),
                "train/iou": final_iou.item(),
                "train/l1": loss_l1.item()
            }
            if self.use_wandb:
                wandb.log(metrics)
            
            pbar.set_postfix(loss=loss.item(), iou=final_iou.item())

    def validate_epoch(self, epoch: int):
        """
        Run validation loop to detect overfitting.
        """
        self.agent.eval()
        if self.val_loader is None:
            return
            
        total_iou = 0
        total_loss = 0
        steps = 0
        
        pbar = tqdm.tqdm(self.val_loader, desc=f"Epoch {epoch} (Val)")
        
        with torch.no_grad():
            for batch in pbar:
                pixel_values = batch["pixel_values"].to(self.device)
                gt_bbox = batch["ground_truth_bbox"].to(self.device)
                
                if batch.get("input_ids") is not None and batch["input_ids"][0] is not None:
                    input_ids = batch["input_ids"].to(self.device)
                    attn_mask = batch["attention_mask"].to(self.device)
                else:
                    if not hasattr(self, "_processor"):
                         self._processor = AutoProcessor.from_pretrained(self.agent.encoder.model_name)
                    instructions = batch["instruction"]
                    text_inputs = self._processor(text=list(instructions), return_tensors="pt", padding=True, truncation=True).to(self.device)
                    input_ids = text_inputs.input_ids
                    attn_mask = text_inputs.attention_mask
                
                # Forward
                pred_bbox, _, _ = self.agent(pixel_values, input_ids, attn_mask)
                
                # Metrics (Deep Supervision: Use Final Step)
                pred_final = pred_bbox[:, -1, :]
                
                # Loss (Proxy)
                loss_l1 = nn.functional.l1_loss(pred_final, gt_bbox)
                giou = compute_giou(pred_final, gt_bbox)
                loss_val = loss_l1 + (1.0 - giou.mean())
                
                # IoU
                iou = compute_iou(pred_final, gt_bbox)
                
                total_loss += loss_val.item()
                total_iou += iou.mean().item()
                steps += 1
                
                pbar.set_postfix(val_loss=loss_val.item(), val_iou=iou.mean().item())
        
        avg_loss = total_loss / steps if steps > 0 else 0
        avg_iou = total_iou / steps if steps > 0 else 0
        
        if self.use_wandb:
            wandb.log({
                "val/loss": avg_loss,
                "val/iou": avg_iou
            })
        
        logging.info(f"Validation Epoch {epoch}: Loss={avg_loss:.4f}, IoU={avg_iou:.4f}")
            
    def train_rl_epoch(self, epoch: int):
        """
        Train using RL (e.g. REINFORCE baseline or PPO).
        For simplicity, implementing A2C-style update on collected batch (Online).
        """
        self.agent.train()
        pbar = tqdm.tqdm(self.train_loader, desc=f"Epoch {epoch} (RL)")
        
        for batch in pbar:
            self.optimizer.zero_grad()
            
            # 1. Forward (Sampling Action)
            # Same tokenization logic
            # 1. Forward (Sampling Action)
            pixel_values = batch["pixel_values"].to(self.device)
            gt_bbox = batch["ground_truth_bbox"].to(self.device)
            
            # Check for pre-tokenized data
            if batch.get("input_ids") is not None and batch["input_ids"][0] is not None:
                input_ids = batch["input_ids"].to(self.device)
                attn_mask = batch["attention_mask"].to(self.device)
            else:
                 if not hasattr(self, "_processor"):
                    self._processor = AutoProcessor.from_pretrained(self.agent.encoder.model_name)
                    
                 instructions = batch["instruction"]
                 text_inputs = self._processor(text=list(instructions), return_tensors="pt", padding=True, truncation=True).to(self.device)
                 input_ids = text_inputs.input_ids
                 attn_mask = text_inputs.attention_mask
            gt_bbox = batch["ground_truth_bbox"].to(self.device)

            pred_bbox, value_pred, _ = self.agent(pixel_values, input_ids, attn_mask)
            
            # Deep Supervision for RL?
            # Usually we only care about the final action for RL, or we treat each step as a sub-action.
            # To keep it simple: Use FINAL step for RL update.
            # pred_bbox: (B, Steps, 4)
            pred_final = pred_bbox[:, -1, :]
            value_final = value_pred[:, -1, :]
            
            # 2. Compute Reward
            # In online RL we would 'act' and get external reward.
            # Here we simulate valid reward from ground truth.
            # If we treat pred_bbox as deterministic action:
            reward = compute_reward(pred_final, gt_bbox).detach() # (B, )
            
            # 3. Compute Loss (Actor-Critic)
            # Advantage = Reward - Value
            advantage = reward - value_final.squeeze(-1).detach()
            
            # Value Loss
            value_loss = nn.functional.mse_loss(value_final.squeeze(-1), reward)
            
            # Optimizing GIoU directly as a proxy for "Policy Gradient" on deterministic policy
            # This is robust because GIoU provides signal even if IoU=0
            giou = compute_giou(pred_final, gt_bbox)
            loss_policy = -giou.mean()
            loss = loss_policy + 0.5 * value_loss
            
            loss.backward()
            
            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(self.agent.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            self.scheduler.step()
            
            metrics = {
                "train/rl_loss": loss.item(),
                "train/reward": reward.mean().item()
            }
            if self.use_wandb:
                wandb.log(metrics)
            pbar.set_postfix(loss=loss.item(), reward=reward.mean().item())

    def save_checkpoint(self, path: str):
        torch.save(self.agent.state_dict(), path)

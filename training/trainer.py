"""
Enhanced Trainer for UI Grounding with TRM

Supports:
- Supervised pretraining with deep supervision
- RL fine-tuning (Actor-Critic style)
- Adaptive Computation Time (ACT) loss
- Comprehensive logging including halting statistics
"""

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
from models.trm import HaltingInfo
from training.rewards import compute_iou, compute_reward, compute_giou


class Trainer:
    """
    Enhanced trainer for UI grounding with TRM.
    
    Handles:
    - Supervised Pretraining (Regression with deep supervision)
    - RL Fine-tuning (PPO-style or simple Policy Gradient)
    - ACT Loss (Q-learning based halting)
    - Differential learning rates for backbone vs heads
    """
    
    def __init__(
        self, 
        agent: InfoMaxAgent, 
        train_loader: DataLoader, 
        val_loader: Optional[DataLoader] = None,
        lr: float = 1e-4,
        backbone_lr: Optional[float] = None,
        num_epochs: int = 1,
        device: str = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"),
        use_wandb: bool = False,
        act_loss_weight: float = 0.5
    ):
        self.agent = agent.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.use_wandb = use_wandb
        self.act_loss_weight = act_loss_weight
        
        # Check if ACT is enabled
        self.use_act = getattr(agent.core, 'use_act', False)
        
        # Build optimizer with differential learning rates
        self.optimizer = self._build_optimizer(lr, backbone_lr)
        
        # Scheduler: Linear warm-up + Cosine decay
        steps_per_epoch = len(train_loader) if train_loader is not None else 1000
        self.total_steps = steps_per_epoch * num_epochs
        self.warmup_steps = min(1000, self.total_steps // 10)  # 1000 steps or 10% of training
        
        # Use cosine schedule after warmup
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=self.total_steps - self.warmup_steps
        )
        
        # Store base LRs for warm-up calculation
        self.base_lrs = [group['lr'] for group in self.optimizer.param_groups]
        
        logging.info(f"Scheduler: {self.warmup_steps} warmup steps, then cosine decay")
        
        self.step = 0
    
    def _build_optimizer(self, lr: float, backbone_lr: Optional[float] = None) -> optim.Optimizer:
        """
        Build optimizer with optional differential learning rates.
        
        If backbone_lr is provided:
        - Vision backbone (encoder.backbone): uses backbone_lr
        - Everything else (TRM, heads, projections): uses lr
        
        If backbone_lr is None:
        - All parameters use lr (single param group)
        """
        if backbone_lr is None:
            # Simple case: all params same LR
            return optim.AdamW(self.agent.parameters(), lr=lr)
        
        # Differential LR: separate backbone from the rest
        backbone_params = []
        head_params = []
        
        for name, param in self.agent.named_parameters():
            if not param.requires_grad:
                continue
            
            # Check if this is a backbone parameter
            # encoder.backbone.* are the pretrained weights
            # encoder.vis_proj/txt_proj are new projection layers (use higher lr)
            if "encoder.backbone" in name:
                backbone_params.append(param)
            else:
                head_params.append(param)
        
        param_groups = [
            {"params": backbone_params, "lr": backbone_lr, "name": "backbone"},
            {"params": head_params, "lr": lr, "name": "heads"}
        ]
        
        # Log parameter counts
        n_backbone = sum(p.numel() for p in backbone_params)
        n_heads = sum(p.numel() for p in head_params)
        logging.info(f"Differential LR setup:")
        logging.info(f"  - Backbone params: {n_backbone:,} @ lr={backbone_lr:.2e}")
        logging.info(f"  - Head params: {n_heads:,} @ lr={lr:.2e}")
        
        return optim.AdamW(param_groups)
        
    def compute_act_loss(
        self, 
        halting_info: HaltingInfo,
        final_iou: torch.Tensor,
        threshold: float = 0.5
    ) -> torch.Tensor:
        """
        Compute ACT loss for adaptive halting.
        
        The Q-halt head should learn:
        - Halt early if prediction is already good (IoU > threshold)
        - Continue if prediction needs refinement (IoU < threshold)
        
        Args:
            halting_info: Halting information from TRM
            final_iou: IoU at the final step (B,)
            threshold: IoU threshold for considering prediction "good"
            
        Returns:
            act_loss: Binary cross-entropy loss for halt decisions
        """
        # Target: Halt if IoU is good enough
        halt_target = (final_iou > threshold).float()
        
        # Use the last halt logit for loss
        # Shape: (B, Steps) -> (B,) last step
        last_halt_logits = halting_info.halt_logits[:, -1]
        
        # Binary cross-entropy loss
        act_loss = nn.functional.binary_cross_entropy_with_logits(
            last_halt_logits, 
            halt_target
        )
        
        return act_loss
        
    def train_supervised_epoch(self, epoch: int):
        """
        Train one epoch using Supervised Regression.
        
        Loss components:
        - L1 Loss: Coordinate regression
        - GIoU Loss: Bounding box quality
        - ACT Loss: Halting decision (if enabled)
        
        Uses deep supervision (loss on all TRM steps).
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
                input_ids = batch["input_ids"].to(self.device)
                attn_mask = batch["attention_mask"].to(self.device)
            else:
                # Fallback: Tokenize on fly
                if not hasattr(self, "_processor"):
                    self._processor = AutoProcessor.from_pretrained(self.agent.encoder.model_name)
                    
                instructions = batch["instruction"]
                text_inputs = self._processor(
                    text=list(instructions), 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True
                ).to(self.device)
                input_ids = text_inputs.input_ids
                # Some processors (like SigLIP) don't return attention_mask
                attn_mask = getattr(text_inputs, 'attention_mask', None)
                if attn_mask is None:
                    attn_mask = torch.ones_like(input_ids)
            
            # Forward pass (now returns 4 values including halting_info)
            pred_bbox, value_pred, _, halting_info = self.agent(
                pixel_values, input_ids, attn_mask
            )
            
            # =============================================================
            # Loss: Deep Supervision
            # =============================================================
            # pred_bbox shape: (B, Steps, 4)
            # gt_bbox shape: (B, 4) -> Expand to (B, Steps, 4)
            Steps = pred_bbox.shape[1]
            gt_bbox_expanded = gt_bbox.unsqueeze(1).expand(-1, Steps, -1)
            
            # L1 Loss
            loss_l1 = nn.functional.l1_loss(pred_bbox, gt_bbox_expanded)
            
            # GIoU Loss
            # Flatten to (B*Steps, 4) for reward func
            pred_flat = pred_bbox.view(-1, 4)
            gt_flat = gt_bbox_expanded.reshape(-1, 4)
            
            iou_flat = compute_iou(pred_flat, gt_flat)
            giou_flat = compute_giou(pred_flat, gt_flat)
            
            loss_giou = 1.0 - giou_flat.mean()
            
            # Total supervised loss
            loss = loss_l1 + loss_giou
            
            # =============================================================
            # ACT Loss (if enabled)
            # =============================================================
            act_loss = torch.tensor(0.0, device=self.device)
            avg_steps = 0.0
            
            if self.use_act and halting_info is not None:
                # Use final step IoU as signal for halting quality
                final_iou = iou_flat.view(-1, Steps)[:, -1]
                act_loss = self.compute_act_loss(halting_info, final_iou)
                loss = loss + self.act_loss_weight * act_loss
                avg_steps = halting_info.steps_taken.float().mean().item()
            
            loss.backward()
            
            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(self.agent.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Warmup LR schedule
            if self.step < self.warmup_steps:
                # Linear warmup: scale LR from 0 to base_lr
                warmup_factor = (self.step + 1) / self.warmup_steps
                for i, param_group in enumerate(self.optimizer.param_groups):
                    param_group['lr'] = self.base_lrs[i] * warmup_factor
            else:
                # After warmup, use cosine schedule
                self.scheduler.step()
            
            self.step += 1
            
            # Debug: Log first batch predictions to verify output distribution
            if self.step == 1:
                pred_sample = pred_bbox[0, -1].detach().cpu().numpy()  # First sample, final step
                gt_sample = gt_bbox[0].detach().cpu().numpy()
                logging.info(f"[DEBUG] First batch - Pred: {pred_sample}, GT: {gt_sample}")
            
            # =============================================================
            # Logging
            # =============================================================
            # Use Final Step for metrics
            final_iou = iou_flat.view(-1, Steps)[:, -1].mean()
            
            # Get current LR (from first param group)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            metrics = {
                "train/loss": loss.item(),
                "train/iou": final_iou.item(),
                "train/l1": loss_l1.item(),
                "train/giou_loss": loss_giou.item(),
                "train/lr": current_lr
            }
            
            if self.use_act:
                metrics["train/act_loss"] = act_loss.item()
                metrics["train/avg_steps"] = avg_steps
            
            if self.use_wandb:
                wandb.log(metrics)
            
            pbar.set_postfix(
                loss=loss.item(), 
                iou=final_iou.item(),
                steps=avg_steps if self.use_act else pred_bbox.shape[1]
            )

    def validate_epoch(self, epoch: int):
        """
        Run validation loop to detect overfitting.
        """
        self.agent.eval()
        if self.val_loader is None:
            return
            
        total_iou = 0
        total_loss = 0
        total_steps = 0
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
                    text_inputs = self._processor(
                        text=list(instructions), 
                        return_tensors="pt", 
                        padding=True, 
                        truncation=True
                    ).to(self.device)
                    input_ids = text_inputs.input_ids
                    attn_mask = getattr(text_inputs, 'attention_mask', None)
                    if attn_mask is None:
                        attn_mask = torch.ones_like(input_ids)
                
                # Forward
                pred_bbox, _, _, halting_info = self.agent(pixel_values, input_ids, attn_mask)
                
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
                
                if halting_info is not None:
                    total_steps += halting_info.steps_taken.float().mean().item()
                else:
                    total_steps += pred_bbox.shape[1]
                    
                steps += 1
                
                pbar.set_postfix(val_loss=loss_val.item(), val_iou=iou.mean().item())
        
        avg_loss = total_loss / steps if steps > 0 else 0
        avg_iou = total_iou / steps if steps > 0 else 0
        avg_steps_taken = total_steps / steps if steps > 0 else 0
        
        metrics = {
            "val/loss": avg_loss,
            "val/iou": avg_iou,
            "val/avg_steps": avg_steps_taken
        }
        
        if self.use_wandb:
            wandb.log(metrics)
        
        logging.info(
            f"Validation Epoch {epoch}: Loss={avg_loss:.4f}, IoU={avg_iou:.4f}, "
            f"AvgSteps={avg_steps_taken:.2f}"
        )
            
    def train_rl_epoch(self, epoch: int):
        """
        Train using Advantage-Weighted RL (Actor-Critic style).
        
        Improvements over basic RL:
        1. Advantage weighting: Focus on samples where model underperformed
        2. Uncertainty estimation: Value Head predicts error for confidence at inference
        3. L1 loss restored: Full coordinate-level signal
        
        Uses GIoU as reward signal for policy gradient.
        """
        self.agent.train()
        pbar = tqdm.tqdm(self.train_loader, desc=f"Epoch {epoch} (RL)")
        
        for batch in pbar:
            self.optimizer.zero_grad()
            
            # Prepare data
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
                text_inputs = self._processor(
                    text=list(instructions), 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True
                ).to(self.device)
                input_ids = text_inputs.input_ids
                attn_mask = getattr(text_inputs, 'attention_mask', None)
                if attn_mask is None:
                    attn_mask = torch.ones_like(input_ids)

            pred_bbox, value_pred, _, halting_info = self.agent(
                pixel_values, input_ids, attn_mask
            )
            
            # Use FINAL step for RL update
            pred_final = pred_bbox[:, -1, :]
            value_final = value_pred[:, -1, :].squeeze(-1)  # (B,)
            
            # =============================================================
            # Option C: L1 Loss (restored for coordinate-level signal)
            # =============================================================
            loss_l1 = nn.functional.l1_loss(pred_final, gt_bbox)
            
            # =============================================================
            # Compute metrics for advantage and value targets
            # =============================================================
            iou = compute_iou(pred_final, gt_bbox)  # (B,)
            giou = compute_giou(pred_final, gt_bbox)  # (B,)
            reward = compute_reward(pred_final, gt_bbox)  # IoU + bonus (B,)
            
            # =============================================================
            # Option B: Value Head predicts UNCERTAINTY (1 - IoU)
            # This makes it useful at inference for confidence estimation
            # =============================================================
            uncertainty_target = (1.0 - iou).detach()  # High = uncertain
            value_loss = nn.functional.mse_loss(value_final, uncertainty_target)
            
            # =============================================================
            # Option A: Advantage-Weighted Policy Loss
            # Samples where model underperformed get stronger gradients
            # =============================================================
            # Advantage = actual reward - expected (baseline from value)
            # Since value now predicts uncertainty: expected_reward ≈ 1 - uncertainty
            expected_reward = (1.0 - value_final).detach()
            advantage = (reward - expected_reward).detach()  # (B,)
            
            # Normalize advantage for stable training
            if advantage.std() > 1e-6:
                advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
            
            # Weighted GIoU loss: harder samples get stronger gradients
            # We use (1 + advantage) to ensure all samples contribute
            weights = torch.clamp(1.0 + advantage, min=0.1, max=3.0)  # (B,)
            loss_policy = -(weights * giou).mean()
            
            # =============================================================
            # Total Loss
            # =============================================================
            loss = loss_l1 + loss_policy + 0.5 * value_loss
            
            # ACT Loss for RL (if enabled)
            act_loss = torch.tensor(0.0, device=self.device)
            if self.use_act and halting_info is not None:
                # Encourage halting when prediction is confident (low uncertainty)
                should_halt = (iou > 0.5).float()  # High IoU = can halt
                last_halt = halting_info.halt_logits[:, -1]
                act_loss = nn.functional.binary_cross_entropy_with_logits(
                    last_halt, should_halt
                )
                loss = loss + self.act_loss_weight * act_loss
            
            loss.backward()
            
            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(self.agent.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            self.scheduler.step()
            
            # =============================================================
            # Logging
            # =============================================================
            metrics = {
                "train/rl_loss": loss.item(),
                "train/rl_l1": loss_l1.item(),
                "train/rl_policy": loss_policy.item(),
                "train/rl_value": value_loss.item(),
                "train/rl_iou": iou.mean().item(),
                "train/rl_advantage_mean": advantage.mean().item(),
                "train/rl_uncertainty": value_final.mean().item(),
                "train/reward": reward.mean().item(),
                "train/lr": self.scheduler.get_last_lr()[0]
            }
            
            if self.use_act and halting_info is not None:
                metrics["train/rl_act_loss"] = act_loss.item()
                metrics["train/rl_avg_steps"] = halting_info.steps_taken.float().mean().item()
            
            if self.use_wandb:
                wandb.log(metrics)
                
            pbar.set_postfix(
                loss=loss.item(), 
                iou=iou.mean().item(),
                adv=advantage.mean().item()
            )

    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.agent.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'step': self.step
        }
        torch.save(checkpoint, path)
        
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.agent.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.step = checkpoint['step']

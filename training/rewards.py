import torch

def compute_iou(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    """
    Compute IoU between two sets of boxes.
    Boxes are [x1, y1, x2, y2]
    
    Args:
        box1: (B, 4)
        box2: (B, 4)
    
    Returns:
        iou: (B, )
    """
    # Intersection
    x1 = torch.max(box1[:, 0], box2[:, 0])
    y1 = torch.max(box1[:, 1], box2[:, 1])
    x2 = torch.min(box1[:, 2], box2[:, 2])
    y2 = torch.min(box1[:, 3], box2[:, 3])
    
    intersection_area = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    
    box1_area = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    box2_area = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    
    union_area = box1_area + box2_area - intersection_area + 1e-6
    
    return intersection_area / union_area

def compute_reward(pred_box: torch.Tensor, gt_box: torch.Tensor, alpha: float = 2.0) -> torch.Tensor:
    """
    Reward function for RL.
    R = IoU + alpha * (Success Threshold?)
    For now, simple IoU based reward.
    
    Args:
        pred_box: (B, 4)
        gt_box: (B, 4)
        
    Returns:
        reward: (B, )
    """
    iou = compute_iou(pred_box, gt_box)
    
    # Optional: Bonus +1 if IoU > 0.5
    bonus = (iou > 0.5).float()
    
    return iou + bonus * alpha

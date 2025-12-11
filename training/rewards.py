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
    # Ensure coordinates are sorted (min, max) to handle inverted predictions
    b1_x1 = torch.min(box1[:, 0], box1[:, 2])
    b1_y1 = torch.min(box1[:, 1], box1[:, 3])
    b1_x2 = torch.max(box1[:, 0], box1[:, 2])
    b1_y2 = torch.max(box1[:, 1], box1[:, 3])
    
    b2_x1 = torch.min(box2[:, 0], box2[:, 2])
    b2_y1 = torch.min(box2[:, 1], box2[:, 3])
    b2_x2 = torch.max(box2[:, 0], box2[:, 2])
    b2_y2 = torch.max(box2[:, 1], box2[:, 3])
    
    # Intersection
    x1 = torch.max(b1_x1, b2_x1)
    y1 = torch.max(b1_y1, b2_y1)
    x2 = torch.min(b1_x2, b2_x2)
    y2 = torch.min(b1_y2, b2_y2)
    
    intersection_area = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    
    box1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    box2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    
    union_area = box1_area + box2_area - intersection_area + 1e-6
    
    return intersection_area / union_area

def compute_giou(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    """
    Generalized IoU: IoU - (C - U) / C
    where C is the smallest enclosing box.
    """
    # Ensure coordinates are sorted (min, max) to handle inverted predictions
    b1_x1 = torch.min(box1[:, 0], box1[:, 2])
    b1_y1 = torch.min(box1[:, 1], box1[:, 3])
    b1_x2 = torch.max(box1[:, 0], box1[:, 2])
    b1_y2 = torch.max(box1[:, 1], box1[:, 3])
    
    b2_x1 = torch.min(box2[:, 0], box2[:, 2])
    b2_y1 = torch.min(box2[:, 1], box2[:, 3])
    b2_x2 = torch.max(box2[:, 0], box2[:, 2])
    b2_y2 = torch.max(box2[:, 1], box2[:, 3])

    # Intersection
    x1 = torch.max(b1_x1, b2_x1)
    y1 = torch.max(b1_y1, b2_y1)
    x2 = torch.min(b1_x2, b2_x2)
    y2 = torch.min(b1_y2, b2_y2)
    
    intersection_area = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    
    box1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    box2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    
    union_area = box1_area + box2_area - intersection_area + 1e-6
    iou = intersection_area / union_area
    
    # Enclosing Box
    cX1 = torch.min(b1_x1, b2_x1)
    cY1 = torch.min(b1_y1, b2_y1)
    cX2 = torch.max(b1_x2, b2_x2)
    cY2 = torch.max(b1_y2, b2_y2)
    
    c_area = (cX2 - cX1) * (cY2 - cY1) + 1e-6
    
    return iou - (c_area - union_area) / c_area

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

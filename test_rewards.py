import torch
from training.rewards import compute_giou, compute_iou

def test_inverted_boxes():
    # Box 1: Inverted [0.9, 0.9, 0.1, 0.1]
    # Box 2: GT [0.0, 0.0, 0.1, 0.1]
    
    b1 = torch.tensor([[0.9, 0.9, 0.1, 0.1]])
    b2 = torch.tensor([[0.0, 0.0, 0.1, 0.1]])
    
    giou = compute_giou(b1, b2)
    print(f"GIoU: {giou.item()}")
    
    assert giou.item() <= 1.0, f"GIoU {giou.item()} should be <= 1.0"
    assert giou.item() >= -1.0, f"GIoU {giou.item()} should be >= -1.0"

if __name__ == "__main__":
    test_inverted_boxes()
    print("Test passed!")

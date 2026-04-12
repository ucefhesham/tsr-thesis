import torch
from src.models.evidential_module import EvidentialResNetModule

def test_edl():
    print("Initializing EvidentialResNetModule...")
    module = EvidentialResNetModule(num_classes=43)
    
    # Mock batch (Batch size 2, Channels 3, size 224x224)
    x = torch.randn(2, 3, 224, 224)
    y = torch.tensor([1, 15])
    
    print("Running Forward Pass...")
    outputs = module(x)
    
    # Check output keys
    for key in ["evidence", "alpha", "prob", "vacuity"]:
        assert key in outputs, f"Key {key} missing from outputs"
        print(f"  {key} shape: {outputs[key].shape}")
        
    # Check vacuity bounds
    assert torch.all(outputs["vacuity"] >= 0) and torch.all(outputs["vacuity"] <= 1), "Vacuity out of bounds [0, 1]"
    print(f"  Vacuity values: {outputs['vacuity'].flatten().tolist()}")

    print("Checking Loss Calculation...")
    y_one_hot = torch.nn.functional.one_hot(y, num_classes=43).float()
    loss = module.criterion(outputs["alpha"], y_one_hot, epoch=5)
    print(f"  Loss at epoch 5: {loss.item():.4f}")
    
    # Check annealing difference
    loss_0 = module.criterion(outputs["alpha"], y_one_hot, epoch=0)
    loss_10 = module.criterion(outputs["alpha"], y_one_hot, epoch=10)
    print(f"  Loss at epoch 0: {loss_0.item():.4f}")
    print(f"  Loss at epoch 10: {loss_10.item():.4f}")
    assert loss_10 > loss_0, "KL annealing not increasing loss penalty correctly"

    print("Smoke Test PASSED!")

if __name__ == "__main__":
    test_edl()

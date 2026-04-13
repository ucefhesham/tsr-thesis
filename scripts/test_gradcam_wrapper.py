"""Quick test: verify that the GradCAM wrapper produces a plain tensor with grad_fn."""
import torch
from src.models.evidential import EvidentialNetwork

net = EvidentialNetwork()

class GradCAMWrapper(torch.nn.Module):
    def __init__(self, evidential_net):
        super().__init__()
        self.evidential_net = evidential_net
    def forward(self, x):
        out = self.evidential_net(x)
        if isinstance(out, dict):
            return out["alpha"]
        return out

w = GradCAMWrapper(net)
x = torch.randn(1, 3, 224, 224, requires_grad=True)
out = w(x)
print(f"Wrapper output type: {type(out)}")
print(f"Shape: {out.shape}")
print(f"Has grad_fn: {out.grad_fn is not None}")

# Now test that Grad-CAM can actually use it
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

feat_ext = net.feature_extractor
target_layers = [feat_ext[-2]]  # Layer4

cam = GradCAM(model=w, target_layers=target_layers)
targets = [ClassifierOutputTarget(0)]
result = cam(input_tensor=x, targets=targets)
print(f"GradCAM output shape: {result.shape}")
print("SUCCESS: GradCAM works with the wrapper!")

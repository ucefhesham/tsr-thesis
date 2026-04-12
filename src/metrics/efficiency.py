import torch
import torch.nn as nn
from fvcore.nn import FlopCountAnalysis
import logging

# Set fvcore logger to warning to avoid verbose output during every eval
logging.getLogger("fvcore").setLevel(logging.WARNING)

def compute_model_flops(model: nn.Module, input_res: int = 224) -> float:
    """
    Computes the total GFLOPs for a single forward pass of the model.
    
    Args:
        model: The PyTorch model to profile.
        input_res: The spatial resolution of the input image (assumed square).
        
    Returns:
        float: Total GFLOPs (Gigaflopping-point operations).
    """
    # Move model to CPU for profiling to avoid CUDA overhead/errors in some environments
    device = next(model.parameters()).device
    model.eval()
    
    # Create a dummy input (Batch size 1)
    dummy_input = torch.randn(1, 3, input_res, input_res).to(device)
    
    try:
        # fvcore's FlopCountAnalysis is highly accurate for modern architectures like ConvNeXt
        flops = FlopCountAnalysis(model, dummy_input)
        total_flops = flops.total()
        gflops = total_flops / 1e9
        return gflops
    except Exception as e:
        print(f"Error computing GFLOPs: {e}")
        return 0.0

def calculate_tpe(delta_ece: float, delta_gflops: float) -> float:
    """
    Calculates Trust-Per-Efficiency (TPE) as defined in the thesis proposal.
    TPE = (Gain in Calibration) / (Cost in GFLOPs)
    
    Args:
        delta_ece: The reduction in Expected Calibration Error (ECE_base - ECE_method).
        delta_gflops: The additional GFLOPs required by the method.
        
    Returns:
        float: TPE score.
    """
    # Use small epsilon to avoid division by zero for methods with zero overhead
    return delta_ece / max(delta_gflops, 1e-6)

"""
Device utilities for Apple Silicon MPS backend.
Provides automatic device detection and fallback handling.
"""
import torch
import warnings
from typing import Literal

DeviceType = Literal["mps", "cuda", "cpu"]


def get_device(preferred: DeviceType = "mps") -> torch.device:
    """
    Get the best available device for computation.
    Priority: preferred > mps > cuda > cpu
    
    Args:
        preferred: Preferred device type
        
    Returns:
        torch.device: Best available device
    """
    if preferred == "mps" and torch.backends.mps.is_available():
        if torch.backends.mps.is_built():
            return torch.device("mps")
        else:
            warnings.warn("MPS available but not built. Falling back to CPU.")
    
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    
    # Fallback chain
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    
    return torch.device("cpu")


def to_device(tensor_or_module, device: torch.device = None):
    """
    Move tensor or module to device, handling MPS compatibility.
    
    Some operations aren't supported on MPS yet, this handles fallbacks.
    """
    if device is None:
        device = get_device()
    
    return tensor_or_module.to(device)


def mps_safe_operation(func):
    """
    Decorator for operations that might not be supported on MPS.
    Falls back to CPU if MPS operation fails.
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except RuntimeError as e:
            if "MPS" in str(e) or "mps" in str(e):
                warnings.warn(f"MPS operation failed, falling back to CPU: {e}")
                # Move args to CPU, run, then move back
                cpu_args = []
                for arg in args:
                    if isinstance(arg, torch.Tensor):
                        cpu_args.append(arg.cpu())
                    else:
                        cpu_args.append(arg)
                result = func(*cpu_args, **kwargs)
                if isinstance(result, torch.Tensor):
                    return result.to("mps")
                return result
            raise
    return wrapper


def check_mps_capabilities():
    """
    Check and print MPS capabilities for debugging.
    """
    info = {
        "mps_available": torch.backends.mps.is_available(),
        "mps_built": torch.backends.mps.is_built() if torch.backends.mps.is_available() else False,
        "pytorch_version": torch.__version__,
    }
    
    if info["mps_available"] and info["mps_built"]:
        # Test basic operations
        try:
            x = torch.randn(100, 100, device="mps")
            y = torch.matmul(x, x)
            info["matmul_support"] = True
        except:
            info["matmul_support"] = False
            
        try:
            conv = torch.nn.Conv2d(3, 64, 3).to("mps")
            x = torch.randn(1, 3, 64, 64, device="mps")
            _ = conv(x)
            info["conv2d_support"] = True
        except:
            info["conv2d_support"] = False
            
        try:
            gru = torch.nn.GRU(64, 128).to("mps")
            x = torch.randn(10, 1, 64, device="mps")
            _ = gru(x)
            info["gru_support"] = True
        except:
            info["gru_support"] = False
    
    return info


if __name__ == "__main__":
    print("MPS Capability Check:")
    caps = check_mps_capabilities()
    for k, v in caps.items():
        print(f"  {k}: {v}")
    print(f"\nRecommended device: {get_device()}")

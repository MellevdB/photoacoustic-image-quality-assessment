"""
Estimate FLOPs and measure inference time for the three models:
1. best_model (PhotoacousticQualityNet)
2. IQDCNN
3. EfficientNetIQA
"""

import torch
import time
import os
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from dl_model.inference import load_model_checkpoint

# Initialize flags
PTFLOPS_AVAILABLE = False
FVCORE_AVAILABLE = False

try:
    from ptflops import get_model_complexity_info
    PTFLOPS_AVAILABLE = True
except ImportError:
    try:
        from fvcore.nn import flop_count
        FVCORE_AVAILABLE = True
    except ImportError:
        print("Warning: Neither ptflops nor fvcore is installed. Install one with:")
        print("  pip install ptflops")
        print("  or")
        print("  pip install fvcore")
        print("Falling back to inference time only...")

def count_flops_ptflops(model, input_shape):
    """Count FLOPs using ptflops."""
    # ptflops expects (C, H, W) format
    flops, params = get_model_complexity_info(
        model, input_shape, print_per_layer_stat=False, verbose=False
    )
    # Extract numeric value from string like "123.45 GFLOPs"
    # flops is a string like "0.123 GFLOPs"
    flops_str = flops.split()[0]
    unit = flops.split()[1] if len(flops.split()) > 1 else ''
    
    flops_val = float(flops_str)
    if 'G' in unit or 'B' in unit:
        flops_val *= 1e9
    elif 'M' in unit:
        flops_val *= 1e6
    elif 'K' in unit:
        flops_val *= 1e3
    
    return flops_val, {}

def count_flops_fvcore(model, input_shape):
    """Count FLOPs using fvcore."""
    model.eval()
    dummy_input = torch.randn(1, *input_shape)
    
    # fvcore API
    flops_dict, _ = flop_count(model, (dummy_input,))
    
    total_flops = sum(flops_dict.values())
    return total_flops, flops_dict

def measure_inference_time(model, dummy_input, device='cpu', num_runs=100, warmup_runs=10):
    """Measure average inference time."""
    model.eval()
    dummy_input = dummy_input.to(device)
    model = model.to(device)
    
    # Warmup runs
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(dummy_input)
    
    # Synchronize GPU if available
    if device.startswith('cuda') and torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Measure inference time
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            if device.startswith('cuda') and torch.cuda.is_available():
                torch.cuda.synchronize()
            
            start_time = time.time()
            _ = model(dummy_input)
            
            if device.startswith('cuda') and torch.cuda.is_available():
                torch.cuda.synchronize()
            
            end_time = time.time()
            times.append((end_time - start_time) * 1000)  # Convert to milliseconds
    
    avg_time = sum(times) / len(times)
    std_time = (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5
    return avg_time, std_time

def analyze_model(model_name, model_path, device='cpu', input_shape=(1, 128, 128)):
    """Analyze a single model: load it, count FLOPs, and measure inference time."""
    print(f"\n{'='*60}")
    print(f"Analyzing model: {model_name}")
    print(f"{'='*60}")
    
    # Check if checkpoint exists
    if not os.path.exists(model_path):
        print(f"Warning: Checkpoint not found at {model_path}")
        print("Skipping this model...")
        return None
    
    # Load model
    print(f"Loading model from: {model_path}")
    try:
        model = load_model_checkpoint(model_path, device=device)
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return None
    
    # Create dummy input
    dummy_input = torch.randn(1, *input_shape)
    print(f"Input shape: {dummy_input.shape}")
    
    # Count FLOPs
    print("\nCounting FLOPs...")
    if PTFLOPS_AVAILABLE:
        total_flops, _ = count_flops_ptflops(model, input_shape)
        print(f"Total FLOPs: {total_flops:,.0f}")
        print(f"Total FLOPs (GFLOPs): {total_flops / 1e9:.4f}")
    elif FVCORE_AVAILABLE:
        total_flops, flops_dict = count_flops_fvcore(model, input_shape)
        print(f"Total FLOPs: {total_flops:,.0f}")
        print(f"Total FLOPs (GFLOPs): {total_flops / 1e9:.4f}")
    else:
        print("FLOP counting libraries not available. Skipping FLOP count.")
        total_flops = None
    
    # Measure inference time
    print(f"\nMeasuring inference time (device: {device})...")
    try:
        avg_time, std_time = measure_inference_time(
            model, dummy_input, device=device, num_runs=100, warmup_runs=10
        )
        print(f"Average inference time: {avg_time:.3f} ms ± {std_time:.3f} ms")
    except Exception as e:
        print(f"Error measuring inference time: {e}")
        avg_time, std_time = None, None
    
    return {
        'model_name': model_name,
        'model_path': model_path,
        'total_flops': total_flops,
        'gflops': total_flops / 1e9 if total_flops is not None else None,
        'avg_inference_time_ms': avg_time,
        'std_inference_time_ms': std_time
    }

def main():
    """Main function to analyze all three models."""
    # Define model paths
    model_paths = {
        "best_model": "models/best_model/SSIM/best_model.pth",
        "IQDCNN": "models/IQDCNN/SSIM/best_model.pth",
        "EfficientNetIQA": "models/EfficientNetIQA/SSIM/best_model.pth",
    }
    
    # Detect device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Analyze each model
    results = []
    for model_name, model_path in model_paths.items():
        result = analyze_model(model_name, model_path, device=device)
        if result is not None:
            results.append(result)
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Model':<20} {'GFLOPs':<12} {'Avg Time (ms)':<15} {'Std Time (ms)':<15}")
    print("-" * 60)
    
    for result in results:
        gflops_str = f"{result['gflops']:.4f}" if result['gflops'] is not None else "N/A"
        avg_time_str = f"{result['avg_inference_time_ms']:.3f}" if result['avg_inference_time_ms'] is not None else "N/A"
        std_time_str = f"{result['std_inference_time_ms']:.3f}" if result['std_inference_time_ms'] is not None else "N/A"
        
        print(f"{result['model_name']:<20} {gflops_str:<12} {avg_time_str:<15} {std_time_str:<15}")

if __name__ == "__main__":
    main()


# File: check_gpu.py

import torch
import subprocess

def check_gpu_status():
    print("Checking GPU status...")
    
    # Check if CUDA is available
    print(f"CUDA is available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        # Get the number of GPUs
        gpu_count = torch.cuda.device_count()
        print(f"Number of GPUs: {gpu_count}")
        
        # Get information about each GPU
        for i in range(gpu_count):
            print(f"\nGPU {i}:")
            print(f"  Name: {torch.cuda.get_device_name(i)}")
            print(f"  Total memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
            print(f"  Memory allocated: {torch.cuda.memory_allocated(i) / 1e9:.2f} GB")
            print(f"  Memory cached: {torch.cuda.memory_reserved(i) / 1e9:.2f} GB")
    
    # Run nvidia-smi command
    try:
        print("\nnvidia-smi output:")
        nvidia_smi_output = subprocess.check_output(["nvidia-smi"]).decode("utf-8")
        print(nvidia_smi_output)
    except subprocess.CalledProcessError:
        print("nvidia-smi command failed. Make sure it's installed and in your PATH.")
    except FileNotFoundError:
        print("nvidia-smi command not found. Make sure it's installed and in your PATH.")

if __name__ == "__main__":
    check_gpu_status()

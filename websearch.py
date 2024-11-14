import torch

print(f"CUDA driver version: {torch.version.cuda}")
print(f"CUDA runtime version: {torch.version.cuda_runtime}")

# Compare the CUDA driver version with the CUDA runtime version
if torch.version.cuda != torch.version.cuda_runtime:
    print("CUDA driver version mismatch!")
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from lib.data import tvb192_weights_lengths
import numpy as np

W, D = tvb192_weights_lengths()

total = W.size
zeros = np.sum(W == 0)
nonzeros = total - zeros
sparsity = zeros / total

print(f"Matrix shape:              {W.shape}")
print(f"Total elements:            {total}")
print(f"Zero elements:             {zeros}")
print(f"Non-zero elements:         {nonzeros}")
print(f"Sparsity (fraction zeros): {sparsity:.4f}  ({sparsity * 100:.2f}%)")
print(f"Density (fraction nonzeros): {1 - sparsity:.4f}  ({(1 - sparsity) * 100:.2f}%)")

import numpy as np
import pickle

# Load the matrices from the pickle files
with open('matrix1.pkl', 'rb') as f:
    matrix1 = pickle.load(f)

with open('matrix2.pkl', 'rb') as f:
    matrix2 = pickle.load(f)

# Compare the matrices
comparison = np.allclose(matrix1, matrix2)

print(f"The matrices are {'the same' if comparison else 'different'}")

# Find the indices where the matrices differ
differences = np.where(matrix1 != matrix2)

print(f"The matrices differ at indices: {list(zip(differences[0], differences[1]))}")
import torch
import concurrent.futures

# Create two 1000 x 1000 matrices with random float values
MATRIX_SIZE = 1000
matrix_a = torch.randn(MATRIX_SIZE, MATRIX_SIZE)
matrix_b = torch.randn(MATRIX_SIZE, MATRIX_SIZE)

# Initialize the result matrix
result = torch.zeros(MATRIX_SIZE, MATRIX_SIZE)

# Define the function for matrix multiplication
def matrix_multiply(start, end):
    result[start:end] = torch.mm(matrix_a[start:end], matrix_b)

# Use ThreadPoolExecutor to perform the matrix multiplication in parallel
with concurrent.futures.ThreadPoolExecutor() as executor:
    futures = [executor.submit(matrix_multiply, i, i + MATRIX_SIZE // 4) for i in range(0, MATRIX_SIZE, MATRIX_SIZE // 4)]
    concurrent.futures.wait(futures)

# Print the result matrix
print(result)

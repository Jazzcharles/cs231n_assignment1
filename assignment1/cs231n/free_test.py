import numpy as np
import math

def cal():
    a = np.array([[1, 2, 3], [4, 5, 6]])
    b = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    mask = np.random.choice(5, 3)
    print(mask)
    print(b[mask, :])

if __name__ == '__main__':
    cal()


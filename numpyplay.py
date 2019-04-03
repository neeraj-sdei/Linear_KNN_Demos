import numpy as np

arr= np.array([[1,2,3],[4,5,6],[5,5,6]])

print(arr)
print(arr.shape)

print(arr.ravel())

print(arr.shape)
print(arr.sum(axis=1))
print(np.corrcoef(arr))
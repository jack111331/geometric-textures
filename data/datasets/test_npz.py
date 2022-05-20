import numpy as np
import cv2
npz = np.load("02691156_test_images.npz")
# "arr_0": 
item = list(npz.items())[0][1]
# print(len(item))
# for i in item:
#     print(i.shape)
img = np.zeros((64, 64, 3))
img[..., 0] = item[0][0, 0]
img[..., 1] = item[0][0, 1]
img[..., 2] = item[0][0, 2]
cv2.imwrite("02691156_test_images0.png", img)
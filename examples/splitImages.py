import numpy as np
from PIL import Image
import os
import glob

npzPath = glob.glob(os.path.join("./", "*.npz"))

index = 0
print(npzPath)
for path in npzPath:
    dirPath = os.path.join('./' + str(index) + '/')
    
    if not os.path.exists(dirPath):
        os.mkdir(dirPath)

    a = np.load(path, allow_pickle=True)['arr_0']
    imgNum = 0
    for i in range(a.shape[0]):
        photo = np.transpose(a[i][0], (1,2,0))
        img = Image.fromarray(photo, 'RGBA')
        img.save(dirPath + str(imgNum) + '.png')
        imgNum += 1
    index += 1
# a = a['arr_0'][0]
# print(a)aaa                                                             
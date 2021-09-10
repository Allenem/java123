import matplotlib
from matplotlib import pylab as plt
import nibabel as nib
from nibabel.viewers import OrthoSlicer3D

file = r'D:\Download\Wechat\WeChat Files\wxid_v9z9ks7h6bou22\FileStorage\File\2021-08\TrainingData\TrainingData\1_QSM.nii.gz'
img = nib.load(file)
print(img)
print('————————————————————')
print(type(img))
print('————————————————————')
print(img.header['db_name'])  # 输出nii的头文件
print('————————————————————')
print(img.header)
print('————————————————————')
width, height, queue = img.dataobj.shape
print(type(img.dataobj))
print(width, height, queue)
print('————————————————————')
OrthoSlicer3D(img.dataobj).show()

num = 1
for i in range(0, queue, 10):
    img_arr = img.dataobj[:, :, i]
    print(type(img_arr))
    plt.subplot(5, 4, num)
    plt.imshow(img_arr, cmap='gray')
    num += 1
plt.show()
'''
file = r'D:\Download\Wechat\WeChat Files\wxid_v9z9ks7h6bou22\FileStorage\File\2021-08\TrainingData\TrainingData\1_QSM.nii.gz'
img = nib.load(file)
img_arr = img.dataobj # (width, height, queue)
'''
import SimpleITK as sitk
import skimage.io as io
from matplotlib import pylab as plt


def read_img(path):
    img = sitk.ReadImage(path)
    print(type(img))
    data = sitk.GetArrayFromImage(img)
    return data


# 显示一个系列图
def show_img(data):
    for i in range(data.shape[0]):
        plt.imshow(data[i, :, :], cmap='gray')
        print(i)
        plt.show()


# 单张显示
# def show_img(ori_img):
#     io.imshow(ori_img[100], cmap='gray')
#     io.show()

# window下的文件夹路径
path = '../data/CAT08/dataset00/image00.nii.gz'
data = read_img(path)
print(data.shape)  # queue, height, width
show_img(data)
'''
file = r'D:\Download\Wechat\WeChat Files\wxid_v9z9ks7h6bou22\FileStorage\File\2021-08\TrainingData\TrainingData\1_QSM.nii.gz'
img = sitk.ReadImage(path)
img_arr = sitk.GetArrayFromImage(img)  # (queue, height, width)
'''
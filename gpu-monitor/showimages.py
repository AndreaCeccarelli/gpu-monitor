import numpy as np
from PIL import Image
from skimage import data
from skimage.color import rgb2gray

def to_gray_uint(image):
    return np.uint8(rgb2gray(image) * 255)


w, h = 32, 32
data1=np.load("./synteticattacks/Zoo_cifar10_pytorch_x.npy")
data = np.swapaxes(data1, 1, 3).astype(np.float32)

d = data[10].astype(np.float32) #/ 1 # normalize the data to 0 - 1
d = 255 * d # Now scale by 255
img = d.astype(np.uint8)
#print(data[8])
print (type(img))
img1 = Image.fromarray(img, 'RGB')

img1.show()

'''
#PER STAMPARE MNIST
w, h = 28, 28
data1=np.load("allwhite_mnist_pytorch.npy")
data = np.swapaxes(data1, 1, 3).astype(np.float32)
d = data[10].astype(np.float32) #/ 1 # normalize the data to 0 - 1

d = 255 * d # Now scale by 255
img = d.astype(np.uint8)
#print(data[8])
print (type(img))
img1 = Image.fromarray(img.squeeze(), 'L')

img1.show()
'''

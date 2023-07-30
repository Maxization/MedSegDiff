import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import scipy.ndimage as ndi

result_path = 'E:\\examples'
image_path = 'D:\\Segmentacja\\Experiments\\MRI\\data\\mri_normalized\\TCGA_CS_4941_19960909\\TCGA_CS_4941_19960909_12.tif'
mask_path = 'D:\\Segmentacja\\Experiments\\MRI\\data\\mri_normalized\\TCGA_CS_4941_19960909\\TCGA_CS_4941_19960909_12_mask.tif'
tran_list = [transforms.Resize((256, 256)), transforms.Grayscale(),]
transform_train = transforms.Compose(tran_list)

tran_list2 = [transforms.Resize((256, 256)), transforms.Lambda(lambda x: ndi.gaussian_filter(x, sigma=1)),]
transform_train2 = transforms.Compose(tran_list2)

img = Image.open(image_path).convert('RGB')
img = transform_train(img)
img.save(result_path + "/" + 'gray.png')

img2 = Image.open(image_path).convert('RGB')
img2 = transform_train2(img2)
img2 = Image.fromarray(img2)
img2.save(result_path + "/" + 'filter.png')

img3 = Image.open(image_path).convert('RGB')
img3.save(result_path + "/" + 'orginal.png')

img4 = Image.open(mask_path).convert('L')
img4.save(result_path + "/" + 'mask.png')




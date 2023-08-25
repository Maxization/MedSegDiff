import cv2
import numpy as np
from PIL import Image


def segment_breast(img, low_int_threshold=.05, crop=True):
  '''Perform breast segmentation
  Args:
      low_int_threshold([float or int]): Low intensity threshold to
              filter out background. It can be a fraction of the max
              intensity value or an integer intensity value.
      crop ([bool]): Whether or not to crop the image.
  Returns:
      An image of the segmented breast.
  NOTES: the low_int_threshold is applied to an image of dtype 'uint8',
      which has a max value of 255.
  '''
  # Create img for thresholding and contours.
  img_8u = (img.astype('float32') / img.max() * 255).astype('uint8')
  if low_int_threshold < 1.:
    low_th = int(img_8u.max() * low_int_threshold)
  else:
    low_th = int(low_int_threshold)
  _, img_bin = cv2.threshold(
    img_8u, low_th, maxval=255, type=cv2.THRESH_BINARY)

  contours, _ = cv2.findContours(
    img_bin.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  cont_areas = [cv2.contourArea(cont) for cont in contours]
  idx = np.argmax(cont_areas)  # find the largest contour, i.e. breast.
  breast_mask = cv2.drawContours(
    np.zeros_like(img_bin), contours, idx, 255, -1)  # fill the contour.

  return breast_mask

path = 'E:\\mri\\train\\TCGA_CS_4942_19970222\\TCGA_CS_4942_19970222_9.tif'

img = cv2.imread(path)
img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)
mask = segment_breast(np.array(img))
mask = mask[:, :, None]
print(mask.shape)
#img_mask = Image.fromarray(mask)
#img_mask.show()


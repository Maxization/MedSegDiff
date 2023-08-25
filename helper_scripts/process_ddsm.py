import pydicom
from PIL import Image
import numpy as np
import cv2


def select_largest_obj(img_bin, lab_val=255, fill_holes=False,
                       smooth_boundary=False, kernel_size=15):
  '''Select the largest object from a binary image and optionally
  fill holes inside it and smooth its boundary.
  Args:
      img_bin (2D array): 2D numpy array of binary image.
      lab_val ([int]): integer value used for the label of the largest
              object. Default is 255.
      fill_holes ([boolean]): whether fill the holes inside the largest
              object or not. Default is false.
      smooth_boundary ([boolean]): whether smooth the boundary of the
              largest object using morphological opening or not. Default
              is false.
      kernel_size ([int]): the size of the kernel used for morphological
              operation. Default is 15.
  Returns:
      a binary image as a mask for the largest object.
  '''
  n_labels, img_labeled, lab_stats, _ = \
    cv2.connectedComponentsWithStats(img_bin, connectivity=8,
                                     ltype=cv2.CV_32S)
  largest_obj_lab = np.argmax(lab_stats[1:, 4]) + 1
  largest_mask = np.zeros(img_bin.shape, dtype=np.uint8)
  largest_mask[img_labeled == largest_obj_lab] = lab_val
  # import pdb; pdb.set_trace()
  if fill_holes:
    bkg_locs = np.where(img_labeled == 0)
    bkg_seed = (bkg_locs[0][0], bkg_locs[1][0])
    img_floodfill = largest_mask.copy()
    h_, w_ = largest_mask.shape
    mask_ = np.zeros((h_ + 2, w_ + 2), dtype=np.uint8)
    cv2.floodFill(img_floodfill, mask_, seedPoint=bkg_seed,
                  newVal=lab_val)
    holes_mask = cv2.bitwise_not(img_floodfill)  # mask of the holes.
    largest_mask = largest_mask + holes_mask
  if smooth_boundary:
    kernel_ = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    largest_mask = cv2.morphologyEx(largest_mask, cv2.MORPH_OPEN,
                                    kernel_)

  return largest_mask

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
  # segment the breast.
  img_breast_only = cv2.bitwise_and(img, img, mask=breast_mask)
  x, y, w, h = cv2.boundingRect(contours[idx])
  if crop:
    img_breast_only = img_breast_only[y:y + h, x:x + w]
    breast_mask = breast_mask[y:y + h, x:x + w]
  return img_breast_only, breast_mask, (x, y, w, h)

def normalize(arr):
  arr = arr.astype('float')
  minval = arr.min()
  maxval = arr.max()
  if minval != maxval:
    arr -= minval
    arr *= (255.0 / (maxval - minval))
  return arr

def get_uint8_array(dcm):
  arr = dcm.pixel_array
  if dcm.BitsAllocated == 16:
    arr = (arr / 256).astype('uint8')

  arr = normalize(arr)
  return arr.astype('uint8')

path = "E:\\CBIS-DDSM\\Calc-Test_P_00038_LEFT_CC\\1.3.6.1.4.1.9590.100.1.2.85935434310203356712688695661986996009\\1.3.6.1.4.1.9590.100.1.2.374115997511889073021386151921807063992\\1-1.dcm"

image_dcm = pydicom.dcmread(path)
image_array = get_uint8_array(image_dcm)

image_array, breast_mask, _ = segment_breast(image_array)

img = Image.fromarray(image_array)
img.show()

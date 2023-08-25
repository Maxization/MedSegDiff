from PIL import Image, ImageOps
import pandas as pd
import pydicom
import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import cv2

def normalize(arr):
  arr = arr.astype('float')
  minval = arr.min()
  maxval = arr.max()
  if minval != maxval:
    arr -= minval
    arr *= (255.0 / (maxval - minval))
  return arr

def crop(img, mask):
  """
  Crop breast ROI from image.
  @img : numpy array image
  @mask : numpy array mask of the lesions
  return: numpy array of the ROI extracted for the image,
          numpy array of the ROI extracted for the breast mask,
          numpy array of the ROI extracted for the masses mask
  """
  # Otsu's thresholding after Gaussian filtering
  blur = cv2.GaussianBlur(img, (5, 5), 0)
  _, breast_mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

  img1 = Image.fromarray(breast_mask)
  img1.show()
  cnts, _ = cv2.findContours(breast_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  cnt = max(cnts, key=cv2.contourArea)
  x, y, w, h = cv2.boundingRect(cnt)

  return img[y:y + h, x:x + w], mask[y:y + h, x:x + w]

def segment_breast(img, mask, low_int_threshold=.05, crop=True):
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
    mask = mask[y:y + h, x:x + w]
  return img_breast_only, mask

def clear_calc(calc_data):
  calc_cleaning_1 = calc_data.copy()
  calc_cleaning_1 = calc_cleaning_1.rename(columns={'calc type': 'calc_type'})
  calc_cleaning_1 = calc_cleaning_1.rename(columns={'calc distribution': 'calc_distribution'})
  calc_cleaning_1 = calc_cleaning_1.rename(columns={'image view': 'image_view'})
  calc_cleaning_1 = calc_cleaning_1.rename(columns={'left or right breast': 'left_or_right_breast'})
  calc_cleaning_1 = calc_cleaning_1.rename(columns={'breast density': 'breast_density'})
  calc_cleaning_1 = calc_cleaning_1.rename(columns={'abnormality type': 'abnormality_type'})
  calc_cleaning_1 = calc_cleaning_1.rename(columns={'abnormality_id': 'abnormality_id'})
  calc_cleaning_1 = calc_cleaning_1.rename(columns={'image file path': 'image_file_path'})
  calc_cleaning_1 = calc_cleaning_1.rename(columns={'cropped image file path': 'cropped_image_file_path'})
  calc_cleaning_1 = calc_cleaning_1.rename(columns={'ROI mask file path': 'ROI_mask_file_path'})
  calc_cleaning_1['pathology'] = calc_cleaning_1['pathology'].astype('category')
  calc_cleaning_1['calc_type'] = calc_cleaning_1['calc_type'].astype('category')
  calc_cleaning_1['calc_distribution'] = calc_cleaning_1['calc_distribution'].astype('category')
  calc_cleaning_1['abnormality_type'] = calc_cleaning_1['abnormality_type'].astype('category')
  calc_cleaning_1['image_view'] = calc_cleaning_1['image_view'].astype('category')
  calc_cleaning_1['left_or_right_breast'] = calc_cleaning_1['left_or_right_breast'].astype('category')
  calc_cleaning_1['calc_type'].fillna(method='bfill', axis=0, inplace=True)
  calc_cleaning_1['calc_distribution'].fillna(method='bfill', axis=0, inplace=True)

  calc_cleaning_1 = calc_cleaning_1.reset_index(drop=True)
  return calc_cleaning_1

def clear_mass(mass_data):
  mass_cleaning_2 = mass_data.copy()
  mass_cleaning_2 = mass_cleaning_2.rename(columns={'mass shape': 'mass_shape'})
  mass_cleaning_2 = mass_cleaning_2.rename(columns={'left or right breast': 'left_or_right_breast'})
  mass_cleaning_2 = mass_cleaning_2.rename(columns={'mass margins': 'mass_margins'})
  mass_cleaning_2 = mass_cleaning_2.rename(columns={'image view': 'image_view'})
  mass_cleaning_2 = mass_cleaning_2.rename(columns={'abnormality type': 'abnormality_type'})
  mass_cleaning_2 = mass_cleaning_2.rename(columns={'abnormality_id': 'abnormality_id'})
  mass_cleaning_2 = mass_cleaning_2.rename(columns={'image file path': 'image_file_path'})
  mass_cleaning_2 = mass_cleaning_2.rename(columns={'cropped image file path': 'cropped_image_file_path'})
  mass_cleaning_2 = mass_cleaning_2.rename(columns={'ROI mask file path': 'ROI_mask_file_path'})
  mass_cleaning_2['left_or_right_breast'] = mass_cleaning_2['left_or_right_breast'].astype('category')
  mass_cleaning_2['image_view'] = mass_cleaning_2['image_view'].astype('category')
  mass_cleaning_2['mass_margins'] = mass_cleaning_2['mass_margins'].astype('category')
  mass_cleaning_2['mass_shape'] = mass_cleaning_2['mass_shape'].astype('category')
  mass_cleaning_2['abnormality_type'] = mass_cleaning_2['abnormality_type'].astype('category')
  mass_cleaning_2['pathology'] = mass_cleaning_2['pathology'].astype('category')
  mass_cleaning_2['mass_shape'].fillna(method='bfill', axis=0, inplace=True)
  mass_cleaning_2['mass_margins'].fillna(method='bfill', axis=0, inplace=True)

  mass_cleaning_2 = mass_cleaning_2.reset_index(drop=True)
  return mass_cleaning_2

def get_uint8_array(dcm):
  arr = dcm.pixel_array
  if dcm.BitsAllocated == 16:
    arr = (arr / 256).astype('uint8')

  arr = normalize(arr)
  return arr.astype('uint8')

def get_image_paths(image_folder, mask_folder):
  image_path = os.path.join(image_folder, "1-1.dcm")
  mask_path = os.path.join(mask_folder, "1-1.dcm")
  cropped_path = os.path.join(mask_folder, "1-2.dcm")

  if not os.path.isfile(image_path) or not os.path.isfile(mask_path) or not os.path.isfile(cropped_path):
    print('Missing image')
    return -1, -1, -1

  file_stats1 = os.stat(mask_path)
  file_stats2 = os.stat(cropped_path)
  if file_stats1.st_size < file_stats2.st_size:
    tmp = cropped_path
    cropped_path = mask_path
    mask_path = tmp

  return image_path, cropped_path, mask_path

def process_dataset(df, save_train_path):
  image_number = 1
  for index, row in df.iterrows():
    if image_number % 10 == 0:
      print(f'Image {image_number}')

    image_path = path + 'CBIS-DDSM/' + row['image_file_path']
    mask_path = path + 'CBIS-DDSM/' + row['ROI_mask_file_path']

    image_folder = os.path.dirname(os.path.abspath(image_path))
    mask_folder = os.path.dirname(os.path.abspath(mask_path))

    (image_path, _, mask_path) = get_image_paths(image_folder, mask_folder)

    if image_path == -1:
      continue

    image_dcm = pydicom.dcmread(image_path)
    mask_dcm = pydicom.dcmread(mask_path)

    image_array = get_uint8_array(image_dcm)
    mask_array = get_uint8_array(mask_dcm)

    if image_array.shape[0] != mask_array.shape[0] or image_array.shape[1] != mask_array.shape[1]:
      print("Invalid shapes")
      #exit(-1)

    margin = 60
    image_array = image_array[margin:image_array.shape[0] - margin, margin:image_array.shape[1] - margin]
    mask_array = mask_array[margin:image_array.shape[0] - margin, margin:image_array.shape[1] - margin]

    image_array, mask_array = segment_breast(image_array, mask_array)

    im2 = Image.fromarray(mask_array)


    im1 = Image.fromarray(image_array)
    #im1.show()
    #im1 = ImageOps.equalize(im1, mask=None)
    #im1.show()
    im1 = im1.resize((256, 256))

    #im2.show()
    im2 = im2.resize((256, 256))
    image_save_path = save_train_path + str(image_number) + ".tif"
    mask_save_path = save_train_path + str(image_number) + "_mask.tif"
    im1.save(image_save_path, quality=100)
    #im2.save(mask_save_path, quality=100)

    image_number = image_number + 1

path = "E:/"

mass_train_data = pd.read_csv(path + "mass_case_description_train_set.csv")
mass_test_data = pd.read_csv(path + "mass_case_description_test_set.csv")


mass_shape = ['OVAL', 'OVAL-LYMPH_NODE', 'ROUND', 'ROUND-IRREGULAR-ARCHITECTURAL_DISTORTION', 'ROUND-OVAL']
mass_margins = ['SPICULATED', 'ILL_DEFINED-SPICULATED', 'MICROLOBULATED-SPICULATED', 'OBSCURED-SPICULATED', 'OBSCURED-ILL_DEFINED-SPICULATED']
image_view_MLO = 'MLO'
image_view_CC = 'CC'


# Mass breast_density
mass_train_data = clear_mass(mass_train_data)
mass_train_data = mass_train_data.loc[mass_train_data['assessment'] > 3]
mass_train_data = mass_train_data.loc[mass_train_data['image_view'] == image_view_CC]
mass_train_data = mass_train_data[mass_train_data['mass_shape'].isin(mass_shape)]

mass_test_data = clear_mass(mass_test_data)
mass_test_data = mass_test_data.loc[mass_test_data['assessment'] > 3]
mass_test_data = mass_test_data.loc[mass_test_data['image_view'] == image_view_CC]
mass_test_data = mass_test_data[mass_test_data['mass_shape'].isin(mass_shape)]

print(len(mass_test_data))
print(len(mass_train_data))

process_dataset(mass_test_data, "E:/experiment2/oval_cc/test/")
process_dataset(mass_train_data, "E:/experiment2/oval_cc/train/")
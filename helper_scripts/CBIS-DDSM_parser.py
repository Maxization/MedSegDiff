from PIL import Image
import pandas as pd
import pydicom
import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def normalize(arr):
  arr = arr.astype('float')
  minval = arr.min()
  maxval = arr.max()
  if minval != maxval:
    arr -= minval
    arr *= (255.0 / (maxval - minval))
  return arr

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

    im1 = Image.fromarray(image_array)
    im1 = im1.resize((320, 512))

    im2 = Image.fromarray(mask_array)
    im2 = im2.resize((320, 512))

    image_save_path = save_train_path + str(image_number) + ".tif"
    mask_save_path = save_train_path + str(image_number) + "_mask.tif"
    im1.save(image_save_path, quality=100)
    im2.save(mask_save_path, quality=100)

    image_number = image_number + 1

path = "E:/"

calc_train_data = pd.read_csv(path + "calc_case_description_train_set.csv")
mass_train_data = pd.read_csv(path + "mass_case_description_train_set.csv")

calc_test_data = pd.read_csv(path + "calc_case_description_test_set.csv")
mass_test_data = pd.read_csv(path + "mass_case_description_test_set.csv")

# Mass
mass_train_data = clear_mass(mass_train_data)
mass_train_data = mass_train_data.loc[mass_train_data['image_view'] == 'MLO']
mass_train_data = mass_train_data.loc[mass_train_data['assessment'] > 3]

mass_test_data = clear_mass(mass_test_data)
mass_test_data = mass_test_data.loc[mass_test_data['image_view'] == 'MLO']
mass_test_data = mass_test_data.loc[mass_test_data['assessment'] > 3]

# Calc
#calc_test_data = clear_calc(calc_test_data)
#calc_test_data = calc_test_data.loc[calc_test_data['image_view'] == 'MLO']
#calc_test_data = calc_test_data.loc[calc_test_data['assessment'] > 3]

#calc_train_data = clear_calc(calc_train_data)
#calc_train_data = calc_train_data.loc[calc_train_data['image_view'] == 'MLO']
#calc_train_data = calc_train_data.loc[calc_train_data['assessment'] > 3]

print(len(mass_test_data))
print(len(mass_train_data))

#print(len(mass_test_data))
#print(len(mass_train_data))

process_dataset(mass_test_data, "E:/mass_mlo_mali_test/")
process_dataset(mass_train_data, "E:/mass_mlo_mali_train/")

#process_dataset(mass_test_data, "E:/mass_mlo_test/")
#process_dataset(mass_train_data, "E:/mass_mlo_train/")

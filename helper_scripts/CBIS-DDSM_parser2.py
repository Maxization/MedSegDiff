from PIL import Image
import pandas as pd
import pydicom
import numpy as np
from pathlib import Path

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

  return mass_cleaning_2

path = "E:/"

calc_train_data = pd.read_csv(path + "calc_case_description_train_set.csv")
mass_train_data = pd.read_csv(path + "mass_case_description_train_set.csv")

calc_test_data = pd.read_csv(path + "calc_case_description_test_set.csv")
mass_test_data = pd.read_csv(path + "mass_case_description_test_set.csv")

calc_train_data = clear_calc(calc_train_data)

image_path = path + 'CBIS-DDSM/' + calc_train_data.iloc[0]['image_file_path']
dcm = pydicom.dcmread(image_path)

im1 = Image.fromarray(dcm.pixel_array)
im1.show()
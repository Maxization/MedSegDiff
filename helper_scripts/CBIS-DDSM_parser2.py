from PIL import Image
import pandas as pd
import numpy as np
from pathlib import Path

path = "D:/Segmentacja/Experiments/CBIS-DDSM/"
csv_path = path + "csv/dicom_info.csv"
dicom_data = pd.read_csv(csv_path)

cropped_images=dicom_data[dicom_data.SeriesDescription == 'cropped images']
cropped_images["image_path"] = cropped_images["image_path"].apply(lambda x: x.replace('CBIS-DDSM/', ""))
cropped_images = cropped_images.reset_index()

full_mammogram_images=dicom_data[dicom_data.SeriesDescription == 'full mammogram images']
full_mammogram_images["image_path"] = full_mammogram_images["image_path"].apply(lambda x: x.replace('CBIS-DDSM/', ""))
full_mammogram_images = full_mammogram_images.reset_index()

ROI_mask_images=dicom_data[dicom_data.SeriesDescription == 'ROI mask images']
ROI_mask_images["image_path"] = ROI_mask_images["image_path"].apply(lambda x: x.replace('CBIS-DDSM/', ""))
ROI_mask_images = ROI_mask_images.reset_index()

calc_data = pd.read_csv(path + "csv/calc_case_description_train_set.csv")
mass_data = pd.read_csv(path + "csv/mass_case_description_train_set.csv")

calc_test_data = pd.read_csv(path + "csv/calc_case_description_test_set.csv")
mass_test_data = pd.read_csv(path + "csv/mass_case_description_test_set.csv")

def clear_calc(calc_data):
  calc_cleaning_1 = calc_data.copy()
  calc_cleaning_1 = calc_cleaning_1.rename(columns={'calc type': 'calc_type'})
  calc_cleaning_1 = calc_cleaning_1.rename(columns={'calc distribution': 'calc_distribution'})
  calc_cleaning_1 = calc_cleaning_1.rename(columns={'image view': 'image_view'})
  calc_cleaning_1 = calc_cleaning_1.rename(columns={'left or right breast': 'left_or_right_breast'})
  calc_cleaning_1 = calc_cleaning_1.rename(columns={'breast density': 'breast_density'})
  calc_cleaning_1 = calc_cleaning_1.rename(columns={'abnormality type': 'abnormality_type'})
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

def parse_df(df):
  image_file_paths = []
  cropped_image_file_paths = []
  ROI_mask_file_paths = []
  for i in range(0, len(df)):
    str = df.iloc[i]["image_file_path"]
    str = str[:str.rfind('/')]
    str = str[str.rfind('/') + 1:]

    if (full_mammogram_images[full_mammogram_images["SeriesInstanceUID"] == str].image_path).empty:
      image_file_paths.append(np.nan)
    else:
      image_file_path = full_mammogram_images[full_mammogram_images["SeriesInstanceUID"] == str].image_path.iloc[0]
      image_file_paths.append(image_file_path)

    str = df.iloc[i]["cropped_image_file_path"]
    str = str[:str.rfind('/')]
    str = str[str.rfind('/') + 1:]

    if (cropped_images[cropped_images["SeriesInstanceUID"] == str].image_path).empty:
      cropped_image_file_paths.append(np.nan)
    else:
      cropped_image_file_path = cropped_images[cropped_images["SeriesInstanceUID"] == str].image_path.iloc[0]
      cropped_image_file_paths.append(cropped_image_file_path)

    str = df.iloc[i]["ROI_mask_file_path"]
    str = str[:str.rfind('/')]
    str = str[str.rfind('/') + 1:]

    if (ROI_mask_images[ROI_mask_images["SeriesInstanceUID"] == str].image_path).empty:
      ROI_mask_file_paths.append(np.nan)
    else:
      roi_mask_file_path = ROI_mask_images[ROI_mask_images["SeriesInstanceUID"] == str].image_path.iloc[0]
      ROI_mask_file_paths.append(roi_mask_file_path)

  df["image_file_path"] = image_file_paths
  df["cropped_image_file_path"] = cropped_image_file_paths
  df["ROI_mask_file_path"] = ROI_mask_file_paths

  df = df.dropna(subset=['image_file_path', 'cropped_image_file_path', 'ROI_mask_file_path'])
  df = df.reset_index()
  return df

def display_df(df):
  for i in range(0, len(df)):
    test = df.iloc[i]
    im1 = Image.open(path + test["image_file_path"])
    im3 = Image.open(path + test["ROI_mask_file_path"])

    print(im1.size[0] / im1.size[1])
    print(im1.size)
    print(im3.size)

    im1.show()
    im3.show()

#calc_cleaning_1 = clear_calc(calc_data)
#calc_test_cleaning = clear_calc(calc_test_data)

mass_cleaning_2 = clear_mass(mass_data)
mass_test_cleaning = clear_mass(mass_test_data)

#calc_cleaning_1 = parse_df(calc_cleaning_1)
#calc_test_cleaning = parse_df(calc_test_cleaning)

mass_cleaning_2 = parse_df(mass_cleaning_2)
mass_test_cleaning = parse_df(mass_test_cleaning)

print(len(mass_cleaning_2))
print(len(mass_test_cleaning))

mass_cleaning_2.to_csv('D:/Segmentacja/Experiments/CBIS-DDSM/parsed_csv/mass_case_description_train_set.csv')
mass_test_cleaning.to_csv('D:/Segmentacja/Experiments/CBIS-DDSM/parsed_csv/mass_case_description_test_set.csv')

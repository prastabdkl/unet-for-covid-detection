import pydicom as dicom
import os
import cv2
import gdcm
from sklearn.model_selection import train_test_split
import splitfolders

# import PIL # optional
# make it True if you want in PNG format
# PNG = False
# Specify the .dcm folder path
# folder_path = r"C:\Users\prast\Downloads\CS 7000\Final report\COVID-CT-MD\COVID_dcm\P001"
# Specify the output jpg/png folder path

# covid_data =  r"C:\Users\prast\Downloads\CS 7000\Final report\COVID-CT-MD\COVID"
covid_data = r"C:\Users\prast\Downloads\CS 7000\Final report\u-net-code\unet-for-covid-detection\newdata"
train_data_dir = r'C:\Users\prast\Downloads\CS 7000\Final report\u-net-code\unet-for-covid-detection\dataset\data_c_vs_nc\train'
validation_data_dir = r'C:\Users\prast\Downloads\CS 7000\Final report\u-net-code\unet-for-covid-detection\dataset\data_c_vs_nc\validate'

output_covid = r"C:\Users\prast\Downloads\CS 7000\Final report\u-net-code\unet-for-covid-detection\output"
splitfolders.ratio(covid_data, output ="./output", seed=1337, ratio=(.8, 0.1,0.1),group_prefix=None) 
print('done')


import random

step_size =.8
path,dirs,files = next(os.walk(covid_data))
file_count = len(files)
random_train_sample = random.sample(range(file_count), .8*file_count)

image = f"IM{str(random_train_sample).zfill(4)}.jpg"

cv2.imwrite(os.path.join(output_covid, image))

        # ds = dicom.dcmread(os.path.join(folder_path, image))
# print(splitfolders.ratio)
# jpg_folder_path = r"C:\Users\prast\Downloads\CS 7000\Final report\COVID-CT-MD\COVID"

# train_size = .6


# # image_count = 1
# for i in range(1,169):
#     folder_path = r'C:\Users\prast\Downloads\CS 7000\Final report\COVID-CT-MD\COVID_dcm\P'+f'{str(i).zfill(3)}' 
    
#     images_path = os.listdir(folder_path)    

#     for j in range(-5,5):
#         middle_image_index = file_count // 2
#         image = f"IM{str(middle_image_index + j).zfill(4)}.dcm"
#         # image =f"IM{str(file_count//2).zfill(4)}.dcm"
#         # print(image)
#         image = image.replace('.dcm', '.jpg')
    
#         # image = f"Covid_{i}.jpg" + image
#         image = f"Covid_{image_count}.jpg" 
    
#         pixel_array_numpy = ds.pixel_array
#         # print(pixel_array_numpy)
#         image_count+=1
#     # print(folder_path)  
    
# print(image_count-1,' images converted')
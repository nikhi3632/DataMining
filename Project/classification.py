#!/usr/bin/env python3

import os, sys, shutil
import numpy as np
import pandas as pd
import warnings
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
seed = 0
np.random.seed(seed)
tf.random.set_seed(seed)
warnings.filterwarnings('ignore')

def get_path(index, paths):
    return paths[index]

def generate_data(df):
    cwd = os.getcwd()
    data_folder = cwd + '/classificationData'
    if os.path.exists(data_folder):
        shutil.rmtree(data_folder)
    if not os.path.exists(data_folder):
        os.mkdir(data_folder)
    labels = [label for label in df['Label'].value_counts().to_dict()]
    for label in labels:
        label_data_folder = data_folder + '/' + str(label)
        if not os.path.exists(label_data_folder):
            os.mkdir(label_data_folder)
        ic_images = df[df['Label']==label]['IC_image_path'].tolist()
        for ic_image in ic_images:
            rename_img = ic_image.split('/')[-2].split('_')[-1]+ '_'+ ic_image.split('/')[-1]
            shutil.copy(ic_image, label_data_folder + '/' + rename_img)
    return data_folder

def generate_model(folder):
    model_name = 'cnn-model.h5'
    model_path = os.getcwd() + '/' + model_name
    if os.path.exists(model_path):
        os.remove(model_path)
    image_generator = ImageDataGenerator(rescale=1/255)
    train_dataset = image_generator.flow_from_directory(
                                                        batch_size=2,
                                                        directory=folder,
                                                        shuffle=True,
                                                        target_size=(224, 224), 
                                                        subset="training",
                                                        class_mode='categorical'
                                                        )
    model = keras.models.Sequential([
                                    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape = [224, 224,3]),
                                    keras.layers.MaxPooling2D(),
                                    keras.layers.Conv2D(64, (2, 2), activation='relu'),
                                    keras.layers.MaxPooling2D(),
                                    keras.layers.Conv2D(64, (2, 2), activation='relu'),
                                    keras.layers.Flatten(),
                                    keras.layers.Dense(100, activation='relu'),
                                    keras.layers.Dense(2, activation ='softmax')
                                    ])
    model.compile(optimizer='adam',loss = 'binary_crossentropy',metrics=['accuracy'])
    model.fit(train_dataset, epochs=10)
    model.save(model_name, save_format='h5')
    return model

def process_label_csv_files(labels_path):
    label_str = '_Labels.csv'
    patient_str = 'Patient_'
    thresh_str = '_thresh.png'
    label_csv_files = [labels_path + '/' + file for file in os.listdir(labels_path) if file.endswith(label_str)]
    df_list = []
    if len(label_csv_files):
        for csv_file in label_csv_files:
            ic_paths = {}
            patient_index = int(csv_file.split('_')[-2])
            data = pd.read_csv(csv_file)
            data['Patient'] = [patient_index]*data.shape[0]
            patient_folder = patient_str + str(patient_index)
            patient_folder_path = labels_path + '/' + patient_folder
            for file_name in os.listdir(patient_folder_path):
                if file_name.endswith(thresh_str):
                    ic_index = int(file_name.split('_')[-2])
                    ic_path = patient_folder_path + '/' + file_name
                    ic_paths[ic_index] = ic_path
            data['IC_image_path'] = data.apply(lambda row: get_path(row['IC'], ic_paths), axis=1)
            df_list.append(data)
        data_frame = pd.concat(df_list, axis=0, ignore_index=True)
        data_frame['Label'] = np.where(data_frame['Label'] == 0, 0, 1)
        csv_name = 'Patient_Labels.csv'
        data_frame.to_csv(csv_name, index=False)
        return data_frame
    else:
        print("Patient Labels csv files not found")
        sys.exit(1)

def main():
    try:
        cwd = os.getcwd()
        patient_labels = cwd + '/PatientData'
        df = process_label_csv_files(patient_labels)
        data_dir = generate_data(df)
        model = generate_model(data_dir)
        print(model.summary())
    except Exception as e:
        print("Exception Occured in classification main:", e)
        sys.exit(1) 

if __name__ == "__main__":
    main()    
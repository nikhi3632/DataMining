#!/usr/bin/env python3
from tensorflow import keras
import os, sys
from natsort import natsorted
import tensorflow as tf
import numpy as np
import pandas as pd

def load_image(img_path):
    img = tf.keras.utils.load_img(img_path, target_size=(224, 224))
    img_tensor = tf.keras.utils.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255.
    return img_tensor

def test():
    try:
        model = keras.models.load_model('cnn-model.h5')
        cwd = os.getcwd()
        test_data_path = cwd + '/testPatient/test_Data'
        test_labels_csv = cwd + '/testPatient/test_Labels.csv'
        preds = []
        thresh_images = [file for file in os.listdir(test_data_path) if file.endswith('thresh.png')]
        for img in natsorted(thresh_images):
            new_image = load_image(test_data_path + '/' + img)
            pred = model.predict(new_image)
            preds.append(np.argmax(pred))
        df = pd.read_csv(test_labels_csv)
        df['Prediction'] = preds
        df_results = df[['IC', 'Prediction']]
        results_csv_name = 'Results.csv'
        df_results.to_csv(results_csv_name, index=False)
        labels = df['Label'].tolist()
        predictions = df['Prediction'].tolist()
        tp=0
        fp=0
        fn=0
        tn=0
        for i in range(len(labels)):
            if(labels[i]>0 and predictions[i]==1):
                tp+=1
            elif(labels[i]==0 and predictions[i]==0):
                tn+=1
            elif(labels[i]==0 and predictions[i]==1):
                fp+=1
            else:
                fn+=1
        Accuracy = ((tp+tn)/len(labels)) * 100
        Precision = (tp/(tp+fp)) * 100
        Specificity =  (tn/(tn+fp)) * 100
        Sensitivity = (tp/(tp+fn)) * 100
        data = [['Accuracy',Accuracy], ['Precision',Precision], ['Specificity',Specificity], ['Sensitivity',Sensitivity]]
        columns = ['Metric', 'Value']
        df_metrics = pd.DataFrame(data, columns=columns)
        metrics_csv_name = 'Metrics.csv'
        df_metrics.to_csv(metrics_csv_name, index=False)
    except Exception as e:
        print("Exception Occured in test:", e)
        sys.exit(1)    

if __name__ == "__main__":
    test()  

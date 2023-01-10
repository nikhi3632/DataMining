#!/usr/bin/env python3

import random, shutil, hashlib, collections, cv2, os, csv, sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import DBSCAN
from operator import itemgetter

BLACK_IMAGE_THRESHOLD   =   0
PIXEL_THRESHOLD         =   135
DBSCAN_EPSILON          =   5

def detect_image_boundary(image_name, image_path, boundary_folder, folder_name):
    '''Modify this robustly'''
    if not os.path.exists(boundary_folder):
        os.mkdir(boundary_folder)
    folder_path = boundary_folder + '/' + folder_name
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    image = cv2.imread(image_path)
    cvt_image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    lower = [np.mean(cvt_image[:,:,i]) for i in range(3)]
    upper = [255, 255, 255]
    lower = np.array(lower, dtype="uint8")
    upper = np.array(upper, dtype="uint8")
    mask = cv2.inRange(cvt_image, lower, upper)
    output = cv2.bitwise_and(cvt_image, cvt_image, mask=mask)
    _,thresh = cv2.threshold(mask, 0, 255, 0)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) != 0:
        cv2.drawContours(output, contours, -1, (0, 255, 0), 3)
    detected_image = folder_path + '/' + image_name[:-4] + '_detected.png'    
    cv2.imwrite(detected_image, output)

def get_ic_count(ic_path, compare_string):
    count = 0
    for name in os.listdir(ic_path):
        if name.endswith(compare_string):
            count += 1
    return count        

def get_random_image(ic_path):
    compare_string = '_thresh.png'
    ic_count = get_ic_count(ic_path, compare_string)
    suffix = random.randint(1, ic_count)
    return ic_path + 'IC_' + str(suffix) + compare_string

def generate_output_folder(ic_path):
    random_ic_image = get_random_image(ic_path)
    image = cv2.imread(random_ic_image)
    new_image = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = cv2.threshold(gray, 100, 255, cv2.THRESH_OTSU)[1]
    inverted_binary = ~binary
    contours = cv2.findContours(inverted_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]       
    with_contours = cv2.drawContours(image, contours, -1, (255, 0, 255), 3)
    first_contour = cv2.drawContours(new_image, contours, 0, (255 ,0, 255), 3)
    x, y, w, h = cv2.boundingRect(contours[0])
    cv2.rectangle(first_contour,(x,y), (x+w,y+h), (255, 0, 0), 5)
    sorted_contours= sorted(contours, key=cv2.contourArea, reverse=True)
    for (i,c) in enumerate(sorted_contours):
        x, y, w, h = cv2.boundingRect(c)
        if (cv2.contourArea(c)) > 10:
            cv2.rectangle(with_contours,(x,y), (x+w,y+h), (255, 0, 0), 5)
            cropped_contour= new_image[y:y+h, x:x+w]
            image_name= "output_number_" + str(i+1) + ".png"
            path = os.getcwd()
            new_folder = path + "/output"
            new_image_path = new_folder + "/" + image_name
            if os.path.exists(new_folder):
                cv2.imwrite(new_image_path, cropped_contour)
            else:
                os.mkdir(new_folder)
                cv2.imwrite(new_image_path, cropped_contour)

def nest_list(l, rows, columns):    
        result = []               
        start = 0
        end = columns
        for _ in range(rows): 
            result.append(l[start:end])
            start += columns
            end += columns
        return result

def get_num_cols(points):
    p1 = points[0][1]
    c1 = 0
    for p in points:
        if p[1] == p1:
            c1 += 1
        else:
            break
    return c1        

def get_r_image(unique, r_hash):
    '''We assume r-image occurs most number of times when the IC image is chunked by contours'''
    for item in unique:
        if r_hash in item:
            required_image = item[0]
            break
    return required_image

def generate_r_image(pth):
    file_list = os.walk(pth)
    unique = []
    current_dir = os.getcwd() 
    for root, _ ,files in file_list:
        for file in files:
            path = Path(os.path.join(root,file))
            fileHash = hashlib.md5(open(path,'rb').read()).hexdigest()
            unique.append((file, fileHash))
    r_hash = collections.Counter([i[1] for i in unique]).most_common()[0][0]
    old_r_image_path = current_dir + "/output/" + get_r_image(unique, r_hash)
    new_r_image_path = current_dir + "/r.png"
    os.rename(old_r_image_path, new_r_image_path)
    output_folder = current_dir + "/output/"
    shutil.rmtree(output_folder)

def get_images(image_path, template_path, folder_path, threshold = 0.9, crop_pad = 5):
    image = cv2.imread(image_path)
    template = cv2.imread(template_path)
    w, h = template.shape[:-1]
    result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    loc = np.where(result >= threshold)
    coordinates = []
    for point in zip(*loc[::-1]):
        coordinates.append((point[0] + w, point[1] + h))
    cords = coordinates[:-1]
    num_cols = get_num_cols(cords)
    num_rows = int(len(cords)/num_cols)
    # assert (num_cols*num_rows == len(cords))               
    nested_points = nest_list(cords, num_rows, num_cols)
    y_cords = [nested_points[index][0][1] for index in range(len(nested_points))]
    x_cords = [point[0] for point in nested_points[0]]
    x_cords.append(x_cords[-1] + x_cords[1] - x_cords[0])
    counter = 1
    for i in range(len(x_cords) - 1):
        for j in range(len(y_cords) - 1):
            crop_img = image[y_cords[j] : y_cords[j+1], x_cords[i]: x_cords[i+1]]
            width, height = crop_img.shape[0:2]
            cropped_img = crop_img[crop_pad : width - crop_pad, crop_pad: height - crop_pad]
            image_name = "image" + "_" + str(counter)
            cropped_image_path = folder_path + "/" + image_name + ".png"
            if os.path.exists(folder_path):
                cv2.imwrite(cropped_image_path, cropped_img)
            else:
                os.mkdir(folder_path)
                cv2.imwrite(cropped_image_path, cropped_img)
            counter += 1

def detect_black_image_with_threshold(img_pth, thresh):
    image = cv2.imread(img_pth, 0)
    if cv2.countNonZero(image) <= thresh:
        return True
    return False

def store_images(img_pth, store_pth, image_name, folder_name):
    image = cv2.imread(img_pth)
    folder_path = store_pth + '/' + folder_name
    if os.path.exists(store_pth):
        if os.path.exists(folder_path)==False:
            os.mkdir(folder_path)
        cv2.imwrite(folder_path + '/' + image_name, image)
    else:
        os.mkdir(store_pth)
        if os.path.exists(folder_path)==False:
            os.mkdir(folder_path)
        cv2.imwrite(folder_path + '/' + image_name, image)

def seperate_images(imgs_path, sep_folder, black_pth, colour_pth):
    print('Seperating images....................................')
    for dir in os.listdir(imgs_path):
        dir_imgs_list = os.listdir(imgs_path + '/'+ dir)
        for img in dir_imgs_list:
            img_path = imgs_path + '/'+ dir + '/'+ img
            if not os.path.exists(sep_folder):
                os.mkdir(sep_folder)
            if detect_black_image_with_threshold(img_path, BLACK_IMAGE_THRESHOLD):
                store_images(img_path, black_pth, dir + '_' +img,dir)
            else:
                store_images(img_path, colour_pth, dir + '_' +img,dir)          
    print('Finished seperating images...........................')

def get_colour_images_to_detect_boundary(pth, boundary_dir):
    for folder in os.listdir(pth):
        folder_name = folder
        folder = pth + '/' + folder
        for img in os.listdir(folder):
            img_path = folder + '/' + img
            detect_image_boundary(img, img_path, boundary_dir, folder_name)

def generate_csv(name, path, res, field_names):
    csv_name = name + '.csv'
    print("Writing to csv {}".format(csv_name))
    csv_path = path + '/' + csv_name
    required_dict = {}
    required_list = []
    for key, val in res.items():
        required_key = int(key.split('.')[0].split('_')[-1])
        required_dict[field_names[0]] = required_key
        required_dict[field_names[1]] = val
        required_list.append(required_dict)
        required_dict = {}
    sorted_list = sorted(required_list, key=itemgetter(field_names[0]))  
    with open(csv_path, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames = field_names)
        writer.writeheader()
        writer.writerows(sorted_list)
    print("Saved to csv {}".format(csv_name))
    print("---------------------------------------------------------")
    print("---------------------------------------------------------")   

def preprocess_and_perform_clustering(pth):
    cluster_folder = os.getcwd() + '/Clusters'
    if not os.path.exists(cluster_folder):
        os.mkdir(cluster_folder)
    for dir in os.listdir(pth):
        results_map = {}
        for image_name in os.listdir(pth + '/' + dir):
            print("Starting Preproccesing and Clustering on {}".format(image_name))
            image_path = pth + '/' + dir + '/' + image_name
            img = cv2.imread(image_path)
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img[:,:,0] = img[:,:,0] - gray_img
            img[:,:,1] = img[:,:,1] - gray_img
            img[:,:,2] = img[:,:,2] - gray_img
            converted_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            coordinates = []
            for i in range(converted_gray.shape[0]):
                for j in range(converted_gray.shape[1]):
                    if converted_gray[i][j]:
                        converted_gray[i][j] = 255 #white
                        coordinates.append((i,j))
            ic_folder = image_path.split('/')[-2]
            ic_cluster_folder_path = cluster_folder + '/' + ic_folder
            if not os.path.exists(ic_cluster_folder_path):
                os.mkdir(ic_cluster_folder_path)
            new_image_name = image_name.split('.')[0] + '_cluster.png'
            new_image_path = ic_cluster_folder_path + '/' + new_image_name
            cv2.imwrite(new_image_path, converted_gray)
            df = pd.DataFrame(coordinates, columns = ['x', 'y'])
            if df.shape[0]:
                db = DBSCAN(eps=DBSCAN_EPSILON).fit(df)
                clusters = pd.DataFrame(db.labels_, columns=['cluster_id'])
                results = pd.concat((df, clusters), axis=1)
                results_dict = dict(results['cluster_id'].value_counts())
                cluster_count = 0
                for val in results_dict.values():
                    if val > PIXEL_THRESHOLD:
                        cluster_count += 1
                results_map[image_name] = cluster_count
            else:
                results_map[image_name] = 0
            print("Finished Preproccesing and Clustering on {}".format(image_name))
            print("---------------------------------------------------------")
        field_names_ = ['SliceNumber', 'ClusterCount']
        generate_csv(dir, ic_cluster_folder_path, results_map, field_names_)

def main():
    try:
        path = os.getcwd()
        images_dir_path = path + "/" + 'Images'
        if os.path.exists(images_dir_path):
            shutil.rmtree(images_dir_path)
        ic_path = path + '/testPatient/test_Data/'
        generate_output_folder(ic_path)
        pth = path +'/output'
        generate_r_image(pth)
        template_path = path + '/r.png'
        compare_string = '_thresh.png'
        if not os.path.exists(images_dir_path):
            os.mkdir(images_dir_path)
        for name in os.listdir(ic_path):
            if name.endswith(compare_string):
                image_path = ic_path + name
                idx = len(compare_string)
                folder_name = name[:-idx]
                folder_path = images_dir_path + "/" + folder_name
                print("Starting to extract brain slices from {}".format(name))
                get_images(image_path, template_path, folder_path)
                print("Finished extracting brain slices from {}".format(name))
                print("---------------------------------------------------------")
        os.remove(template_path)
        seperation_folder = path + '/' + 'Slices'
        black_path = seperation_folder + '/' + 'black'
        colour_path = seperation_folder + '/' + 'colour'
        if os.path.exists(seperation_folder):
            shutil.rmtree(seperation_folder)
        seperate_images(images_dir_path, seperation_folder, black_path, colour_path)
        cluster_folder = path + '/Clusters'
        if os.path.exists(cluster_folder):
            shutil.rmtree(cluster_folder)
        preprocess_and_perform_clustering(colour_path)
        boundary_folder = path + '/' + 'Boundaries'
        print('Starting to detect image boundaries..................')
        if os.path.exists(boundary_folder):
            shutil.rmtree(boundary_folder)
        get_colour_images_to_detect_boundary(colour_path, boundary_folder)
        print('Finished Detecting image boundaries..................')
        print("---------------------------------------------------------")
        shutil.rmtree(images_dir_path)
    except Exception as e:
        print("Exception occured : {}".format(e))
        sys.exit(1)

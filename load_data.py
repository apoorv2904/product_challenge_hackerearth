
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2
import os, sys
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def read_img(img_path, min_side=256):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    try:
        w, h, _ = img.shape
	if w < min_side:
	    wpercent = (min_side/float(w))
	    hsize = int((float(h)*float(wpercent)))
	    img = cv2.resize(img, (min_side,hsize))
	elif h < min_side:
	    hpercent = (min_side/float(h))
	    wsize = int((float(w)*float(hpercent)))
	    img = cv2.resize(img, (wsize, min_side))
    except:
        print('Skipping bad image: ', img_path)
    return img

def read_images_df(PATH, files, min_side ):
    imgs = []
    for img_path in tqdm(files['image_id'].values):
        imgs.append(read_img(PATH + img_path + '.png', min_side=min_side))
    return imgs

def getLabelKeys( files ):

    label_list = files['label'].tolist()
    fwd_key = {k:v for v,k in enumerate(set(label_list))}
    rev_key = {v:k for v,k in enumerate(set(label_list))}
    return fwd_key, rev_key

def load_data( seed, validation_percentage, min_side  ):

    np.random.seed(seed)
    ## load files
    train = pd.read_csv('Train_Csvs/train.csv')
    test = pd.read_csv('test.csv')

    ## set path for images
    TRAIN_PATH = 'train_img/'
    TEST_PATH = 'test_img/'

    # Split into val/train data
    train_files, val_files = train_test_split(train, test_size = validation_percentage)
    
    # Load Images
    train_img = read_images_df( TRAIN_PATH, train_files, min_side=min_side )
    val_img = read_images_df( TRAIN_PATH, val_files, min_side=min_side )
    test_img = read_images_df( TEST_PATH, test, min_side=min_side )


    # normalize images
    x_train = np.array(train_img, np.float32) 
    x_test = np.array(test_img, np.float32) 
    x_val = np.array(val_img, np.float32) 

    # target variable - encoding numeric value
    fwd_key, rev_key = getLabelKeys( train_files )
    y_train = [fwd_key[k] for k in train_files['label'].tolist()]   
    y_train = np.array(y_train)

    y_val = [fwd_key[k] for k in val_files['label'].tolist()]   
    y_val = np.array(y_val)


    return x_train, x_val, x_test, y_train, y_val, fwd_key, rev_key



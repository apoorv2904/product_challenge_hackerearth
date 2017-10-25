
import image_gen_extended as T
import multiprocessing as mp
from multiprocess import Pool
from scipy.misc import imresize
import numpy as np
import pandas as pd
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.utils import to_categorical
import copy

try:
    from importlib import reload
    reload(T)
except:
    reload(T)

def reverse_preprocess_input(x0):
    x = x0 
    x *= 255.
    return x

def show_images( generator,fname, unprocess=True ):
    i_cnt = 0
    for x in generator:
	i_cnt = i_cnt + 1
	fig, axes = plt.subplots(nrows=8, ncols=4)
	fig.set_size_inches(8, 8)
	page = 0
	page_size = 32
	start_i = page * page_size
	for i, ax in enumerate(axes.flat):
	    img = x[0][i+start_i]
            print( x[0].shape )
	    if unprocess:
		im = ax.imshow( reverse_preprocess_input(img).astype('uint8') )
	    else:
		im = ax.imshow(img)
	    ax.set_axis_off()
	    ax.title.set_visible(False)
	    ax.xaxis.set_ticks([])
	    ax.yaxis.set_ticks([])
	    for spine in ax.spines.values():
		spine.set_visible(False)
	plt.subplots_adjust(left=0, wspace=0, hspace=0)
	plt.savefig(fname + '.jpg')
	break

def resize_image_batch(imgs, im_shape,interp='bilinear' ):
    resized_imgs = np.zeros( ( imgs.shape[0],im_shape[0], im_shape[1], imgs.shape[3] ))
    for img_num in range(imgs.shape[0]):
        img = imgs[img_num]
        resized_img = imresize( img, im_shape,interp )
        resized_imgs[img_num] = resized_img 
    return resized_imgs

def factors(n):    
    return set(reduce(list.__add__, 
                ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0)))

def get_working_batch_size( len_train, len_val, len_post_val, len_test, threshold ):
    factors_train = np.sort(np.asarray( list( factors( len_test)),dtype=int))
    factors_val = np.sort(np.asarray( list( factors( len_test)),dtype=int))
    factors_post_val = np.sort(np.asarray( list( factors( len_test)),dtype=int))
    factors_test = np.sort(np.asarray( list( factors( len_test)),dtype=int))
    
    batch_size_train = np.max(factors_train[factors_train > threshold])
    batch_size_val = np.max(factors_val[factors_val > threshold])
    batch_size_post_val = np.max(factors_post_val[factors_post_val > threshold])
    batch_size_test = np.max(factors_test[factors_test > threshold])

    return batch_size_train, batch_size_val, batch_size_post_val, batch_size_test

def get_data_generators( x_train, y_train, x_val, y_val, x_test,  batch_size, seed, im_shape, test_batch_size ):

    try:
        pool.terminate()
    except:
        pass

    num_processes = 6
    pool = Pool(processes=num_processes)

    y_train = to_categorical(y_train)
    y_val = to_categorical(y_val)
    
    y_test_dummy = to_categorical( np.ones( len(x_test )))

    x_test_rz = resize_image_batch(x_test, im_shape, interp='bilinear')

    x_train = x_train / 255.
    x_val = x_val / 255.
    x_test = x_test / 255.
    x_test_rz = x_test_rz / 255.

    #batch_size_train, batch_size_val, batch_size_post_val, batch_size_test = get_working_batch_size( len(x_train), len(x_val), len(x_post_val), len(x_test), 400 )
    train_datagen = T.ImageDataGenerator(
        #rescale=1./255,
        rotation_range=40,
	shear_range=0.2,
	zoom_range=0.2,
        width_shift_range=0.2,
        height_shift_range=0.2, 
        horizontal_flip=True
	)

    if im_shape[0] % 2 == 0:
        center_crop_transform = T.center_crop
    else:
        center_crop_transform = T.center_crop_odd

    train_datagen.config['random_crop_size'] = im_shape
    train_datagen.set_pipeline([ T.random_transform, T.random_crop, T.random_color_noise, T.random_brightness ])
    train_generator = train_datagen.flow(x_train, y_train, batch_size=batch_size, seed=seed, pool=pool, shuffle=True )

    x_train_feats = copy.deepcopy( x_train )
    y_train_feats = copy.deepcopy( y_train )

    train_datagen_feats =  T.ImageDataGenerator()
    train_datagen_feats.config['center_crop_size'] = im_shape
    train_datagen_feats.config['length'] = len(y_train_feats)
    train_datagen_feats.set_pipeline([ center_crop_transform,T.random_transform ])
    #train_generator_feats = train_datagen_feats.flow(x_train_feats, y_train_feats, batch_size=len(y_train_feats), seed=seed, pool=pool, shuffle=False )
    train_generator_feats = train_datagen_feats.flow(x_train_feats, y_train_feats, batch_size=test_batch_size, seed=seed, pool=pool, shuffle=False )


    val_datagen = T.ImageDataGenerator()#rescale=1./255)
    val_datagen.config['center_crop_size'] = im_shape
    val_datagen.config['length'] = len(y_val)
    val_datagen.set_pipeline([center_crop_transform, T.random_transform])
    #val_generator = val_datagen.flow(x_val, y_val, batch_size=len(y_val), seed=seed, pool=pool , shuffle=False)
    val_generator = val_datagen.flow(x_val, y_val, batch_size=test_batch_size, seed=seed, pool=pool , shuffle=False)
    

    test_datagen = T.ImageDataGenerator()#rescale=1./255)
    test_datagen.config['center_crop_size'] = im_shape
    test_datagen.config['length'] = len(y_test_dummy)
    test_datagen.set_pipeline([center_crop_transform, T.random_transform])
    #test_generator = test_datagen.flow(x_test,y_test_dummy,  batch_size=len(y_test_dummy), seed=seed, pool=pool , shuffle=False)
    test_generator = test_datagen.flow(x_test,y_test_dummy,  batch_size=test_batch_size, seed=seed, pool=pool , shuffle=False)

    
    test_datagen_rz = T.ImageDataGenerator()#rescale=1./255)
    test_datagen_rz.config['length'] = len(y_test_dummy)
    test_datagen_rz.set_pipeline([T.random_transform])
    #test_generator_rz = test_datagen_rz.flow(x_test_rz,y_test_dummy, batch_size=len(y_test_dummy), seed=seed, pool=pool , shuffle=False)
    test_generator_rz = test_datagen_rz.flow(x_test_rz,y_test_dummy, batch_size=test_batch_size, seed=seed, pool=pool , shuffle=False)

    return train_generator, train_generator_feats, val_generator, test_generator, test_generator_rz

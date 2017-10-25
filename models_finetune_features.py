import numpy as np
import pandas as pd

from finetune_models import *
from finetune_models.densenet161_network import *
from finetune_models.densenet169_network import *
from finetune_models.resnet_101_network import *
from finetune_models.resnet_50_network import *
from finetune_models.custom_layers.scale_layer import Scale
from finetune_models.custom_layers.scale_layer import Scale

import copy
from keras import applications
from keras.models import Model
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.metrics import categorical_accuracy
#from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.callbacks import ReduceLROnPlateau


def createDenseNet169( img_rows=224, img_cols=224, channel=3, num_classes=25 ):
    model = densenet169_model(img_rows=img_rows, img_cols=img_cols, color_type=channel, num_classes=num_classes)
    model.summary()
    return model

def createDenseNet161( img_rows=224, img_cols=224, channel=3, num_classes=25 ):
    model = densenet161_model(img_rows=img_rows, img_cols=img_cols, color_type=channel, num_classes=num_classes)
    model.summary()
    return model


def createVGG16( img_rows=256, img_cols=256, channel=3, num_classes=25  ):
    base_model = applications.VGG16(weights='imagenet', include_top=False, input_shape=(img_rows, img_cols, channel))

    ## set model architechture 
    add_model = Sequential()
    add_model.add(Flatten(input_shape=base_model.output_shape[1:]))
    add_model.add(Dense(256, activation='relu'))
    add_model.add(Dropout(0.5))
    add_model.add(Dense(num_classes, activation='softmax'))

    model = Model(inputs=base_model.input, outputs=add_model(base_model.output))
    model.compile(loss='categorical_crossentropy', optimizer=optimizers.SGD(lr=1e-3, momentum=0.9), metrics=['accuracy'])

    model.summary()
    return model


def createXception( img_rows=299, img_cols=299, channel=3, num_classes=25 ):
    base_model = applications.Xception(input_shape=(img_rows, img_cols, channel), weights='imagenet', include_top=False)
    out = base_model.output
    out = GlobalAveragePooling2D()(out)
    out = Dropout(0.5)(out)
    pred_out = Dense(num_classes, activation='softmax')(out)

    # add your top layer block to your base model
    model = Model(base_model.input, pred_out)


    model.compile(loss='categorical_crossentropy', optimizer=optimizers.SGD(lr=1e-3, momentum=0.9, decay=1e-6, nesterov=True),
              metrics=['accuracy'])

    model.summary()
    return model


def createResnet50( img_rows=224, img_cols=224, channel=3, num_classes=25 ):
    model = resnet50_model(img_rows=img_rows, img_cols=img_cols, color_type=channel, num_classes=num_classes)
    model.summary()
    return model

    
def fit_model(model, train_generator, val_generator, train_steps, val_steps, epochs, savename):
    #val_steps = len(val_generator) //val_batch_size + 1
    history = model.fit_generator(
	train_generator,
        steps_per_epoch= train_steps,
        epochs=epochs,
        validation_data=val_generator,
	validation_steps=val_steps,
        callbacks = [ModelCheckpoint(savename, monitor='val_acc', save_best_only=True),
    	    	    EarlyStopping(monitor='val_acc', min_delta=0.0, patience=15, verbose=0, mode='auto'),
		    ReduceLROnPlateau(monitor='val_acc', patience=7, verbose=1, factor=0.8, min_lr=0.0001) ]
    )
    return history

def load_trained_model( model_name, savename ):

    model = None
    if model_name == 'DenseNet169' or model_name == 'DenseNet161' or model_name == 'Resnet50' or model_name == 'Resnet101':
        model = load_model(savename, custom_objects={'Scale': Scale})
    if model_name == 'VGG16' or model_name =='Xception' or model_name== 'VGG19':
        model = load_model(savename)
    return model

def predict_model( model, generator, batch_size ):
    val_steps = len(generator) //batch_size + 1
    prediction_probs = model.predict_generator( generator, val_steps )
    prediction_class = np.argmax( prediction_probs, 1 )
    return prediction_probs, prediction_class

def evaluate_model( model, generator, batch_size ):
    val_steps = len(generator) //batch_size + 1
    acc = model.evaluate_generator( generator, val_steps)[1]
    return acc

def createModel( model_name, num_classes=25 ):
    if model_name == 'DenseNet169':
        return createDenseNet169(  num_classes=num_classes )
        
    if model_name == 'DenseNet161':
        return createDenseNet161(  num_classes=num_classes )

    if model_name == 'VGG16':
        return createVGG16(  num_classes=num_classes )

    if model_name == 'Xception':
        return createXception( num_classes=num_classes )

    if model_name == 'Resnet50':
        return createResnet50( num_classes=num_classes )

def densenet_extract_features(base_model):
    print( base_model )
    feature_model = Model(input=base_model.input, output=base_model.get_layer('pool5').output)
    print( feature_model )
    feature_model.summary()
    return feature_model
 
def res50_extract_features(base_model):
    print( base_model )
    feature_model = Model(input=base_model.input, output=base_model.get_layer('pool5').output)
    print( feature_model )
    feature_model.summary()
    return feature_model

def xception_extract_features(base_model):
    print( base_model )
    return base_model
    '''
    feature_model = Model(input=base_model.input, output=base_model.get_layer('pool5').output)
    print( feature_model )
    feature_model.summary()
    return feature_model
    '''
def vgg16_extract_features(base_model):
    print( base_model )
    return base_model
    #feature_model = Model(input=base_model.input, output=base_model.get_layer('pool5').output)
    #print( feature_model )
    #feature_model.summary()
    #return feature_model


def extract_features( model_name, model ):
    if model_name == 'DenseNet169' or model_name=='DenseNet161':
        return densenet_extract_features( model )
    elif model_name == 'VGG16':
        return vgg16_extract_features( model )
    elif model_name == 'Resnet50':
        return res50_extract_features( model )
    elif model_name == 'Xception':
        return xception_extract_features( model )
    else:
        print( 'Model Not Found' )

def get_validation_metrics( model_restored, val_generator,  test_batch_size, filename ):
    val_accuracy = evaluate_model( model_restored, val_generator, test_batch_size )
    print( 'Validation Accuracy: ', val_accuracy )
    np.savetxt( './Validation/' + filename + '.txt', [val_accuracy] )

    return val_accuracy

def generate_test_submissions( model_restored, test_generator, test_generator_rz, rev_key, test_batch_size, filename  ):
    test_probs_crop, test_preds_crop =  predict_model( model_restored, test_generator, test_batch_size  )
    test_probs_rz, test_preds_rz =  predict_model( model_restored, test_generator_rz, test_batch_size  )
    
    pred_labels_crop = [rev_key[k] for k in test_preds_crop]
    pred_labels_rz = [rev_key[k] for k in test_preds_rz]
    
    test = pd.read_csv('test.csv')

    sub_crop = pd.DataFrame({'image_id':test.image_id, 'label':pred_labels_crop})
    sub_crop.to_csv('./Predictions/sub_' + filename + '_crop.csv', index=False) ## ~0.59

    sub_rz = pd.DataFrame({'image_id':test.image_id, 'label':pred_labels_rz})
    sub_rz.to_csv('./Predictions/sub_' + filename + '_rz.csv', index=False) ## ~0.59
    
    return


def get_features( model_name, model, train_generator_feats, val_generator,  test_generator, test_generator_rz, test_batch_size, filename ):

    model_features = extract_features( model_name, model )
    #del model

    train_feats, _ =  predict_model( model_features, train_generator_feats, test_batch_size  )
    val_feats, _ =  predict_model( model_features, val_generator, test_batch_size  )
    test_feats, _ =  predict_model( model_features, test_generator, test_batch_size  )
    test_feats_rz, _ =  predict_model( model_features, test_generator_rz, test_batch_size  )

    np.savetxt( './Features/' + filename + '_train_feats.txt', train_feats )
    np.savetxt( './Features/' + filename + '_val_feats.txt', val_feats )
    np.savetxt( './Features/' + filename + '_test_feats.txt', test_feats )
    np.savetxt( './Features/' + filename + '_test_feats_rz.txt', test_feats_rz )

    return



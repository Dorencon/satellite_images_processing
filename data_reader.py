import zipfile
import os
import cv2
from sklearn.model_selection import train_test_split
import glob
import csv
import numpy as np
import pandas as pd
import shutil
import subprocess
from keras.callbacks import ModelCheckpoint, EarlyStopping
import keras

from generator import generator_from_lists
from create_mask import create_mask

class TrainDataRepresentation:
    def __init__(self, train_generator, steps_per_epoch, validation_data, validation_steps, callbacks, output):
        self.train_generator = train_generator
        self.steps_per_epoch = steps_per_epoch
        self.validation_data = validation_data
        self.validation_steps = validation_steps
        self.callbacks = callbacks
        self. output = output

class DataRepresentaion:
    def __init__(self):


class DataReader:
    def __init__(self, data_sourse, output, ma_exist = False):
        self.data_sourse = data_sourse
        self.ma_exist = ma_exist
        self.output = output

    def unzip(self):
        output = os.path.join(self.data_sourse, 'decompressed')
        os.makedirs(output, exist_ok = True)
        patches = os.path.join(output, 'patches')
        mask = os.path.join(output, 'mask')
        os.makedirs(patches, exist_ok = True)
        os.makedirs(mask, exist_ok = True)
        zipfile.ZipFile(os.path.join(self.data_sourse, 'patches', 'landsat_patches.zip')).extractall(patches)
        zipfile.ZipFile(os.path.join(self.data_sourse, 'patches', 'landsat_patches.zip')).extractall(mask)
        self.data = output
        if (self.ma_exist):
            ma_data = os.path.join(self.data, 'manually_annotated')
            os.makedirs(ma_data, exist_ok = True)
            zipfile.ZipFile(os.path.join(self.data_sourse, 'patches', 'manual_annotations_patches.zip')).extractall(ma_data)

    def unzip_allinone(self):
        self.data = os.path.join(self.data_sourse, 'decompressed')
        for zip_name in os.walk(self.data_sourse)[2]:
            os.makedirs(self.data, exist_ok = True)
            patches = os.path.join(self.data, 'patches')
            mask = os.path.join(self.data, 'mask')
            os.makedirs(patches, exist_ok = True)
            os.makedirs(mask, exist_ok = True)
            zipfile.ZipFile(os.path.join(self.data_sourse, zip_name)).extractall(patches)
            meta = glob.glob(os.path.join(patches, '*MTL.txt'))
            create_mask(meta, mask)

            to_del = glob.glob(os.path.join(patches, '*.txt'))
            for file in to_del:
                os.remove(file)
            to_del = glob.glob(os.path.join(patches, '*.xml'))
            os.remove(to_del[0])

    def create_csv(self):
        MASK_ALGORITHM = 'Kumar-Roy'

        os.makedirs(os.path.join(self.data, 'csv'), exist_ok = True)
        masks = glob.glob(os.path.join(self.data, 'mask', '*{}*.tif'.format(MASK_ALGORITHM)))
        with open(os.path.join(self.data, 'images_masks.csv', 'w')) as f:
            writer = csv.writer(f, delimiter = ',')
            for mask in tqdm(masks):
                _, mask_name = os.path.split(mask)
                image_name = mask_name.replace('_{}_'.format(MASK_ALGORITHM), '_')
                writer.writerow([image_name, mask_name])
        df = pd.read_csv(os.path.join(self.data, 'images_masks.csv'))
        images_df = df[['images']]
        masks_df = df[['masks']]

        train_ratio = 0.4
        validation_ratio = 0.1
        test_ratio = 0.5
        RANDOM_STATE = 42

        x_train, x_test, y_train, y_test = train_test_split(images_df, masks_df, test_size = 1 - train_ratio,
                                                            random_state = RANDOM_STATE)
        x_val, x_test, y_val, y_test = train_test_split(x_test, y_test,
                                                        test_size = test_ratio/(test_ratio + validation_ratio),
                                                        random_state=RANDOM_STATE)
        x_train.to_csv(os.path.join(self.data, 'images_train.csv'), index=False)
        y_train.to_csv(os.path.join(self.data, 'masks_train.csv'), index=False)
        x_val.to_csv(os.path.join(self.data, 'images_val.csv'), index=False)
        y_val.to_csv(os.path.join(self.data, 'masks_val.csv'), index=False)
        x_test.to_csv(os.path.join(self.data, 'images_test.csv'), index=False)
        y_test.to_csv(os.path.join(self.data, 'masks_test.csv'), index=False)

    def get_train_data(self):
        x_train = pd.read_csv(os.path.join(self.data, 'images_train.csv'))
        y_train = pd.read_csv(os.path.join(self.data, 'masks_train.csv'))
        x_val = pd.read_csv(os.path.join(self.data, 'images_val.csv'))
        y_val = pd.read_csv(os.path.join(self.data, 'masks_val.csv'))
        x_test = pd.read_csv(os.path.join(self.data, 'images_test.csv'))
        y_test = pd.read_csv(os.path.join(self.data, 'masks_test.csv'))

        images_train = [os.path.join(self.data, 'patches', image) for image in x_train['images']]
        masks_train = [os.path.join(self.data, 'mask', mask) for mask in y_train['masks']]

        images_validation = [os.path.join(self.data, 'patches', image) for image in x_val['images']]
        masks_validation = [os.path.join(self.data, 'mask', mask) for mask in y_val['masks']]

        RANDOM_STATE = 42
        BATCH_SIZE = 16
        EARLY_STOP_PATIENCE = 5
        MODEL_NAME = 'unet'
        MASK_ALGORITHM = 'Kumar-Roy'
        CHECKPOINT_MODEL_NAME = 'checkpoint-{}-{}-epoch_{{epoch:02d}}.hdf5'.format(MODEL_NAME, MASK_ALGORITHM)
        CHECKPOINT_PERIOD = 5

        train_generator = generator_from_lists(images_train, masks_train, batch_size=BATCH_SIZE,
                                               random_state=RANDOM_STATE)
        validation_generator = generator_from_lists(images_validation, masks_validation, batch_size=BATCH_SIZE,
                                                    random_state=RANDOM_STATE)

        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=EARLY_STOP_PATIENCE)
        checkpoint = ModelCheckpoint(os.path.join(self.output, CHECKPOINT_MODEL_NAME), monitor='loss', verbose=1,
                                     save_best_only=True, mode='auto', period=CHECKPOINT_PERIOD)

        return TrainDataRepresentation(
            train_generator,
            len(images_train),
            validation_generator,
            len(images_validation),
            [checkpoint, es],
            self.output
        )
    def get_data(self):
        x_train = pd.read_csv(os.path.join(self.data, 'images_train.csv'))
        y_train = pd.read_csv(os.path.join(self.data, 'masks_train.csv'))
        x_val = pd.read_csv(os.path.join(self.data, 'images_val.csv'))
        y_val = pd.read_csv(os.path.join(self.data, 'masks_val.csv'))
        x_test = pd.read_csv(os.path.join(self.data, 'images_test.csv'))
        y_test = pd.read_csv(os.path.join(self.data, 'masks_test.csv'))
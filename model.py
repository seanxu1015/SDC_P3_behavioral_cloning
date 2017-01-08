import numpy as np
import tensorflow as tf
from keras.models import Sequential, model_from_json 
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
import keras
from keras import backend as K
import csv, os, random
from PIL import Image
import pandas as pd
tf.python.control_flow_ops = tf


class Config:
    """Set hyperparameters here.
    """
    epochs = 10
    batch_size = 200 
    dropout = 0.5
    drv_log = './data/driving_log.csv'
    model_name = 'model.json'
    save_path = 'model.h5'
    learning_rate = 1e-5
    weight_decay = 1e-5
    gpu_worker = '/gpu:0'


class BCModel:

    def __init__(self, config):
        """Construct a model.
        args: config, a Config instance for setting parameters.
        """
        self.config = config
        self.load_data()
        with tf.device(self.config.gpu_worker):
            self.create_model()

    def load_data(self):
        """read data, and split for training validation and testing.
        """
        with open(self.config.drv_log, 'r') as f:
            add_data = [] # addtional data require angle smoothing
            ori_data = [] # given data
            ddir = os.path.dirname(self.config.drv_log)
            lines = list(csv.reader(f))
            np.random.shuffle(lines)
            for line in lines:
                if line[0] == 'center':
                    continue
                ste_ang = float(line[3])
                if ste_ang == 0. and random.random() < 0.8:
                    continue
                if '12_01' in line[0]:
                    ori_data.append([os.path.join(ddir, line[0].strip()), ste_ang])
                    ori_data.append([os.path.join(ddir, line[1].strip()), ste_ang])
                    ori_data.append([os.path.join(ddir, line[2].strip()), ste_ang])
                else:
                    add_data.append([os.path.join(ddir, line[0].strip()), ste_ang])
                    add_data.append([os.path.join(ddir, line[1].strip()), ste_ang])
                    add_data.append([os.path.join(ddir, line[2].strip()), ste_ang])
        
        def angle_smoothing(data):
            """helper function for smoothing steering angles.
            reference: 
            https://carnd-forums.udacity.com/questions/24807517/tips-for-behavior-cloning
            """
            angles = np.asarray([itm[1] for itm in data])
            fwd = pd.ewma(angles, span=20)
            bwd = pd.ewma(angles[::-1], span=20)
            smooth = np.vstack((fwd, bwd[::-1]))
            smooth = np.mean(smooth, axis=0)
            angles = np.ndarray.tolist(smooth)
            smooth_data = []
            for i in range(len(data)):
                smooth_data.append([data[i][0], angles[i]])
            return smooth_data
        #all_data = ori_data + angle_smoothing(add_data) #the smoothing method didn't work well
        all_data = ori_data + add_data
        n1 = int(len(all_data) * 0.7)
        n2 = int(len(all_data) * 0.85)
        self.train = self.data_generator(all_data[:n1])
        self.valid = self.data_generator(all_data[n1:n2])
        self.test = self.data_generator(all_data[n2:])
        self.train_len = n1 // self.config.batch_size * self.config.batch_size
        self.valid_len = (n2 - n1) // self.config.batch_size * self.config.batch_size
        self.test_len = (len(all_data) - n2) // self.config.batch_size * self.config.batch_size

    def data_generator(self, data):
        """Helper function to create a image generator.
        args : data, a list of [iamge_path, streering_angle]
        yield: (rgb_images, steering_angles)
        """
        X, y = [], []
        while 1:
            np.random.shuffle(data)
            for line in data:
                img = Image.open(line[0])
                img = img.resize((32, 16))
                img = np.asarray(img, dtype=np.float32)
                img = img / 128. - 1.
                img = np.transpose(img, (2, 0, 1))                
                X.append(img)
                y.append(line[1])
                if len(X) == self.config.batch_size:
                    batch = (np.asarray(X), np.asarray(y))
                    X = []
                    y = []
                    yield batch

    def create_model(self):
        """Setup model here.
        """
        model = Sequential()
        model.add(Conv2D(32, 3, 3, input_shape=(3, 16, 32), border_mode='same',
                         W_regularizer=l2(self.config.weight_decay), 
                         b_regularizer=l2(self.config.weight_decay),
                         activation='elu'))
        model.add(MaxPooling2D((2, 2), (2, 2)))
        model.add(Dropout(self.config.dropout))
        model.add(Conv2D(64, 3, 3, border_mode='same',
                         W_regularizer=l2(self.config.weight_decay), 
                         b_regularizer=l2(self.config.weight_decay),
                         activation='elu'))
        model.add(MaxPooling2D((2, 2), (2, 2)))
        model.add(Dropout(self.config.dropout))
        model.add(Conv2D(128, 1, 3, border_mode='same', 
                         W_regularizer=l2(self.config.weight_decay), 
                         b_regularizer=l2(self.config.weight_decay),
                         activation='elu'))
        model.add(Dropout(self.config.dropout))
        model.add(Flatten())
        model.add(Dense(100, W_regularizer=l2(self.config.weight_decay), 
                        b_regularizer=l2(self.config.weight_decay),
                        activation='elu'))
        model.add(Dropout(self.config.dropout))
        model.add(Dense(50, W_regularizer=l2(self.config.weight_decay), 
                        b_regularizer=l2(self.config.weight_decay),
                        activation='elu'))
        model.add(Dropout(self.config.dropout))
        model.add(Dense(10, W_regularizer=l2(self.config.weight_decay), 
                        b_regularizer=l2(self.config.weight_decay),
                        activation='elu'))
        model.add(Dropout(self.config.dropout))
        model.add(Dense(1,  W_regularizer=l2(self.config.weight_decay), 
                        b_regularizer=l2(self.config.weight_decay)))
        model.summary()
        adam = keras.optimizers.Adam(lr=self.config.learning_rate)
        model.compile(loss='mse', optimizer='adam')
        self.model = model
                
        json_string = self.model.to_json()
        open(self.config.model_name, 'w').write(json_string)

    def train_model(self):
        """method for training
        """
        history = self.model.fit_generator(generator=self.train, 
                samples_per_epoch=self.train_len, nb_epoch=self.config.epochs,
                validation_data=self.valid, nb_val_samples=self.valid_len)
        self.model.save_weights(self.config.save_path)
        return history

    def evaluation(self):
        """method for evaluation
        """
        model = model_from_json(open(self.config.model_name).read())
        model.compile(loss='mse', optimizer='adam')
        model.load_weights(self.config.save_path)
        res = model.evaluate_generator(generator=self.test, val_samples=self.test_len)
        print('Testing Loss: {:.4f}'.format(res))
        return res
        
    def tune_model(self):
        """method for tuning model in problem areas
        """
        model = model_from_json(open(self.config.model_name).read())
        model.compile(loss='mse', optimizer='adam')
        model.load_weights(self.config.save_path)
        history = self.model.fit_generator(generator=self.train, 
                samples_per_epoch=self.train_len, nb_epoch=self.config.epochs,)
        self.model.save_weights(self.config.save_path)
        return history


if __name__ == '__main__':
    config = Config()
    bcmodel = BCModel(config)
    tf_cfg = tf.ConfigProto()
    tf_cfg.gpu_options.allow_growth = True
    with tf.Session(config=tf_cfg) as sess:
        K.set_session(sess)
        bcmodel.train_model()
        bcmodel.tune_model()
        # bcmodel.evaluation()


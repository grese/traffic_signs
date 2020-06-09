import os, shutil
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import BatchNormalization, Conv2D, MaxPooling2D, Activation, Flatten, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import plot_model
from sklearn.metrics import classification_report, confusion_matrix
from skimage import io

class TrafficSignNet:
    @staticmethod
    def build(width, height, depth, classes):
        model = Sequential()
        input_shape = (height, width, depth)
        chan_dim = -1
        # CONV => RELU => BN => POOL
        model.add(Conv2D(8, (5, 5), padding='same', input_shape=input_shape))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=chan_dim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        # first set of (CONV => RELU => CONV => RELU) * 2 => POOL
        model.add(Conv2D(16, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=chan_dim))
        model.add(Conv2D(16, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=chan_dim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        # second set of (CONV => RELU => CONV => RELU) * 2 => POOL
        model.add(Conv2D(32, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=chan_dim))
        model.add(Conv2D(32, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=chan_dim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        # first set of FC => RELU layers
        model.add(Flatten())
        model.add(Dense(128))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        # second set of FC => RELU layers
        model.add(Flatten())
        model.add(Dense(128))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        # softmax classifier
        model.add(Dense(classes))
        model.add(Activation('softmax'))
        # return the constructed network architecture
        return model

    @staticmethod
    def build_and_train(X_train, y_train, X_test, y_test, epochs=30, lr=1e-3, bs=64):
        n_labels = len(np.unique([r.tolist().index(1) for r in y_train]))
        class_totals = y_train.sum(axis=0)
        class_weights = {i:w for i,w in enumerate(class_totals.max()/class_totals)}
        # construct the image generator for data augmentation
        aug = ImageDataGenerator(
            rotation_range=10,
            zoom_range=0.15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.15,
            horizontal_flip=False,
            vertical_flip=False,
            fill_mode='nearest')
        # initialize the optimizer and compile the model
        print('[INFO] compiling model...')
        opt = Adam(lr=lr, decay=lr / (epochs * 0.5))
        model = TrafficSignNet.build(width=32, height=32, depth=3, classes=n_labels)
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        # train the network
        print('[INFO] training network...')
        history = model.fit(
            aug.flow(X_train, y_train, batch_size=bs),
            validation_data=(X_test, y_test),
            steps_per_epoch=X_train.shape[0] // bs,
            epochs=epochs,
            class_weight=class_weights,
            verbose=1)
        return model, history
    
    @staticmethod
    def evaluate(model, X_test, y_test, signs, save_dir='./artifacts/', bs=64, save=False):
        # evaluate the network
        print('[INFO] evaluating network...')
        predictions = model.predict(X_test, batch_size=bs)
        report = classification_report(y_test.argmax(axis=1), predictions.argmax(axis=1), target_names=signs.values())
        print(report)
        if save: TrafficSignNet.save_model(model, save_dir=save_dir)
        return predictions
    
    @staticmethod
    def load_model(modelpath):
        model = load_model(modelpath)

    @staticmethod
    def save_model(model, save_dir='./artifacts'):
        # save the network to disk
        save_path = os.path.join(save_dir, 'model')
        if not os.path.exists(save_dir): os.makedirs(save_dir, 0o755, exist_ok=True)
        if os.path.exists(save_path): shutil.rmtree(save_path, ignore_errors=True)
        print(f'[INFO] serializing network to {save_path}...')
        return model.save(save_path)
    
    @staticmethod
    def plot_history(model, h, out_dir='./artifacts/images', show=True, save=True, dpi=150):
        n_epochs = len(h.history['loss'])
        n = np.arange(0, n_epochs)
        plt.style.use('ggplot')
        plt.figure()
        plt.plot(n, h.history['loss'], label='train_loss')
        plt.plot(n, h.history['val_loss'], label='val_loss')
        plt.plot(n, h.history['accuracy'], label='train_acc')
        plt.plot(n, h.history['val_accuracy'], label='val_acc')
        plt.title('Training Loss and Accuracy on Dataset')
        plt.xlabel('Epoch #')
        plt.ylabel('Loss/Accuracy')
        plt.legend(loc='lower left')
        if not os.path.exists(out_dir): os.makedirs(out_dir, 0o755, exist_ok=True)
        if save: plt.savefig(os.path.join(out_dir, 'train.png'), dpi=dpi)
        if show: plt.show()
    
    @staticmethod
    def plot_network(model, out_dir='./artifacts/images', show=True, save=True, horizontal=False, nested=True, dpi=150):
        rankdir = 'LR' if horizontal else 'TB'
        outpath = os.path.join(out_dir, 'model.png')
        if not os.path.exists(out_dir): os.makedirs(out_dir, 0o755, exist_ok=True)
        if save: plot_model(model, to_file=outpath, show_shapes=True, show_layer_names=True, rankdir=rankdir, expand_nested=nested, dpi=dpi)
        if show: io.imshow(io.imread(outpath))
    
    @staticmethod
    def plot_predictions(predictions, y, signs, save_dir='./artifacts/images', size=(40,50), show=True, save=True):
        print('[INFO] creating confusion matrix...')
        cm = confusion_matrix(y_true=np.argmax(y, axis=1), y_pred=np.argmax(predictions, axis=1))
        labels = sorted(signs)
        df = pd.DataFrame(cm, labels, labels)
        plt.figure(figsize=size)
        heatmap = sns.heatmap(df, annot=True)
        if not os.path.exists(save_dir): os.makedirs(save_dir, 0o755, exist_ok=True)
        if save: heatmap.get_figure().savefig(os.path.join(save_dir, 'predictions.png'))
        if show: plt.show()

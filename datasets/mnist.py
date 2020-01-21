from keras.datasets import mnist
from keras.utils import np_utils

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.cnn1_models import cnn1_mnist_model
from models.cnn1_models_double import cnn1_double_mnist_model
from models.cnn1_models_epoch import cnn1_epoch_mnist_model
from models.cnn1_models_half import cnn1_half_mnist_model
from models.cnn2_models import cnn2_mnist_model
from models.cnn2_models_double import cnn2_double_mnist_model
from models.cnn2_models_half import cnn2_half_mnist_model
from models.cnn2_models_epoch import cnn2_epoch_mnist_model
from models.pgdtrained_models import pgdtrained_mnist_model

from models.distillation_models import distillation_mnist_model

class MNISTDataset:
    def __init__(self):
        self.dataset_name = "MNIST"
        self.image_size = 28
        self.num_channels = 1
        self.num_classes = 10

    def get_test_dataset(self):
        (X_train, y_train), (X_test, y_test) = mnist.load_data()

        X_test = X_test.reshape(X_test.shape[0], self.image_size, self.image_size, self.num_channels)
        X_test = X_test.astype('float32')
        X_test /= 255
        Y_test = np_utils.to_categorical(y_test, self.num_classes)
        del X_train, y_train
        return X_test, Y_test

    def get_train_dataset(self):
        (X_train, y_train), (X_test, y_test) = mnist.load_data()

        X_train = X_train.reshape(X_train.shape[0], self.image_size, self.image_size, self.num_channels)
        X_train = X_train.astype('float32')
        X_train /= 255
        Y_train = np_utils.to_categorical(y_train, self.num_classes)
        del X_test, y_test
        return X_train, Y_train

    def get_val_dataset(self):
        (X_train, y_train), (X_test, y_test) = mnist.load_data()

        val_size = 5000
        X_val = X_train[:val_size]
        X_val = X_val.reshape(X_val.shape[0], self.image_size, self.image_size, self.num_channels)
        X_val = X_val.astype('float32') / 255
        y_val = y_train[:val_size]
        Y_val = np_utils.to_categorical(y_val, self.num_classes)
        del X_train, y_train, X_test, y_test

        return X_val, Y_val

    def load_model_by_name(self, model_name, logits=False, input_range_type=1, pre_filter=lambda x:x):
        """
        :params logits: return logits(input of softmax layer) if True; return softmax output otherwise.
        :params input_range_type: {1: [0,1], 2:[-0.5, 0.5], 3:[-1, 1]...}
        """
        if model_name not in ["cnn2", 'cnn2_adv_trained', 'cnn1', 'pgdtrained', 'pgdbase', 'cnn1_double', 'cnn1_half', 'cnn1_30','cnn1_40', 'cnn2_half', 'cnn2_double','cnn2_30', 'cnn2_40','distillation']:
            raise NotImplementedError("Undefined model [%s] for %s." % (model_name, self.dataset_name))
        self.model_name = model_name

        model_weights_fpath = "%s_%s.keras_weights.h5" % (self.dataset_name, model_name)
        model_weights_fpath = os.path.join('downloads/trained_models', model_weights_fpath)

        # self.maybe_download_model()
        if model_name in ["cnn2", 'cnn2_adv_trained']:
            model = cnn2_mnist_model(logits=logits, input_range_type=input_range_type, pre_filter=pre_filter)
        elif model_name in ['cnn1']:
            model = cnn1_mnist_model(logits=logits, input_range_type = input_range_type, pre_filter=pre_filter)
        elif model_name in ['pgdtrained', 'pgdbase']:
            model = pgdtrained_mnist_model(logits=logits, input_range_type = input_range_type, pre_filter=pre_filter)
        elif model_name in ['cnn1_double']:
            model = cnn1_double_mnist_model(logits=logits, input_range_type=input_range_type, pre_filter=pre_filter)
        elif model_name in ['cnn1_half']:
            model = cnn1_half_mnist_model(logits=logits, input_range_type=input_range_type, pre_filter=pre_filter)
        elif model_name in ['cnn1_30']:
            model = cnn1_epoch_mnist_model(logits=logits, input_range_type=input_range_type, pre_filter=pre_filter)
        elif model_name in ['cnn1_40']:
            model = cnn1_epoch_mnist_model(logits=logits, input_range_type=input_range_type, pre_filter=pre_filter)
        elif model_name in ['cnn2_half']:
            model = cnn2_half_mnist_model(logits=logits, input_range_type=input_range_type, pre_filter=pre_filter)
        elif model_name in ['cnn2_double']:
            model = cnn2_double_mnist_model(logits=logits, input_range_type=input_range_type,
                                                  pre_filter=pre_filter)
        elif model_name in ['cnn2_30']:
            model = cnn2_epoch_mnist_model(logits=logits, input_range_type=input_range_type,
                                                 pre_filter=pre_filter)
        elif model_name in ['cnn2_40']:
            model = cnn2_epoch_mnist_model(logits=logits, input_range_type=input_range_type,
                                                 pre_filter=pre_filter)
        elif model_name in ['distillation']:
            model = distillation_mnist_model(logits=logits, input_range_type=input_range_type,
                                             pre_filter=pre_filter)
        print("\n===Defined TensorFlow model graph.")
        if model_name not in ['distillation']:
            model.load_weights(model_weights_fpath)
        print ("---Loaded MNIST-%s model.\n" % model_name)
        return model

if __name__ == '__main__':
    # from datasets.mnist import *
    dataset = MNISTDataset()
    X_test, Y_test = dataset.get_test_dataset()
    print (X_test.shape)
    print (Y_test.shape)

    model_name = 'cnn2'
    model = dataset.load_model_by_name(model_name)

    model.compile(loss='categorical_crossentropy',optimizer='sgd', metrics=['acc'])
    _,accuracy = model.evaluate(X_test, Y_test, batch_size=128)
    print ("\nTesting accuracy: %.4f" % accuracy)



#
# python main.py --dataset_name MNIST --model_name1 cnn1 --model_name2 cnn1_half --model_name3 cnn1_double --model_name4 cnn1 cnn1_30 --model_name5 cnn1_40 --model_name6 cnn2 --model_name7 cnn2_half --model_name8 cnn2_double --model_name9 cnn2_30 --model_name10 cnn2_40 --attacks "bim?eps=0.3&eps_iter=0.06;"

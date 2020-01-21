from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split

lfw_people=fetch_lfw_people(min_faces_per_person=14,color=True,slice_=(slice(61,189),slice(61,189)),resize=0.5)

from keras.utils import np_utils

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.cnn2_models import cnn2_LFW_model
from models.cnn1_models import cnn1_LFW_model

# from models.carlini_models import carlini_mnist_model
# from models.carlini_models_double import carlini_double_mnist_model
# from models.carlini_models_epoch import carlini_epoch_mnist_model
# from models.carlini_models_half import carlini_half_mnist_model
# from models.cleverhans_models import cleverhans_mnist_model
# from models.cleverhans_models_double import cleverhans_double_mnist_model
# from models.cleverhans_models_half import cleverhans_half_mnist_model
# from models.cleverhans_models_epoch import cleverhans_epoch_mnist_model
# from models.pgdtrained_models import pgdtrained_mnist_model
#
# from models.distillation_models import distillation_mnist_model

x=lfw_people.images
y=lfw_people.target

target_names=lfw_people.target_names
n_classes=target_names.shape[0]

print (x.shape)
print (y.shape)


class LFWDataset:
    def __init__(self):
        self.dataset_name = "LFW"
        self.image_size = 64
        self.num_channels = 3
        self.num_classes = n_classes

    def get_test_dataset(self):

        X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.25,shuffle=None)

        X_test = X_test.reshape(X_test.shape[0], self.image_size, self.image_size, self.num_channels)
        X_test = X_test.astype('float32')
        X_test /= 255
        Y_test = np_utils.to_categorical(y_test, self.num_classes)
        del X_train, y_train
        return X_test, Y_test

    def get_train_dataset(self):
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25,shuffle=None)

        X_train = X_train.reshape(X_train.shape[0], self.image_size, self.image_size, self.num_channels)
        X_train = X_train.astype('float32')
        X_train /= 255
        Y_train = np_utils.to_categorical(y_train, self.num_classes)
        del X_test, y_test
        return X_train, Y_train

    def get_val_dataset(self):
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25,shuffle=None)

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
            model = cnn2_LFW_model(logits=logits, input_range_type=input_range_type, pre_filter=pre_filter)
        elif model_name in ['cnn1']:
            model = cnn1_LFW_model(logits=logits, input_range_type = input_range_type, pre_filter=pre_filter)
        # elif model_name in ['pgdtrained', 'pgdbase']:
        #     model = pgdtrained_mnist_model(logits=logits, input_range_type = input_range_type, pre_filter=pre_filter)
        # elif model_name in ['carlini_double']:
        #     model = carlini_double_mnist_model(logits=logits, input_range_type=input_range_type, pre_filter=pre_filter)
        # elif model_name in ['carlini_half']:
        #     model = carlini_half_mnist_model(logits=logits, input_range_type=input_range_type, pre_filter=pre_filter)
        # elif model_name in ['carlini_30']:
        #     model = carlini_epoch_mnist_model(logits=logits, input_range_type=input_range_type, pre_filter=pre_filter)
        # elif model_name in ['carlini_40']:
        #     model = carlini_epoch_mnist_model(logits=logits, input_range_type=input_range_type, pre_filter=pre_filter)
        # elif model_name in ['cleverhans_half']:
        #     model = cleverhans_half_mnist_model(logits=logits, input_range_type=input_range_type, pre_filter=pre_filter)
        # elif model_name in ['cleverhans_double']:
        #     model = cleverhans_double_mnist_model(logits=logits, input_range_type=input_range_type,
        #                                           pre_filter=pre_filter)
        # elif model_name in ['cleverhans_30']:
        #     model = cleverhans_epoch_mnist_model(logits=logits, input_range_type=input_range_type,
        #                                          pre_filter=pre_filter)
        # elif model_name in ['cleverhans_40']:
        #     model = cleverhans_epoch_mnist_model(logits=logits, input_range_type=input_range_type,
        #                                          pre_filter=pre_filter)
        # elif model_name in ['distillation']:
        #     model = distillation_mnist_model(logits=logits, input_range_type=input_range_type,
        #                                      pre_filter=pre_filter)
        print("\n===Defined TensorFlow model graph.")
        if model_name not in ['distillation']:
            model.load_weights(model_weights_fpath)
        print ("---Loaded LFW-%s model.\n" % model_name)
        return model

if __name__ == '__main__':
    # from datasets.LFW import *
    dataset = LFWDataset()
    X_test, Y_test = dataset.get_test_dataset()
    print (X_test.shape)
    print (Y_test.shape)

    model_name = 'cnn2'
    model = dataset.load_model_by_name(model_name)

    model.compile(loss='categorical_crossentropy',optimizer='sgd', metrics=['acc'])
    _,accuracy = model.evaluate(X_test, Y_test, batch_size=128)
    print ("\nTesting accuracy: %.4f" % accuracy)



#
# python main.py --dataset_name MNIST --model_name1 carlini --model_name2 carlini_half --model_name3 carlini_double --model_name4 carlini carlini_30 --model_name5 carlini_40 --model_name6 cleverhans --model_name7 cleverhans_half --model_name8 cleverhans_double --model_name9 cleverhans_30 --model_name10 cleverhans_40 --attacks "bim?eps=0.3&eps_iter=0.06;"

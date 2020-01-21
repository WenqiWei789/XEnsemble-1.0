from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import pdb
import pickle
import time
import random

import keras
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import app
from tensorflow.python.platform import flags

FLAGS = flags.FLAGS

flags.DEFINE_string('dataset_name', 'MNIST', 'Supported: MNIST, CIFAR-10, ImageNet.')
flags.DEFINE_string('model_name', 'cleverhans', 'Supported: cleverhans, cleverhans_adv_trained and carlini for MNIST; carlini and DenseNet for CIFAR-10;  ResNet50, VGG19, Inceptionv3 and MobileNet for ImageNet.')

flags.DEFINE_string('input_verifier', '', 'If defined, turn on the input verifier, otherwise these is only output verifier or minimal robustness.')

flags.DEFINE_string('attacks','', 'choose from FGSM?eps=0.1;BIM?eps=0.1&eps_iter=0.02;JSMA?targeted=next;CarliniL2?targeted=next&batch_size=100&max_iterations=1000;CarliniL2?targeted=next&batch_size=100&max_iterations=1000&confidence=2, Attack name and parameters in URL style, separated by semicolon.')
flags.DEFINE_float('clip', -1, 'L-infinity clip on the adversarial perturbations.')
flags.DEFINE_boolean('visualize', True, 'Output the image examples for each attack, enabled by default.')

FLAGS.model_name =FLAGS.model_name.lower()




def load_tf_session():
    # Set TF random seed to improve reproducibility
    tf.set_random_seed(1234)

    # Create TF session and set as Keras backend session
    sess = tf.Session()
    keras.backend.set_session(sess)
    print("Created TensorFlow session and set Keras backend.")
    return sess


def main(argv=None):
    # 0. Select a dataset.
    from datasets import MNISTDataset, CIFAR10Dataset, ImageNetDataset, LFWDataset
    from datasets import get_correct_prediction_idx, evaluate_adversarial_examples, calculate_mean_confidence, calculate_accuracy
    from utils.parameter_parser import parse_params

    if FLAGS.dataset_name == "MNIST":
        dataset = MNISTDataset()
    elif FLAGS.dataset_name == "CIFAR-10":
        dataset = CIFAR10Dataset()
    elif FLAGS.dataset_name == "ImageNet":
        dataset = ImageNetDataset()
    elif FLAGS.dataset_name == "LFW":
        dataset = LFWDataset()


    # 1. Load a dataset.
    print ("\n===Loading %s data..." % FLAGS.dataset_name)
    if FLAGS.dataset_name == 'ImageNet':
        if FLAGS.model_name == 'inceptionv3':
            img_size = 299
        else:
            img_size = 224
        X_test_all, Y_test_all = dataset.get_test_data(img_size, 0, 200)
    else:
        X_test_all, Y_test_all = dataset.get_test_dataset()


    # 2. Load a trained model.

    keras.backend.set_learning_phase(0)

    with tf.variable_scope(FLAGS.model_name):
        """
        Create a model instance for prediction.
        The scaling argument, 'input_range_type': {1: [0,1], 2:[-0.5, 0.5], 3:[-1, 1]...}
        """
        model = dataset.load_model_by_name(FLAGS.model_name, logits=False, input_range_type=1)
        model.compile(loss='categorical_crossentropy',optimizer='sgd', metrics=['acc'])

    X_train_all, Y_train_all = dataset.get_train_dataset()
    if FLAGS.model_name in ['resnet20','resnet32', 'resnet44', 'resnet56', 'resnet110'] and FLAGS.dataset_name=='CIFAR-10':
        # for resnet
        x_train_mean = np.mean(X_train_all, axis=0)
        X_test_all -= x_train_mean



    # 3. Evaluate the trained model.
    print ("Evaluating the pre-trained model...")
    Y_pred_all = model.predict(X_test_all)
    mean_conf_all = calculate_mean_confidence(Y_pred_all, Y_test_all)
    accuracy_all = calculate_accuracy(Y_pred_all, Y_test_all)
    print('Test accuracy on raw legitimate examples %.4f' % (accuracy_all))
    print('Mean confidence on ground truth classes %.4f' % (mean_conf_all))


    if FLAGS.attacks:
        from attacks import load_attack_input

        #attack_string = filter(lambda x: len(x) > 0, FLAGS.attacks.lower())
        attack_string = FLAGS.attacks.lower()


        correct_idx = get_correct_prediction_idx(Y_pred_all, Y_test_all)
        selected_idx = correct_idx[:100]


        X_test_all = load_attack_input(FLAGS.dataset_name, attack_string)
        Y_test_all = Y_test_all[selected_idx]



    # 4. Input denoise defense.

    input_verified = X_test_all

    # input verifier
    if FLAGS.input_verifier != '':
        from input_verifier_method import get_inverifier_by_name


        inverifier_names = [ele.strip() for ele in FLAGS.input_verifier.split(';') if ele.strip() != '']

        for inverifier_name in inverifier_names:

            inverifier = get_inverifier_by_name(inverifier_name, 'python')
            input_verified = np.append(input_verified, inverifier(X_test_all),axis=0)



    iter = input_verified.shape[0]/X_test_all.shape[0]
    batch_iter = X_test_all.shape[0]

    Y_pred= model.predict(input_verified[0:X_test_all.shape[0]])

    output = model.predict(input_verified[0:X_test_all.shape[0]])
    for i in range(int(iter)):
        if i>0:
            output=np.append(output,model.predict(input_verified[i*batch_iter:(i+1)*batch_iter]),axis=0)

            Y_pred= Y_pred + model.predict(input_verified[i*batch_iter:(i+1)*batch_iter])

    Y_pred_inverified = Y_pred/iter

    from datasets.datasets_utils import calculate_msqueezer_accuracy

    avg = np.zeros(( Y_pred_inverified.shape[0], dataset.num_classes))
    for idx in range( Y_pred_inverified.shape[0]):

        if np.max( Y_pred_inverified[idx]) >= 0.6:
            avg[idx] =  Y_pred_inverified[idx]
        elif np.max( Y_pred_inverified[idx]) < 0.6:
            avg[idx] =  Y_pred_inverified[idx] + 1

    accuracy, _, alert_bad = calculate_msqueezer_accuracy(avg, Y_test_all)

    print("Conf-L1 Test accuracy is of %.4f, where correct pred: %.4f, detection: %.4f of the input verifier layer" % (accuracy+ alert_bad / Y_pred_inverified.shape[0],accuracy, alert_bad / Y_pred_inverified.shape[0]))

    accuracy = calculate_accuracy(Y_pred_inverified, Y_test_all)

    print('Majority Voting Test accuracy %.4f' % (accuracy))


if __name__ == '__main__':
    main()




#python input_denoising_portal.py --dataset_name MNIST --model_name CNN1 --attacks "fgsm?eps=0.3" --input_verifier "bit_depth_1;median_filter_2_2;rotation_-6"
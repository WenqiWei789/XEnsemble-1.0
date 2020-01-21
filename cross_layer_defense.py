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

flags.DEFINE_string('input_verifier', '', 'need at least 3 input denoising method.')
flags.DEFINE_string('output_verifier', '', 'need at least 3 models.')

flags.DEFINE_string('attacks','', 'choose from FGSM?eps=0.1;BIM?eps=0.1&eps_iter=0.02;JSMA?targeted=next;CarliniL2?targeted=next&batch_size=100&max_iterations=1000;CarliniL2?targeted=next&batch_size=100&max_iterations=1000&confidence=2, Attack name and parameters in URL style, separated by semicolon.')
flags.DEFINE_float('clip', -1, 'L-infinity clip on the adversarial perturbations.')
flags.DEFINE_boolean('visualize', True, 'Output the image examples for each attack, enabled by default.')

FLAGS.model_name =FLAGS.model_name.lower()
FLAGS.output_verifier =FLAGS.output_verifier.lower()



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




    if FLAGS.output_verifier != '' and FLAGS.attacks != '':
        for ele in FLAGS.output_verifier.split(';'):
          if ele in ['resnet20','resnet32', 'resnet44', 'resnet56', 'resnet110'] and FLAGS.dataset_name=='CIFAR-10' and FLAGS.model_name not in ['resnet20','resnet32', 'resnet44', 'resnet56', 'resnet110']:
              x_train_mean = np.mean(X_train_all, axis=0)
              X_test_all -= x_train_mean
              break



    # 4. XEnsemble defense.

    input_verified = X_test_all

    # input verifier
    if FLAGS.input_verifier != '':
        from input_verifier_method import get_inverifier_by_name


        inverifier_names = [ele.strip() for ele in FLAGS.input_verifier.split(';') if ele.strip() != '']

        for inverifier_name in inverifier_names:

            inverifier = get_inverifier_by_name(inverifier_name, 'python')
            input_verified = np.append(input_verified, inverifier(X_test_all),axis=0)


    if  FLAGS.output_verifier == '':

        iter = input_verified.shape[0]/X_test_all.shape[0]
        batch_iter = X_test_all.shape[0]

        Y_pred= model.predict(input_verified[0:X_test_all.shape[0]])

        output = model.predict(input_verified[0:X_test_all.shape[0]])
        for i in range(int(iter)):
            if i>0:
                output=np.append(output,model.predict(input_verified[i*batch_iter:(i+1)*batch_iter]),axis=0)

                Y_pred= Y_pred + model.predict(input_verified[i*batch_iter:(i+1)*batch_iter])

        Y_pred_inverified = Y_pred/iter   ####TODO Only majority voting is provided here

        from datasets.datasets_utils import calculate_msqueezer_accuracy

        avg = np.zeros((Y_pred_inverified.shape[0], dataset.num_classes))
        for idx in range(Y_pred_inverified.shape[0]):

            if np.max(Y_pred_inverified[idx]) >= 0.6:
                avg[idx] = Y_pred_inverified[idx]
            elif np.max(Y_pred_inverified[idx]) < 0.6:
                avg[idx] = Y_pred_inverified[idx] + 1

        accuracy, _, alert_bad = calculate_msqueezer_accuracy(avg, Y_test_all)

        print(
            "Conf-L1 Test accuracy is of %.4f, where correct pred: %.4f, detection: %.4f of the input verifier layer" % (
            accuracy + alert_bad / Y_pred_inverified.shape[0], accuracy, alert_bad / Y_pred_inverified.shape[0]))

        accuracy = calculate_accuracy(Y_pred_inverified, Y_test_all)

        print('Majority Voting Test accuracy %.4f' % (accuracy))


    # outputput verifier
    if FLAGS.output_verifier != '':

        Y_pred_model_verified = np.zeros((X_test_all.shape[0], dataset.num_classes))

        model_verifier_names = [ele.strip() for ele in FLAGS.output_verifier.split(';') if ele.strip() != '']
        selected_model_verifier_names = model_verifier_names

        size_base = len(model_verifier_names)
        size_team = size_base

        prediction_base = np.zeros((size_base, Y_test_all.shape[0], Y_test_all.shape[1]))
        prediction_base_train = np.zeros((size_base, 5000, Y_train_all.shape[1]))


        model_list = range(size_base)
        for i, model_verifier_name in enumerate(model_verifier_names):
            model_verifier = dataset.load_model_by_name(model_verifier_name, logits=False, input_range_type=1)

            prediction_base[i] = model_verifier.predict(X_test_all)

            locals()['model_verifier' + str(i)] = dataset.load_model_by_name(model_verifier_name, logits=False, input_range_type=1)
            prediction_base_train[i] = model_verifier.predict(X_train_all[:5000])

        model_list = [0,1,2]


        selected_model_verifier_names = []
        for i in range(len(model_list)):
            selected_model_verifier_names.append(model_verifier_names[i])



        #ensemble on selected models
        for m,model_verifier_name in enumerate(selected_model_verifier_names):
            model_verifier = dataset.load_model_by_name(model_verifier_name, logits=False, input_range_type=1)
            model_verifier.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['acc'])

            iter = input_verified.shape[0] / X_test_all.shape[0]
            batch_iter = X_test_all.shape[0]


            Y_pred = model_verifier.predict(input_verified[0:X_test_all.shape[0]])

            #output = model_verifier.predict(input_verified[0:X_test_all.shape[0]])


            for i in range(int(iter)):
                if i>0:
                    #output=np.append(output,model_verifier.predict(input_verified[i*batch_iter:(i+1)*batch_iter]),axis=0)


                    Y_pred= Y_pred + model_verifier.predict(input_verified[i*batch_iter:(i+1)*batch_iter])

            majority_weight=np.ones(size_team)
            weighted_pred=np.zeros((size_team,5000,Y_train_all.shape[1]))


            Y_pred_model_verified = Y_pred_model_verified + majority_weight[m]*Y_pred/iter


        Y_pred_model_verified = Y_pred_model_verified /np.sum(majority_weight)


        from datasets.datasets_utils import calculate_msqueezer_accuracy

        avg = np.zeros((Y_pred_model_verified.shape[0], dataset.num_classes))
        for idx in range(Y_pred_model_verified.shape[0]):

            if np.max(Y_pred_model_verified[idx]) >= 0.6:
                avg[idx] = Y_pred_model_verified[idx]
            elif np.max(Y_pred_model_verified[idx]) < 0.6:
                avg[idx] = Y_pred_model_verified[idx] + 1

        accuracy, _, alert_bad = calculate_msqueezer_accuracy(avg, Y_test_all)


        print("Conf-L1 Test accuracy is of %.4f, where correct pred: %.4f, detection: %.4f" % (accuracy + alert_bad / Y_pred_model_verified.shape[0], accuracy, alert_bad / Y_pred_model_verified.shape[0]))


        accuracy = calculate_accuracy(Y_pred_model_verified, Y_test_all)

        print('Majority Voting Test accuracy %.4f' % (accuracy))


        #comparison

        try:
            #Adversarial training
            model_advt = dataset.load_model_by_name('cnn2_adv_trained', logits=False, input_range_type=1)
            pred_advt = model_advt(X_test_all)
            accuracy, _, alert_bad = calculate_msqueezer_accuracy(pred_advt, Y_test_all)
            print('Adversarial training Test accuracy %.4f' % (accuracy))

            # Defensive Distillation
            model_dd = dataset.load_model_by_name('distillation', logits=False, input_range_type=1)
            pred_dd = model_dd(X_test_all)
            accuracy, _, alert_bad = calculate_msqueezer_accuracy(pred_dd, Y_test_all)
            print('Defensive Distillation Test accuracy %.4f' % (accuracy))


            # Input transformation
            if FLAGS.dataset_name == 'MNIST':
                ensembles_size = 10
                crop_size = 24
                image_size = 28

            if FLAGS.dataset_name == 'CIFAR-10':
                ensembles_size = 10
                crop_size = 28
                image_size = 32


                start_max = image_size - crop_size

                ensembles_def_pred = 0
                for i in xrange(ensembles_size):
                    start_x = np.random.randint(0, start_max)
                    start_y = np.random.randint(0, start_max)
                    # boxes = [[start_y, start_x, start_y + frac, start_x + frac]]
                    X_test_all_crop = X_test_all[:, start_x:start_x + crop_size, start_y:start_y + crop_size, :]

                    if FLAGS.dataset_name == 'MNIST':
                        X_test_all_rescale = np.zeros((X_test_all.shape[0], 28, 28, 1))
                    if FLAGS.dataset_name == 'CIFAR-10':
                        X_test_all_rescale = np.zeros((X_test_all.shape[0], 32, 32, 3))
                    for i in xrange(X_test_all_crop.shape[0]):
                        X_test_all_rescale[i] = rescale(X_test_all_crop[i], np.float(image_size) / crop_size)
                    X_test_all_discret_rescale = reduce_precision_py(X_test_all_rescale,
                                                                     256)  # need to put input into the ensemble
                    pred = model.predict(X_test_all_discret_rescale)

                    ensembles_def_pred = ensembles_def_pred + pred

                Y_defend_all = ensembles_def_pred / ensembles_size

                # All data should be discretized to uint8.

                X_test_all_discret = reduce_precision_py(X_test_all, 256)
                Y_test_all_discret_pred = model.predict(X_test_all_discret)
                accuracy, _, alert_bad = calculate_msqueezer_accuracy( Y_test_all_discret_pred , Y_test_all)
                print('Input transformation ensemble Test accuracy %.4f' % (accuracy))



        except:
            raise




if __name__ == '__main__':
    main()




#python cross_layer_defense.py --dataset_name MNIST --model_name cnn1 --attacks "fgsm?eps=0.3" --input_verifier "bit_depth_1;median_filter_2_2;rotation_-6" --output_verifier "cnn2;cnn1_half;cnn1_double;cnn1_30;cnn1_40"
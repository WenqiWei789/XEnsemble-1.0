from future.standard_library import install_aliases
install_aliases()
from urllib import parse as urlparse

import pickle
import numpy as np
import os
import time



from .cleverhans_wrapper import generate_fgsm_examples, generate_jsma_examples, generate_bim_examples
from .carlini_wrapper import generate_carlini_l2_examples, generate_carlini_li_examples, generate_carlini_l0_examples
from .deepfool_wrapper import generate_deepfool_examples, generate_universal_perturbation_examples
from .adaptive.adaptive_adversary import generate_adaptive_carlini_l2_examples
from .pgd.pgd_wrapper import generate_pgdli_examples


# TODO: replace pickle with .h5 for Python 2/3 compatibility issue.
def maybe_generate_adv_examples(sess, model, x, y, X, Y, attack_name, attack_params, use_cache=False, verbose=True, attack_log_fpath=None):
    x_adv_fpath = use_cache
    if use_cache and os.path.isfile(x_adv_fpath):
        print ("Loading adversarial examples from [%s]." % os.path.basename(x_adv_fpath))
        X_adv, duration = pickle.load(open(x_adv_fpath, "rb"))
    else:
        time_start = time.time()
        X_adv = generate_adv_examples(sess, model, x, y, X, Y, attack_name, attack_params, verbose, attack_log_fpath)
        duration = time.time() - time_start

        if not isinstance(X_adv, np.ndarray):
            X_adv, aux_info = X_adv
        else:
            aux_info = {}

        aux_info['duration'] = duration

        if use_cache:
            pickle.dump((X_adv, aux_info), open(x_adv_fpath, 'wb'))
    return X_adv, duration


def generate_adv_examples(sess, model, x, y, X, Y, attack_name, attack_params, verbose, attack_log_fpath):
    if attack_name == 'none':
        return X
    elif attack_name == 'fgsm':
        generate_adv_examples_func = generate_fgsm_examples
    elif attack_name == 'jsma':
        generate_adv_examples_func = generate_jsma_examples
    elif attack_name == 'bim':
        generate_adv_examples_func = generate_bim_examples
    elif attack_name == 'carlinil2':
        generate_adv_examples_func = generate_carlini_l2_examples
    elif attack_name == 'carlinili':
        generate_adv_examples_func = generate_carlini_li_examples
    elif attack_name == 'carlinil0':
        generate_adv_examples_func = generate_carlini_l0_examples
    elif attack_name == 'deepfool':
        generate_adv_examples_func = generate_deepfool_examples
    elif attack_name == 'unipert':
        generate_adv_examples_func = generate_universal_perturbation_examples
    elif attack_name == 'adaptive_carlini_l2':
        generate_adv_examples_func = generate_adaptive_carlini_l2_examples
    elif attack_name == 'pgdli':
        generate_adv_examples_func = generate_pgdli_examples
    else:
        raise NotImplementedError("Unsuported attack [%s]." % attack_name)

    X_adv = generate_adv_examples_func(sess, model, x, y, X, Y, attack_params, verbose, attack_log_fpath)

    return X_adv





# TODO: replace pickle with .h5 for Python 2/3 compatibility issue.
def load_attack_input(dataset_name, attack_name):
    if attack_name.split('?')[0] in ['fgsm','bim','carlinili','deepfool','carlinil2','carlinil0','jsma','pgdli']:

        print ("Loading adversarial examples from [%s]." % os.path.basename(attack_name))

        if dataset_name == 'MNIST':
            attack_folder = './results/MNIST_100_1d1b8_carlini/adv_examples/'


            x_adv_fname="%s_%s.pickle" % ('MNIST_100_1d1b8_carlini', attack_name)
            x_adv_fpath = os.path.join(attack_folder, x_adv_fname)


        if dataset_name == 'CIFAR-10':
            attack_folder = './results/CIFAR-10_100_de671_densenet/adv_examples/'
            x_adv_fname="%s_%s.pickle" % ('CIFAR-10_100_de671_densenet', attack_name)
            x_adv_fpath = os.path.join(attack_folder, x_adv_fname)

        if dataset_name == 'ImageNet':
            attack_folder = './results/ImageNet_100_a2749_mobilenet/adv_examples/'
            x_adv_fname = "%s_%s.pickle" % ('ImageNet_100_a2749_mobilenet', attack_name)
            x_adv_fpath = os.path.join(attack_folder, x_adv_fname)

        if dataset_name == 'LFW':
            attack_folder = './results/LFW_100_57af0_carlini_random/adv_examples/'
            x_adv_fname = "%s_%s.pickle" % ('LFW_100_57af0_carlini', attack_name)
            x_adv_fpath = os.path.join(attack_folder, x_adv_fname)


        X_attack,_ = pickle.load(open(x_adv_fpath, "rb"),encoding='bytes')

    elif attack_name == 'ood':

        if dataset_name == 'CIFAR-10':
            ood = []
            import cv2
            ood_dir = './datasets/Imagenet_resize/Imagenet_resize/'
            #ood_dir = './datasets/LSUN_resize/LSUN_resize/'
            for file in os.listdir(ood_dir):
                ood.append(cv2.resize(cv2.imread(os.path.join(ood_dir, file)), (32, 32)))

            ood = np.asarray(ood)
            X_attack = ood / 255.0

        else:
            raise NotImplementedError('Prediction model on dataset %s is unsupported. Only CIFAR-10 as in-distribtion dataset is supported.' % attack_name)


    else:
        raise NotImplementedError('Unsupported attack method or attack results not saved: %s'%attack_name)

    return X_attack
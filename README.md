## Introduction
XEnsemble is an advanced robust deep learning package that can defend both adversarial examples and out-of-distribution input(to-be-updated). The intuition behind is the input and model divergence of these attack inputs[1,5]. 

The code package has the following portals:
1. The attack portal(main_attack_portal.py): generate and save adversarial examples.
2. The input denoising robust prediction portal(input_denoising_portal.py): given an input, generate multiple denoised variants and feed them to the target model for prediction.
3. The input-model cross-layer defense portal(cross_layer_defense.py): given an input, generate multiple denoised variants and feed them to multiple diverse models for prediction.(detailed generation of diverse models can be found in [2,4]). We also compare our performance with four adversarial defenses: adversarial training, defensive distillation, input transformation ensemble as provided in the paper.
4. Comparison portal with detection-only adversarial defenses(detection_only_comparison.py): generate defense results of Feature Squeezing, MagNet, and LID. 

XEnsemble now supports four datasets: MNIST, CIFAR-10, ImageNet and LFW.

## How to run 
1. python>=3.6, 
ensorflow-gpu==1.14.0,
keras==2.2.4,
numpy==1.13.3,
matplotlib,
h5py,
pillow,
scikit-learn,
click,
future,
opencv-python,
tinydb

2. main_attack_portal.py: please read the ppt file for more details of attacks.

```
python main_attack_portal.py --dataset_name MNIST --model_name CNN1 --attacks
"fgsm?eps=0.3;bim?eps=0.3&eps_iter=0.06;deepfool?overshoot=10;pgdli?eps=0.3;
fgsm?eps=0.3&targeted=most;fgsm?eps=0.3&targeted=next;fgsm?eps=0.3&targeted=ll;
bim?eps=0.3&eps_iter=0.06&targeted=most;
bim?eps=0.3&eps_iter=0.06&targeted=next;
bim?eps=0.3&eps_iter=0.06&targeted=ll;
carlinili?targeted=most&batch_size=1&max_iterations=1000&confidence=10;
carlinili?targeted=next&batch_size=1&max_iterations=1000&confidence=10;
carlinili?targeted=ll&batch_size=1&max_iterations=1000&confidence=10;
carlinil2?targeted=most&batch_size=100&max_iterations=1000&confidence=10;
carlinil2?targeted=next&batch_size=100&max_iterations=1000&confidence=10;
carlinil2?targeted=ll&batch_size=100&max_iterations=1000&confidence=10;
carlinil0?targeted=most&batch_size=1&max_iterations=1000&confidence=10;
carlinil0?targeted=next&batch_size=1&max_iterations=1000&confidence=10;
carlinil0?targeted=ll&batch_size=1&max_iterations=1000&confidence=10;
jsma?targeted=most;
jsma?targeted=next;
jsma?targeted=ll;"
```
3. input_denoising_portal.py: please read the ppt file for more details of the available input denoising method.
```
python input_denoising_portal.py --dataset_name MNIST --model_name CNN1 --attacks "fgsm?eps=0.3" --input_verifier "bit_depth_1;median_filter_2_2;rotation_-6"
```
4. cross_layer_defense.py:please read the ppt file for more details of available choice of models. More diversity ensemble details can be found in the paper.
```
python cross_layer_defense.py --dataset_name MNIST --model_name cnn1 --attacks "fgsm?eps=0.3" --input_verifier "bit_depth_1;median_filter_2_2;rotation_-6" --output_verifier "cnn2;cnn1_half;cnn1_double;cnn1_30;cnn1_40"
```

5. detection_only_comparison.py: please read feature squeezing, MagNet, and LID papers for implementation details.
```
python detection_only_comparison.py --dataset_name MNIST --model_name CNN1 --attacks "fgsm?eps=0.3;bim?eps=0.3&eps_iter=0.06;carlinili?targeted=next&batch_size=1&max_iterations=1000&confidence=10;carlinili?targeted=ll&batch_size=1&max_iterations=1000&confidence=10;carlinil2?targeted=next&batch_size=100&max_iterations=1000&confidence=10;carlinil2?targeted=ll&batch_size=100&max_iterations=1000&confidence=10;carlinil0?targeted=next&batch_size=1&max_iterations=1000&confidence=10;carlinil0?targeted=ll&batch_size=1&max_iterations=1000&confidence=10;jsma?targeted=next;jsma?targeted=ll;" --detection "FeatureSqueezing?squeezers=bit_depth_1&distance_measure=l1&fpr=0.05;FeatureSqueezing?squeezers=bit_depth_2&distance_measure=l1&fpr=0.05;FeatureSqueezing?squeezers=bit_depth_1,median_filter_2_2&distance_measure=l1&fpr=0.05;MagNet"
```
                 

## XEnsemble project
We are continuing the development and there is ongoing work in our lab regarding adversarial attacks and defenses. If you would like to contribute to this project, please contact [Wenqi Wei](https://www.cc.gatech.edu/~wwei66/). 

If you use our code, you are encouraged to cite:
```
[1]@article{wei2018adversarial,
  title={Adversarial examples in deep learning: Characterization and divergence},
  author={Wei, Wenqi and Liu, Ling and Loper, Margaret and Truex, Stacey and Yu, Lei and Gursoy, Mehmet Emre and Wu, Yanzhao},
  journal={arXiv preprint arXiv:1807.00051},
  year={2018}
}


[2]@inproceedings{liu2019deep,
  title={Deep Neural Network Ensembles against Deception: Ensemble Diversity, Accuracy and Robustness},
    author={Liu, Ling and Wei, Wenqi and Chow, Ka-Ho and Loper, Margaret and Gursoy, Mehmet Emre and Truex, Stacey and Wu, Yanzhao},
  booktitle={The 16th IEEE International Conference on Mobile Ad-Hoc and Smart Systems.},
year={2019},
  publisher = {IEEE},
  address = {}
}


[3]@inproceedings{chow2019denoising,
  title={Denoising and Verification Cross-Layer Ensemble Against Black-box Adversarial Attacks," IEEE International Conference on Big Data},
  author={Chow, Ka-Ho and Wei, Wenqi and Wu, Yanzhao and Liu, Ling},
  booktitle={Proceedings of the 2019 IEEE International Conference on Big Data},
  year={2019},
  organization={IEEE}
}


[4]@inproceedings{wei2020cross,
  title={Cross-Layer Strategic Ensemble Defense Against Adversarial Examples.},
  author={Wei, Wenqi and Liu, Ling and Loper, Margaret and Chow, Ka-Ho and Gursoy, Mehmet Emre and Truex, Stacey and Wu, Yanzhao},
  booktitle={International Conference on Computing, Networking and Communications(ICNC)},
  year={2020}
}
```

We have another two papers under review.

```
[5]Wenqi Wei, Ling Liu, Margaret Loper, Mehmet Emre Gursoy, Stacey Truex, Lei Yu, and Yanzhao Wu, "Demystifying Adversarial Examples and Their Adverse Effect on Deep Learning", under the submission of IEEE Transaction on Dependable and Secure Computing.
[6]Wenqi Wei, and Ling Liu, "Robust Deep Learning Ensemble against Deception", under the submission of IEEE Transaction on Dependable and Secure Computing.
```

## Special Acknowledgement
The code package is built on top of the EvadeML. We specially thank the authors in [7].

[7]  W. Xu, D. Evans, and Y. Qi, “Feature Squeezing: Detecting Adversarial Examples in Deep Neural Networks,” in
Proceedings of the 2018 Network and Distributed Systems Security Symposium (NDSS), 2018

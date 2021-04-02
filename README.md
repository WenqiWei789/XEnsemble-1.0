## Introduction
XEnsemble is an advanced robust deep learning package that can defend both adversarial examples and out-of-distribution input. The intuition behind is the input and model divergence of these attack inputs. 

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

Our XEnsemble idea work have generated a number of publications on deception input characteration, deception mitigation for deep learning and ensemble methods.


### XEnsemeble as a defense for adversarial example & OOD inputs
```
- Wei, Wenqi, and Ling Liu. "Robust Deep Learning Ensemble against Deception." IEEE Transactions on Dependable and Secure Computing (2020).

- Wei, Wenqi, Ling Liu, Margaret Loper, Ka-Ho Chow, Emre Gursoy, Stacey Truex, and Yanzhao Wu. "Cross-layer strategic ensemble defense against adversarial examples." In 2020 International Conference on Computing, Networking and Communications (ICNC), pp. 456-460. IEEE, 2020.

- Liu, Ling, Wenqi Wei, Ka-Ho Chow, Margaret Loper, Emre Gursoy, Stacey Truex, and Yanzhao Wu. "Deep neural network ensembles against deception: Ensemble diversity, accuracy and robustness." In 2019 IEEE 16th International Conference on Mobile Ad Hoc and Sensor Systems (MASS), pp. 274-282. IEEE, 2019.


- Chow, Ka-Ho, Wenqi Wei, Yanzhao Wu, and Ling Liu. "Denoising and verification cross-layer ensemble against black-box adversarial attacks." In 2019 IEEE International Conference on Big Data (Big Data), pp. 1282-1291. IEEE, 2019.
```

### Characterization of adversarial example
```
- Wei, Wenqi, Ling Liu, Margaret Loper, Ka-Ho Chow, Mehmet Emre Gursoy, Stacey Truex, and Yanzhao Wu. "Adversarial Deception in Deep Learning: Analysis and Mitigation." In 2020 Second IEEE International Conference on Trust, Privacy and Security in Intelligent Systems and Applications (TPS-ISA), pp. 236-245. IEEE, 2020.


- Wei, Wenqi, Ling Liu, Margaret Loper, Stacey Truex, Lei Yu, Mehmet Emre Gursoy, and Yanzhao Wu. "Adversarial examples in deep learning: Characterization and divergence." arXiv preprint arXiv:1807.00051 (2018).
```

### Ensemble methodology
```
- Yanzhao Wu, Ling Liu, Zhongwei Xie, Ka-Ho Chow, and Wenqi Wei. "Boosting Ensemble Accuracy by Revisiting Ensemble Diversity Metrics", IEEE Conference on Computer Vision and Pattern Recognition (CVPR 2021), 2021
```

## Acknowledgement
The code package is built on top of the EvadeML. We specially thank the authors. We also thank authors in Cleverhans, Carlini&Wagner attacks, PGD attacks, MagNet, universal(and DeepFool) attacks, keras models and those impletmented neural network models with trained weights.

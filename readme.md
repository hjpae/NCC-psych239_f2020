Recognizing Causal Features Using Machine Learning
=============

## Introduction 
&nbsp;&nbsp;*What came first, the chicken or the egg?* Inferring proper causality of a given situation has been an old riddle from every domain of science. It is crucial to discover proper causal relationship more than just correlation to give robust explanation on why such things happen in the real world, however, is intricate to pin down particulars. 

&nbsp;&nbsp;Emerging machine learning techniques could become a breakthrough for the hardships of causal inference. Using informational attribute of causal relationship, recognizing causality could be done by building and training a proper classification model. In this study, we would like to explore discovering causal relationship using machine learning, by introducing the notion of Neural Causation Coefficient(NCC)[ref1]. NCC has been proposed as a highly competitive model on visual recognizing task by discovering causal dependence between an object and its features from image data. By detecting such causal signals in image, the model successfully determines what the object shown from the image is.

<p align="center">
  <img src="https://github.com/hjpae/NCC-psych239_f2020/blob/main/figures/fig1(Lopez-Paz%202017).PNG" width="70%" height="70%">
</p>

**Figure 1.** Causal connection between object and its feature. Figure from *Discovering causal signals in images (Lopez-Paz et al., 2017)*. 

&nbsp;&nbsp;Here, we define causal relationship using the idea of causal intervention<sup>[1](#fn1)</sup>[ref2]. An *object* and its *features* are considered as being connected under causal relationship. From **Figure 1**, we can consider the feature *wheel* is causally dependent to the object *car*. This idea starts from the intuition based on general fact that there could be many different objects which contains *wheel*, such as *bicycle*, *bus* or *truck*, but it is hard to say a *car* is an object without *wheel* under normal condition. That is, the presence of car can guarantee the presence of wheel, while the presence of wheel only by itself cannot guarantee the presence of car. This indicates the expectation of wheel when car is present is not only conditional but also interventional, while the expectation of car when wheel is present could be only considered as conditional. From the definition of cause and effect, the causal power is identical with interventional power, which means we are always possible to modify the effect by modifying the cause<sup>[2](#fn2)</sup>. Thus, it is sufficient to count the interventional expectation of feature wheel when object car is present as causal power. To sum up, we can consider car causes wheel, while wheel cannot cause car. 

&nbsp;&nbsp;Interventional expectation from causal relationship is much more robust than just correlation. Causal inference further gives explanations about the relationship and dependency among other similar features, while correlation does not guarantee existence of such dependency. This enables the model to predict much more information when capturing the causal signals. After training an inference model distinguishing cause and effect, the model could be used on various tasks over other domains than visual recognizing task. One example we can expect is to detect causal connectivity, or effective connectome, from brain activity data. When the model successfully provides answer to what caused what, we can figure out more significant and robust explanation on the brain activity than from conventional time series correlation analysis<sup>[3](#fn3)</sup>. 

&nbsp;&nbsp;In this study, we will explore NCC with modified architectures. We also propose Deep NCC and Residual NCC model to increase performance of original NCC. By comparing Deep NCC with others, we are going to check if causality classifier shows better performance when trained with deeper layers, even though it demands more training time and computing power. Another method of increasing accuracy is to add residual block on the network. This method is well known for acquiring high accuracy with less time and computing power compared with simple deeper networks[ref]. We generate residual block on the original NCC to compare if the performance significantly increases as from other models used from visual recognition tasks.


## Methods 
&nbsp;&nbsp;We build three different neural network architectures that determines causal signals. Each model is structured under different architecture: fully-connected feedforward, fully-connected deep feedforward, and feedforward with residual blocks. The models are trained under artificially generated training dataset. After training, each model is tested by predicting cause-effect scenarios of the test dataset. Finally, test accuracy is measured by correct count / test count ratio. We use test accuracy to compare performance of models. 

### Datasets 
&nbsp;&nbsp;Our dataset is consisted of (X<sub>i</sub>, Y<sub>i</sub>) pairs where each data points follow distribution under causal link between X and Y. For example, if X is the altitude and Y is the temperature, we can put a data point pair (X<sub>i</sub>, Y<sub>i</sub>) as 'X<sub>i</sub> causes Y<sub>i</sub>' according to interventional causal power. For each pair we show causal direction of X and Y by assigning binary values. If X causes Y (X → Y), the data point pair is labeled as *causal* with binary value of 1. If Y causes X (X ← Y), the data point pair is labeled as *anti-causal* with binary value of 0. The dataset also contains *noise pair* with binary value of 0.5, which refers to (X, Y) pair not having any causal relationship. 

<p align="center">
  <img src="https://github.com/hjpae/NCC-psych239_f2020/blob/main/figures/binaryclassification(Guyon%202013).PNG" width="50%" height="50%">
</p>

**Figure 2.** Distinguishable distributions of each different causal pairs. X → Y indicates causal distribution, where Y → X indicates anti-causal distribution. Figure from Cause-effect pairs Kaggle competition, *Guyon I. 2013*. 

&nbsp;&nbsp;Each data point pairs are originated from a distribution under interventional expectation. That is, the causal direction of a pair (X<sub>i</sub>, Y<sub>i</sub>) depends on the causal direction of its underlying distribution (X, Y). **Figure 2** illustrates each different distribution having its own causal direction. Using such data for training, the model learns how to classify a data point pair into appropriate distribution. To sum up, model training from this study follows the logic of binary classification. 

&nbsp;&nbsp;We set different datasets for model training and evaluation. The training data contains a set of artificially generated causal features, where the testing data is from a distribution having ground-truth causal features. We provide train/test loss acquired from running the training dataset, and model accuracy acquired from predicting the testing dataset. We follow the training dataset generating method from[ref3], by modifying codes from GitHub repository[ref4]. 

* **Training dataset.** For training dataset, the most important is to properly featurize each data points while generating them. We apply Gaussian mixture model to keep each causal features' distribution consistent. 30,000 *data points* in total are generated under 128 different causal *feature* distributions. The length of a feature, or the number of data point a feature carries, is set to vary from 100 to 1,000, and is denoted as *m* in **Figure 3, 4, 5**. 

* **Testing dataset.** Testing dataset is used to evaluate the model accuracy. We use the dataset collected from the study[ref5], which could be downloaded from <http://webdav.tuebingen.mpg.de/cause-effect/>. All data points are collected from the real-world observation, and each feature is labeled under its ground-truth causal direction. For instance, if X stands for altitude and Y stands for temperature, then all data points included in such feature distribution is labeled as binary value 1, since altitude causes temperature (X → Y). 

### Models 
&nbsp;&nbsp;Our model follows the structure of Neural causation coefficient (NCC). All model realizations from this study are based on the paper[] and codes from GitHub repository[ref]. Both embedding and classifier layers are consisted of 4 sequential neural network activation functions from PyTorch: linear unit, 1D batch normalization, rectified linear unit, and then dropout unit with dropout rate of 0.75. All layers have 100 neurons and are fully connected. 

* **Neural Causation Coefficient (NCC)** 

<p align="center">
  <img src="https://github.com/hjpae/NCC-psych239_f2020/blob/main/figures/ncc-orig.png" width="100%" height="100%">
</p>

**Figure 3.** Architecture of Neural Causation Coefficient. *n* is the number of neurons, and *m* is the number of features. Original design is from *Discovering causal signals in images (Lopez-Paz et al., 2017)*. 

&nbsp;&nbsp;This feed-forward model is built with two fully connected embedding layers and two fully connected classifier layers. Embedding layers matches the information between data points and its feature, and classifier layers aim to classify binarized causal information of each features. Under embedding layers, each point from a causal feature distribution pass through its individual neural network layer, then averaged all together. Averaged information then passes two classifier layers. Output is calculated from logit into probability by Softmax function. 

* **Deep NCC** 

<p align="center">
  <img src="https://github.com/hjpae/NCC-psych239_f2020/blob/main/figures/ncc-deep.png" width="100%" height="100%">
</p>

**Figure 4.** Architecture of Neural Causation Coefficient with deeper layers. *n* is the number of neurons, and *m* is the number of features. 

&nbsp;&nbsp;Deep NCC is a multi-layered variant of NCC. It has same overall structure but 8 layers for each embedding and classifier layers. 

* **Residual NCC** (Fig.4)

<p align="center">
  <img src="https://github.com/hjpae/NCC-psych239_f2020/blob/main/figures/ncc-res.png" width="100%" height="100%">
</p>

**Figure 5.** Architecture of Neural Causation Coefficient with residual blocks. *n* is the number of neurons, and *m* is the number of features. 




## Results 



## Discussion 



## References 
[^ref1]: 



#### Footnotes
<a name="fn1">[1]</a>: Note that the causality here does not require time series data to be captured. Interventional causal power is defined from the effects of 'do-operators', which naturally involves sequence under time but can be formulated without measurement of exact time points. 

<a name="fn2">[2]</a>: This relationship could be illustrated more precisely by introducing 'do-operator'[ref2], which indicates the relationship where effect must happen after when *doing* the cause. According to J. Pearl, causation is strictly distinguished from correlation, since correlation lacks such strong relationship. The principle of do-operator also enables to figure out the possible counterfactuals of a causal scenario. 

<a name="fn3">[3]</a>: One intriguing theory involving causal connection of brain activity would be Integrated Information Theory of consciousness(IIT)[ref3]. This theory explains the generation mechanism of subjective consciousness by proposing intrinsic causal information of neuronal network. Practical issue of this theory would be the difficulty on discovering exact causal mechanisms from observing the brain activity. However, if it turns out possible to uncover causal relationships among brain elements, treating subjective consciousness as scientific objective would be possible under IIT. 


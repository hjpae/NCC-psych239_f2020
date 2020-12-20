Recognizing Causal Features Using Machine Learning Model
=============

## Introduction 
&nbsp;&nbsp;Causation have been numerous times. It is crucial to reveal the nature of 

&nbsp;&nbsp;In this study, we follow the notion of Neural causation coefficient(NCC)[ref] to build and train a model that finds out causality. This model has been proposed as a highly competitive method on visual recognizing task by discovering causal dependence from image data, capturing the causal relationship between an object and its features. By detecting such causal signals in image, the model successfully determines what the object shown from the image is. 

&nbsp;&nbsp;Here, an object and its features are considered as being connected under causal relationship. As from example shown from Figure 1, we can consider the feature wheel is causally dependent to the object car. We can define causal relationship even on the data without time information, by using the idea of causal intervention. This idea starts from the intuition based on general fact that there could be many different objects which contains wheel, such as bicycle, bus or truck, but it is hard to say a car is an object without wheel under normal condition. That is, the presence of car can guarantee the presence of wheel, while the presence of wheel only by itself cannot guarantee the presence of car. This indicates the expectation of wheel when car is present is not only conditional but also interventional, while the expectation of car when wheel is present could be only considered as conditional. From the definition of cause and effect, the causal power is identical with interventional power, which means we are always possible to modify the effect by modifying the cause[fn1]. Thus, it is sufficient to count the interventional expectation of feature wheel when object car is present as causal power. To sum up, we can consider car causes wheel, while wheel cannot cause car. 

&nbsp;&nbsp;Interventional expectation from causal relationship is much robust than just correlation. When gives explanations on related probability distributions, while correlation does not guarantee 
thus allowing the model much more information when capturing the causal signals. 

&nbsp;&nbsp;By defining model discriminating the cause-effect, we can explain what the cause is and what is the effect to mere situations. For example, to detect causal connectivity (effective connectome) from brain activity data. 

<sup>[1](#fn1)</sup>

## Methods 
&nbsp;&nbsp;In this study, we build three different neural network architectures that determines causal signals. Each model is structured under different architecture: fully-connected linear, fully-connected deep linear, and linear with residual blocks. The models are trained under artificially generated training dataset. After training, each model is tested by predicting cause-effect scenarios of the test dataset. Finally, test accuracy is measured by correct count / test count ratio. We use test accuracy to compare performance of models. 

### Datasets 
&nbsp;&nbsp;Our dataset is consisted of (X<sub>i</sub>, Y<sub>i</sub>) pairs where each data points follow distribution under causal link between X and Y. For example, if X is the altitude and Y is the temperature, we can put a data point pair (X<sub>i</sub>, Y<sub>i</sub>) as ‘X<sub>i</sub> causes Y<sub>i</sub>’ according to interventional causal power. For each pair we show causal direction of X and Y by assigning binary values. If X causes Y (X → Y), the data point pair is labeled as causal with binary value of 1. If Y causes X (X ← Y), the data point pair is labeled as anti-causal with binary value of 0. The dataset also contains noise pair with binary value of 0.5, which refers to (X, Y) pair not having any causal relationship. Each data point pairs are originated from a distribution under interventional expectation. That is, the causal direction of a pair (X<sub>i</sub>, Y<sub>i</sub>) depends on the causal direction of its underlying distribution (X, Y). Figure # illustrates each different distribution having its own causal direction. Using such data for training, the model learns how to classify a data point pair into appropriate distribution. To sum up, model training from this study follows the logic of binary classification. 

### Models 
&nbsp;&nbsp;Our model follows the structure of Neural causation coefficient (NCC)[ref]. All model realizations from this study are based on the paper[ref] and codes from GitHub repository[ref]. 

* NCC

* residual NCC

* deep NCC

### Model training 


## Results 



## Discussion 



## References 


#### Footnotes
<a name="fn1">[1]</a> This relationship could be illustrated more precisely by introducing ‘do-operator’[ref], where cause and effect are connected with ‘do-operator’ while mere correlation lacks such relationship. Do-operator possible to the counterfactuals, where 

---
layout: post
title: "Understanding Forward-Forward Algorithm"
subtitle: "An exploration to the new optimization algorithm"
date: 2023-10-08
author: "Simone"
tags: AI theoretical pro
background: '/img/forward_forward/optimization_cover.jpg'
---

> Paper author: Geoffrey Hinton<br>
> Publish date: 2022-12-27<br>
> Link to the paper: [Forward Forward Paper](https://www.cs.toronto.edu/~hinton/FFA13.pdf)<br>

## TL;DR

* The new **Forward Forward** optimization algorithm is a novel learning procedure for neural networks that <u>replaces</u> the <u>forward and backward passes</u> of backpropagation with <u>two forward passes</u>, one performed with positive (real) data and the other one performed with negative (synthesized/wrongly labelled) data

* Instead of computing a **cumulative** loss function, whose gradient is then backpropagated through all the layers, with **forward forward** each layer has its own objective, i.e. <u>norm of positive samples vector</u>

* Forward-Forward <u>reduces the memory requirements</u> of backpropagation while obtaining comparable performance to its counterpart, however backpropagation is still preferrable when resources are not limited

* Being an introductory exploration, this new algorithm has not been tested in enough different scenarios yet


***

In this article we're going to talk about the newly developed forward-forward algorithm, which should serve as an alternative to the back-propagation learning algorithm. Specifically, I will focus on the motivations behind its use and an explanation on how it works. 

## Motivations

Before delving into the pros of the Forward-Forward Algorithm, it is useful to grasp some of the common flaws of backpropagation, which represent certain limitations. An example is the substantial memory footprint, making it challenging to train a model in an edge computing context. 

#### Drawbacks in backpropagation

Despite being the most popular algorithm used to train neural networks, backpropagation presents some flaws. Let's explore them.

Initially, neural networks were developed with the hope of mimicking human brain behavior, as the concepts of neurons and layers are highly correlated with our understanding of actual neurons and cortical layers. Based on these assumptions, we would prefer to have a training algorithm that closely resembles the training process in the human brain itself. <br>However, there is *<u>"no convincing evidence that cortex explicitly propagates error derivatives or stores neural activities for use in a subsequent backward pass"</u>*. The top-down connections in the visual system do not follow the expected pattern of bottom-up connections, making backpropagation unlikely. Instead, they form loops, with neural activity passing through several cortical layers in both areas before returning. This challenges the feasibility of backpropagation for learning sequences. To process a continuous stream of sensory input without interruptions, the brain needs to streamline data flow and employ a real-time learning mechanism, rather than relying on backpropagation.

Another significant drawback of backpropagation is its dependence on having complete knowledge of the calculations made during the forward pass to calculate accurate derivatives. If we introduce a black box into the forward pass, backpropagation becomes unfeasible unless we can develop a differentiable model for that black box. Interestingly, for the Forward-Forward Algorithm, the presence of a black box doesn't alter the learning process whatsoever, as there's no need for backpropagation through it.

One last fundamental drawback, cited before, is the memory usage. Backpropagation requires storing activations of all intermediate layers. This practical concern becomes evident when aiming to implement deep neural architectures in a production environment where efficiency is a requisite. In such scenarios, the challenges associated with deploying and maintaining these deep models become even more pronounced, potentially affecting the efficiency and agility of the production pipeline.


#### Interesting properties of forward forward

The main idea behind forward forward is to break the need for a full computational graph of the model to perform a training step, allowing for independently train the intermediate layers of any neural networks and avoiding to store all the intermediate computations. This produces a leaner training algorithm which can be very powerful in those context where some part of the models are not differentiable or the context in which the model is trained has limited performance.



## How it works?

Now let's discover about how practically this new algorithm works, starting from understanding how backpropagation itself works, to better highlight the differences between the two.


#### Backpropagation inside

A backpropagation algorithm involves three main steps, depicted below.

1. In the first step the input is propagated through the model computing all the intermediate activations (and storing them) and the output.

2. In the second step, depending on the task (classification or regression) a loss function (Cross Entropy in the example) is used to compute the loss/distance between the outputs and the desired results. The aggregated loss (sum of distances) is then used to perform a learning step of the model.

3. In the third step the gradient of the loss is computed (direction of max increase) and is propagated to all the parameters of the model. Here the idea is to find the contribution of each weight in the final value of the loss, by decomposing the entire loss gradient using the chain rule. Once each contribution (given the input) is computed, all the weights are updated by $$ - \lambda \cdot  \partial L_{CE} $$, where $$\lambda$$ is a scalar that represents the learning rate.

<img src="/img/forward_forward/backprop.png" alt="drawing" style="display: block; margin-left: auto; margin-right: auto; width: 70%;"/>

#### Forward Forward
<img src="/img/forward_forward/forward.png" alt="drawing" style="display: block; margin-left: auto; margin-right: auto; width: 70%;"/>
<img src="/img/forward_forward/layer_norm.png" alt="drawing" style="display: block; margin-left: auto; margin-right: auto; width: 70%;"/>
<img src="/img/forward_forward/positive.png" alt="drawing" style="display: block; margin-left: auto; margin-right: auto; width: 40%;"/>

In the paper, the <u>goodness</u> is defined as the **<u>norm of the activation vector</u>**. To avoid to propagate this information through the layers, but keep propagating the computed features, **layer normalization**. <u>LN</u> normalizes the activation vector, maintaining the **direction** of the activation vector (the propagated features) and reducing the magnitude to 1 (removing any goodness trace before propagating to the next layer



## Future works

Being an introductory investigation, Geoffrey Hinton proposes, at the end of the paper some interesing questions that will be explored in the future. I report them as they have been originally written as I believe they could represent a good stimulus for anybody reading my article.

1. Can FF produce a generative model of images or video that is good enough to create the
negative data needed for unsupervised learning?

2. What is the best goodness function to use? This paper uses the sum of the squared activities
in most of the experiments but minimizing the sum squared activities for positive data and
maximizing it for negative data seems to work slightly better. More recently, just minimizing
the sum of the unsquared activities on positive data (and maximizing on negative) has
worked well

3. What is the best activation function to use? So far, only ReLUs have been explored. There
are many other possibilities whose behaviour is unexplored in the context of FF. Making the
activation be the negative log of the density under a t-distribution is an interesting possibility
(Osindero et al., 2006).

4. For spatial data, can FF benefit from having lots of local goodness functions for different
regions of the image (Löwe et al., 2019)? If this can be made to work, it should allow
learning to be much faster.

5. For sequential data, is it possible to use fast weights to mimic a simplified transformer (Ba
et al., 2016a)?

6. Can FF benefit from having a set of feature detectors that try to maximize their squared
activity and a set of constraint violation detectors that try to minimize their squared activity
(Welling et al., 2003)?


## Conclusions

In this article I proposed an interesting novel ideas to train neural networks in a different domain that those where backpropagation works well. Specifically, its paradigm seems orthogonal to the backpropagation one, allowing for independent layer learning and improved memory efficiency. However, the results demonstrate that, despite being a good alternative in such limited context, when backpropagation requirements are matched, it still outperforms forward-forward. Furthermore, despite in the near and mid term backpropagation seems to dominate the learning procedures, research works by little and apparently useless steps, which sometimes leads to bigger discoveries and paradigm shifts. Despite not being a disruptive alternative, this new little step could potentially introduce another fundamental building blocks in the exploration for an improved paradigm.

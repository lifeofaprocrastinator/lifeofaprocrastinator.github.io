---
layout: post
title: "Understanding Forward-Forward Algorithm"
subtitle: "An exploration to the new optimization algorithm"
date: 2023-10-08
author: "Simone"
tags: ai research learning
background: '/img/forward_forward/optimization_cover.jpg'
---

> Paper author: Geoffrey Hinton<br>
> Publish date: 2022-12-27<br>
> Link to the paper: [forward-forward Paper](https://www.cs.toronto.edu/~hinton/FFA13.pdf)<br>

## TL;DR

* The new **forward-forward** optimization algorithm is a novel learning procedure for neural networks that <u>replaces</u> the <u>forward and backward passes</u> of back-propagation with <u>two forward passes</u>, one performed with positive (real) data and the other one performed with negative (synthesized/wrongly labelled) data

* Instead of computing a **cumulative** loss function, whose gradient is then back-propagated through all the layers, with **forward-forward** each layer has its own objective, i.e. <u>norm of positive samples vector</u>

* Forward-Forward <u>reduces the memory requirements</u> of back-propagation while obtaining comparable performance to its counterpart, however back-propagation is still preferrable when resources are not limited

* Being an introductory exploration, this new algorithm has not been tested in enough different scenarios yet


***

In this article we're going to talk about the newly developed forward-forward algorithm, which should serve as an alternative to the back-propagation learning algorithm. Specifically, in this post I focus on the motivations behind its use-cases and I provide an explanation on how the algorithm works, compared to back-propagation one. 

## Motivations

Before delving into the pros of the Forward-Forward Algorithm, it is useful to grasp some of the common flaws of back-propagation, which represent certain limitations. An example is the substantial memory footprint, making it challenging to train a model in an edge computing context. 

#### Drawbacks of back-propagation

Despite being the most popular algorithm used to train neural networks, back-propagation presents some critical flaws which have not been addressed yet. Let's explore them.

Initially, neural networks were developed with the hope of mimicking human brain behavior, as the concepts of neurons and layers are highly correlated with our understanding of actual neurons and cortical layers. Based on these assumptions, we would prefer to have a training algorithm that closely resembles the training process in the human brain itself. 

However, there is *<u>"no convincing evidence that cortex explicitly propagates error derivatives or stores neural activities for use in a subsequent backward pass"</u>*. The top-down connections in the visual system do not follow the expected pattern of bottom-up connections, making back-propagation unlikely. Instead, they form loops, with neural activity passing through several cortical layers in both areas before returning.

This challenges the feasibility of back-propagation for learning sequences. To process a continuous stream of sensory input without interruptions, the brain needs to streamline data flow and employ a real-time learning mechanism, rather than relying on back-propagation.

Another significant drawback of back-propagation is its dependence on having complete knowledge of the calculations made during the forward pass to calculate accurate derivatives. If we introduce a black box into the forward pass, back-propagation becomes unfeasible unless we can develop a differentiable model for that black box. Interestingly, for the Forward-Forward Algorithm, the presence of a black box doesn't alter the learning process whatsoever, as there's no need for back-propagation through it.

One last fundamental drawback, cited before, is the memory usage. back-propagation requires storing activations of all intermediate layers. This practical concern becomes evident when aiming to implement deep neural architectures in a production environment where efficiency is a requisite. In such scenarios, the challenges associated with deploying and maintaining these deep models become even more pronounced, potentially affecting the efficiency and agility of the production pipeline.


#### An interesting property of forward-forward

The primary concept behind forward-forward is to eliminate the necessity for a complete computational graph of the model when performing a training step. This enables independent training of intermediate layers in any neural network while avoiding the storage of all intermediate computations.

This approach results in a more streamlined training algorithm, which can prove highly effective in situations where certain parts of the model are non-differentiable or when the training environment has limited performance.

## How it works?

Now, before discussing how forward-forward algorithm works, let's explore the inner processes of back-propagation, which is fundamental to understand the improvements that forward-forward proposes. 


#### Inside back-propagation

A back-propagation algorithm involves three main steps, depicted below.

1. In the first step the input is propagated through the model computing all the intermediate activations (and storing them) and the output.

    <img src="/img/forward_forward/backprop1.png" alt="drawing" style="display: block; margin-left: auto; margin-right: auto; width: 70%;"/>

2. In the second step, depending on the task (classification or regression) a loss function (Cross Entropy in the example) is used to compute the loss/distance between the outputs and the desired results. The aggregated loss (sum of distances) is then used to perform a learning step of the model.

    <img src="/img/forward_forward/backprop2.png" alt="drawing" style="display: block; margin-left: auto; margin-right: auto; width: 40%;"/>

3. In the third step the gradient of the loss is computed (direction of max increase) and is propagated to all the parameters of the model. Here the idea is to find the contribution of each weight in the final value of the loss, by decomposing the entire loss gradient using the chain rule. Once each contribution (given the input) is computed, all the weights are updated by $$ - \lambda \cdot  \partial L_{CE} $$, where $$\lambda$$ is a scalar that represents the learning rate.

    <img src="/img/forward_forward/backprop3.png" alt="drawing" style="display: block; margin-left: auto; margin-right: auto; width: 50%;"/>

#### The Forward-Forward

The Forward-Forward algorithm draws inspiration from Boltzmann machines (Hinton and Sejnowski, 1986) and Noise Contrastive Estimation (Gutmann and Hyvärinen, 2010). This approach replaces the traditional forward and backward passes in backpropagation with two forward passes. These two passes are identical in operation but differ in their data source and objectives. 

The positive pass processes actual data, adjusting weights to enhance the <u>goodness</u> in each hidden layer. Conversely, the negative pass deals with "negative data" and tunes weights to diminish the quality in each hidden layer.

Let's now take a look on the inner working of a forward pass by analyzing what happens inside a single layer. Given the layer independence property of forward-forward, understanding the inner working of one layer is equal to understand how the entire process works. 

Initially, the input is provided to the network and an intermediate representation is obtained.

<img src="/img/forward_forward/forwardforward.png" alt="drawing" style="display: block; margin-left: auto; margin-right: auto; width: 60%;"/>

including minus the sum of the squared activities. If the positive and negative passes could be separated in time, the negative passes could be done offline, which would make the learning much simpler in the positive pass and allow video to be pipelined through the network without ever storing activities or stopping to propagate derivatives

<img src="/img/forward_forward/layer_norm.png" alt="drawing" style="display: block; margin-left: auto; margin-right: auto; width: 70%;"/>

including minus the sum of the squared activities. If the positive and negative passes could be separated in time, the negative passes could be done offline, which would make the learning much simpler in the positive pass and allow video to be pipelined through the network without ever storing activities or stopping to propagate derivatives

<img src="/img/forward_forward/positive.png" alt="drawing" style="display: block; margin-left: auto; margin-right: auto; width: 40%;"/>

In the paper, the <u>goodness</u> is defined as the **<u>norm of the activation vector</u>**. To avoid to propagate this information through the layers, but keep propagating the computed features, **layer normalization**. <u>LN</u> normalizes the activation vector, maintaining the **direction** of the activation vector (the propagated features) and reducing the magnitude to 1 (removing any goodness trace before propagating to the next layer


In their current formulation, forward-forward applied to feed-forward networks avoid to later layers to affect what is learned in earlier layers. Despite this being the key for the layer independence, this prevent the model to learn patterns as a whole. To solve this, Hinton proposes to employ a multi-layer recurrent neural network (rnn) architecture where the input is a *boring* video made up of a repeat image. 

RNN allows connectivity at different time-step by design and this consent to learn patterns as a whole, without replacing the layer independency of forward-forward. Applied to rnn architecture, the forward-forward become <u>forward-forward in time</u>.



## Future works

As this is a preliminary exploration, Hinton concludes the paper by presenting a set of intriguing questions that are slated for future investigation. I present them in their original wording because I think they can be an engaging prompt for anyone who reads my article.

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

In this post I brought the attention to an interesting (novel) algorithm for training neural networks, called **Forward-Forward**. Conceptually this novel paradigm is orthogonal to the back-propagation one, allowing for independent layer learning and improved memory efficiency. Despite being a good alternative in some specific contexts (low-power/low-sepcs), when back-propagation requirements are matched it still outperforms the novel forward-forward, suggesting that many iterations are still required to make it useful in real-life scenarios. 

Despite back-propagation seems to dominate in the realm of learning algorithms, the process of research works by little and apparently meaningless steps, which over time leads to greater discoveries and disruptive paradigm shifts. 

To this extent, Forward-Forward perfectly represents a novel building block in machine learning that is just waiting to be harnessed and improved, like every new discovery.

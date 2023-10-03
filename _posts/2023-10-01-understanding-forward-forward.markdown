---
layout: post
title: "Understanding Forward-Forward Algorithm"
subtitle: "An exploration to the new optimization algorithm"
date: 2023-9-01
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


## Motivations

Let's see why it has been developed

#### Drawbacks in backpropagation

There is no convincing evidence
that cortex explicitly propagates error derivatives or stores neural activities for use in a subsequent
backward pass. The top-down connections from one cortical area to an area that is earlier in the
visual pathway do not mirror the bottom-up connections as would be expected if backpropagation
was being used in the visual system. Instead, they form loops in which neural activity goes through
about half a dozen cortical layers in the two areas before arriving back where it started.
Backpropagation through time as a way of learning sequences is especially implausible. To deal with
the stream of sensory input without taking frequent time-outs, the brain needs to pipeline sensory
data through different stages of sensory processing and it needs a learning procedure that can learn
on the fly. The representations in later stages of the pipeline may provide top-down information
that influences the representations in earlier stages of the pipeline at a later time step, but the
perceptual system needs to perform inference and learning in real time without stopping to perform
backpropagation.

A further serious limitation of backpropagation is that it requires perfect knowledge of the computation
performed in the forward pass1
in order to compute the correct derivatives2
. If we insert a black
box into the forward pass, it is no longer possible to perform backpropagation unless we learn a
differentiable model of the black box. As we shall see, the black box does not change the learning
procedure at all for the Forward-Forward Algorithm because there is no need to backpropagate
through it.

#### Interesting properties of forward forward

The main point of this paper is to show that neural networks containing unknown non-linearities do
not need to resort to reinforcement learning. The Forward-Forward algorithm (FF) is comparable
in speed to backpropagation but has the advantage that it can be used when the precise details of
the forward computation are unknown. It also has the advantage that it can learn while pipelining
sequential data through a neural network without ever storing the neural activities or stopping to
propagate error derivatives.



## How it works?

Let's take a look on how backprop works before 




#### Backpropagation inside
<img src="/img/forward_forward/backprop.png" alt="drawing" style="display: block; margin-left: auto; margin-right: auto; width: 70%;"/>

#### Forward Forward
<img src="/img/forward_forward/forward.png" alt="drawing" style="display: block; margin-left: auto; margin-right: auto; width: 70%;"/>
<img src="/img/forward_forward/layer_norm.png" alt="drawing" style="display: block; margin-left: auto; margin-right: auto; width: 70%;"/>
<img src="/img/forward_forward/positive.png" alt="drawing" style="display: block; margin-left: auto; margin-right: auto; width: 40%;"/>

In the paper, the <u>goodness</u> is defined as the **<u>norm of the activation vector</u>**. To avoid to propagate this information through the layers, but keep propagating the computed features, **layer normalization**. <u>LN</u> normalizes the activation vector, maintaining the **direction** of the activation vector (the propagated features) and reducing the magnitude to 1 (removing any goodness trace before propagating to the next layer


## Conclusions
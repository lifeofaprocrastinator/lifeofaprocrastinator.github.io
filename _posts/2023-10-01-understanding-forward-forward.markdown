---
layout: post
title: "Understanding Forward-Forward Algorithm"
subtitle: "An exploration to the new optimization algorithm"
date: 2023-9-01
author: "Simone"
background: '/img/optimization_cover.jpg'
---

Author: Geoffrey Hinton <br>
Date: 2022/12/27 <br>
Link: [Forward Forward Paper](https://www.cs.toronto.edu/~hinton/FFA13.pdf)


## TL;DR

* The new **Forward Forward** optimization algorithm introduces a novel learning procedure for neural networks, replacing the forward and backward passes of backpropagation to two forward passes, one performed with positive (real) data and the other one performed with negative (synthesized/wrong labelled) data

* Instead of computing a **cumulative** loss function, whose gradient is then backpropagated through all the layers, each layer has its own objective, which is defined as the maximum goodness, for positive data, and minimum **goodness** for negative data. The concept of **goodness** can be associated to **positive correlation** or correctness, but analytically can be defined in an arbitrary way

* In the paper, the goodness is defined as the <u>**norm of the activation vector**<u>. To avoid to propagate this information through the layers, but keep propagating the computed features, **layer normalization**. LN normalizes the activation vector, maintaining the **direction** of the activation vector (the propagated features) and reducing the magnitude to 1 (removing any goodness trace before propagating to the next layer)

* Forward-Forward reduces the memory requirements of backpropagation while obtaining comparable performance to its counterpart, however backpropagation is still preferrable when resources are not limited

* Being an introductory exploration, this new algorithm has not been tested in enough different scenarios yet


## Motivations


## How it works?


## ELI5


## Conclusions
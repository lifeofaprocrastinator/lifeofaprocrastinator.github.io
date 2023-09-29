---
layout: post
title: "Understanding Forward-Forward Algorithm"
subtitle: "An exploration to the new optimization algorithm"
date: 2023-9-01
author: "Simone"
tags: AI theoretical pro
background: '/img/optimization_cover.jpg'
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


## How it works?

<!---In the paper, the <u>goodness</u> is defined as the **<u>norm of the activation vector</u>**. To avoid to propagate this information through the layers, but keep propagating the computed features, **layer normalization**. <u>LN</u> normalizes the activation vector, maintaining the **direction** of the activation vector (the propagated features) and reducing the magnitude to 1 (removing any goodness trace before propagating to the next layer)--->


## Conclusions
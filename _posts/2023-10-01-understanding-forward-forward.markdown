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

* Instead of computing a **cumulative** loss function whose gradient is then backpropagated through all the layers each layer has its own objective, which is defined as the maximum goodness, for positive data, and minimum goodness for negative data. The concept of **goodness** can be associated to the one of **positive correlation** or correctness, but analytically can be defined in an arbitrary way

* Forward-Forward reduces the memory requirements of backpropagation while obtaining comparable performance to its counterpart, however backpropagation is still preferrable when resources are not limited

* Being an introductory exploration, this new algorithm has not been tested in a various amount of scenario yet


## Motivations


## How it works?


## ELI5


## Conclusions
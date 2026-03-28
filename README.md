# Finsler-Laplace-Beltrami Operators with Application to Shape Analysis

**Course:** Geodesic Methods and Deformable Models  
**Instructor:** Laurent D. Cohen

## Project Overview

The goal of this project is to study and implement the findings of the paper *Finsler-Laplace-Beltrami Operators with Application to Shape Analysis* (Weber, Dages, Gao, Cremers).

This paper proposes an approach using Finsler geometry to generalize standard geometric operators (such as the Laplace-Beltrami operator), allowing for a more refined and directionally-aware analysis of surfaces, shape matching, and deformations.

## Notebooks & Experiments

| Experiment              | Description                                                                                                                                                         | Link                                                                                                                                                                           |
|:------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Core Implementation** | Mathematical implementation of the FLBO and visual simulation of heat diffusion on 3D meshes.                                                                       | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tpitois/FLBO/blob/master/notebooks/train.ipynb)          |
| **Shape Corres**        | Training and evaluation of a neural network leveraging FLBO for convolutions on the TOSCA dataset, featuring 3D shape correspondence visualization and PCK curves.. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tpitois/FLBO/blob/master/notebooks/heat_diffusion.ipynb) |

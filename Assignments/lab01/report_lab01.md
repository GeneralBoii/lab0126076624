# Lab Report — Lab 01: Efficient Separable Gaussian Convolution

*Fill in each section below. Be specific — name the actual methods/parameters you used, since
your report is graded alongside your code.*

## Overview

_1 sentence: What was the goal of this lab?_

The goal of this lab was to blur an image with a Gaussian filter much faster than the standard 2D convolution method.

## Implementation

_1-2 sentences: Briefly describe the steps you performed to implment efficient convolution._

A Gaussian filter can be split into two 1D filters, so I blurred each row of the image first and then each column, using np.convolve with a 1D Gaussian kernel. This needs far fewer calculations than one big 2D convolution.

## Results

_1-2 sentences: How well did your method perform vs the baseline in the pytests? ._

My method took 0.0549 s compared to 4.0935 s for the convolve2d baseline, about 75x faster. It passed all 10 tests, including every timing threshold.

# K-Means Algorithm for Handwritten Digit Clustering

## Introduction

- Implement K-Means algorithm from scratch.
- Dealt with image data (processed and stored in vector format).
- Used V-measure for evaluation.

## Dataset

The input data (provided under Test) consists of 10,740 images of handwritten digits (0-9). The images were scanned and scaled into 28x28 pixels. For each digit, every pixel is represented as an integer in the range [0, 255], where 0 corresponds to the pixel being completely white, and 255 corresponds to the pixel being completely black. This results in a 28x28 matrix of integers for each digit, which can then be flattened into a 1x784 vector. No labels are provided.

In the input data, each row is a record (image), which contains 784 comma-delimited integers.

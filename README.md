# ResNet-34 Implementation from Scratch

## Description
This project features a complete, from-scratch implementation of the ResNet-34 architecture using TensorFlow and Keras. It demonstrates a deep-dive into modern Convolutional Neural Network (CNN) design, specifically the application of residual learning to solve the vanishing gradient problem in deep networks.

## Architecture Highlights
- **Custom Layers**: Implementation of a specialized `CustomConv2D` layer that integrates 2D Convolution with Batch Normalization.
- **Residual Blocks**: Logic for identity and projection (dotted) shortcuts to handle spatial dimension changes.
- **Full Model Pipeline**: A 34-layer deep architecture concluding with Global Average Pooling and a Softmax Dense layer for ImageNet-scale classification (1000 classes).

## Technical Stack
- Python
- TensorFlow
- Keras Functional & Subclassing API

## Model Complexity
- **Total Parameters**: 21,823,208
- **Trainable Parameters**: 21,806,184
- **Non-trainable Parameters**: 17,024

## Implementation Details
The model follows the standard ResNet design:
1. Initial 7x7 Convolution and Max Pooling.
2. Four stages of Residual Blocks with increasing filter sizes (64, 128, 256, 512).
3. Downsampling performed by blocks with a stride of 2.
4. Global Average Pooling before the final classification head.

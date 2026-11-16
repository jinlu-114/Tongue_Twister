# Tongue Twister 👅

*Exploring oral cancer image classification with deep learning*

## Overview

Tongue Twister is a personal machine learning project exploring how convolutional neural networks can classify oral cancer images using transfer learning.

What started as an experiment in image classification eventually became an opportunity to learn about something equally important: how to determine whether a model's results are actually meaningful.

**Disclaimer:** This is an educational project, not a clinically validated diagnostic tool. Its predictions should not be used for medical decisions!!!

## Research Question

How does model architecture affect oral cancer image classification performance when training data is limited?

I compared two pretrained convolutional neural networks, ResNet18 and ResNet50, to investigate whether a deeper architecture necessarily produces better results.

## Approach

### Dataset

I used the publicly available [Oral Cancer (Lips and Tongue) Images](https://www.kaggle.com/datasets/shivam17299/oral-cancer-lips-and-tongue-images) dataset, containing 131 images organized into cancer and non-cancer categories.

The images were divided into training, validation, and held-out test sets using a stratified split.

### Model Training

I implemented the classification pipeline using PyTorch and torchvision.

Both ResNet18 and ResNet50 were initialized with pretrained ImageNet weights. I froze their feature extraction layers and replaced their final classification layers to perform binary classification.

The training pipeline included image preprocessing, data augmentation for training images, validation after each epoch, and checkpoint saving.

### Evaluation

I evaluated model performance using accuracy, balanced accuracy, precision, recall, F1-score, ROC-AUC, and a confusion matrix.

I was particularly interested in class-specific recall because overall accuracy alone can hide important classification errors.

## Results

The selected model correctly classified **19 out of 20 images** in the held-out test set.

| Metric | Result |
| Test accuracy | 95.0% |
| Balanced accuracy | 91.7% |
| ROC-AUC | 0.94 |
| Cancer recall | 100% (14/14) |
| Non-cancer recall | 83.3% (5/6) |

The model correctly classified every cancer image in the test set but incorrectly classified one non-cancer image as cancerous.

### Architecture Comparison

| Model | Best validation accuracy | Held-out test accuracy |
| ResNet18 | Pending | Add if evaluated |
| ResNet50 | Pending | Add if evaluated |

Although the selected model achieved high test accuracy, the small sample size makes it difficult to draw strong conclusions. With only 20 test images, a single incorrect prediction changes the overall accuracy by five percentage points.

## Project History

My first implementation used a pretrained ResNet50 model and achieved 92.59% validation accuracy on 27 images.

At first, I was excited by the result. However, when I reviewed the evaluation pipeline, I realized there were several weaknesses in my approach. The validation images were receiving random augmentations, and I hadn't reserved a separate test set.

I had spent more time thinking about how to improve the model than how to evaluate it properly.

That realization motivated me to revisit the project. I separated the training and validation transformations, introduced a held-out test set, and expanded the experiment to compare ResNet18 and ResNet50.

Revisiting my original implementation helped me understand that developing a machine learning model isn't just about achieving the highest possible accuracy. It's also about questioning how that accuracy was obtained.

## Limitations

There are several important limitations to this experiment:

- **Small dataset:** With only 131 images, performance estimates are highly sensitive to individual predictions and the choice of data split.
- **Potential data leakage:** Without patient identifiers or a thorough duplicate-image investigation, related images could appear across different subsets.
- **Limited generalizability:** The dataset may not represent the diversity of patients, lesions, or imaging conditions encountered in clinical settings.
- **Transfer learning:** The pretrained feature extractors were frozen, so this experiment does not investigate full-network fine-tuning.
- **No external validation:** The held-out test set comes from the same dataset, not an independent clinical source.

These limitations mean that the reported results should be interpreted as experimental findings rather than evidence of diagnostic reliability.

## What I Learned

This was my first substantial attempt at building an image classification model with PyTorch.

Beyond learning how to implement transfer learning, I became more comfortable with preparing image datasets, training neural networks, saving checkpoints, and interpreting classification metrics.

The most valuable part of the project was recognizing weaknesses in my first approach and having to rethink my methodology.

I also learned that a more complex model isn't automatically a better one. Comparing ResNet18 and ResNet50 gave me a reason to consider computational efficiency alongside predictive performance.

## Next Investigations

If I continue developing Tongue Twister, I'd like to:

1. Repeat training with different random seeds to investigate performance variability.
2. Compare the neural networks against a simpler classification baseline.
3. Investigate duplicate images and introduce patient-level splitting if identifiers become available.
4. Evaluate performance using a larger, independent dataset.
5. Explore MRI-based tongue tumour classification as a separate project using an appropriate imaging dataset.

## Running the Project

The Jupyter notebook contains the data preparation, training, and evaluation pipeline.

Install the required dependencies using the project's `requirements.txt` file, then run the notebook cells in order. The dataset is obtained through KaggleHub.

Saved `.pth` checkpoints can be loaded with the corresponding model architecture to reproduce predictions without retraining.

## Acknowledgments

This project uses publicly available images from the *Oral Cancer (Lips and Tongue) Images* dataset and pretrained ResNet architectures provided through torchvision.

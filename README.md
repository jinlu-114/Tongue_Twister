# Tongue Twister 🧠

**Exploring oral cancer image classification with transfer learning**

An introductory PyTorch project comparing pretrained ResNet18 and ResNet50 on the public Kaggle **Oral Cancer (Lips and Tongue) Images** dataset. The aim is to learn how dataset preparation, model choice, and evaluation design affect reported performance—not to build a diagnostic system.

## Research question

Does a larger pretrained CNN necessarily perform better on a small oral-image dataset?

## Approach

- Inspect class counts and example images.
- Create reproducible, stratified 70/15/15 train/validation/test image-level splits.
- Apply random augmentation only to training images.
- Train the classification heads of frozen ResNet18 and ResNet50 models for 10 epochs.
- Compare validation accuracy and learning curves; select a model using validation only.
- Evaluate the selected model once on the held-out test split, including class-specific metrics and a confusion matrix.
- Inspect incorrect predictions and document limitations.

## Run

Open `Tongue_Twister_Improved.ipynb` in Google Colab, optionally select a GPU, and run cells in order. The notebook downloads the dataset with `kagglehub`. Dependencies are listed in `requirements.txt`. Model weights and figures are generated only after running the notebook.

## Results
ResNet18 vs. ResNet50 Comparison

I compared ResNet18 and ResNet50 to investigate whether a deeper architecture improves classification performance on a relatively small oral cancer image dataset. Both models used pretrained ImageNet weights, with their feature extraction layers frozen and their classification heads retrained.

I initially expected ResNet50 to perform better because of its greater representational capacity. However, ResNet18 requires fewer computational resources, making it potentially more practical if both models achieve similar accuracy.

The comparison focuses on validation accuracy and training behaviour rather than model size alone. With only 131 images, small differences in accuracy may reflect the particular dataset split rather than a meaningful advantage of one architecture.

My goal was to determine whether the additional complexity of ResNet50 was justified by a measurable improvement in classification performance.

## Limitations

The dataset is small, and image-level splits do not guarantee independence between patients or near-duplicate images. The test set contains only around 20 images, so estimates are unstable. The public images are not MRI scans, and the model is not clinically validated. No medical decisions should be based on its outputs.

## Project history

An initial ResNet50-only exploration reported 92.59% validation accuracy on 27 images, but its validation pipeline applied random augmentations and lacked an independent test set. This revision addresses those methodological weaknesses and adds a controlled architecture comparison. The original score should not be compared directly with the revised results.

## Next investigations

Evaluate repeated random seeds, check for duplicates and patient-level overlap, compare against a simple baseline, and explore external validation.

## Acknowledgments

Uses public Kaggle data, torchvision pretrained weights, and PyTorch.
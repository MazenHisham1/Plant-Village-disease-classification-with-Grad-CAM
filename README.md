#  Plant Disease Classification with CNN, Transfer Learning and Grad-CAM

Classification of 38 plant disease and healthy-leaf classes from the **PlantVillage** dataset using TensorFlow/Keras. The project compares a custom CNN trained from scratch against a fine-tuned **EfficientNetB0**, explains the predictions with **Grad-CAM**, and ships a **Streamlit** demo that accepts an uploaded image or a camera photo.


---

## Highlights

- **99.35% test accuracy** and **0.991 macro F1** with transfer learning (EfficientNetB0).
- **Custom CNN baseline** with 91.30% accuracy, for a fair comparison against the pretrained model.
- **Explainable AI:** Grad-CAM heatmaps for both models, including misclassified images.
- **Full evaluation:** accuracy, top-5 accuracy, macro and weighted precision, recall and F1, per-class reports and confusion matrices.
- **Interactive demo:** Streamlit app with image upload or live camera input, model selection and Grad-CAM overlay.
- Every plot, report and model is saved automatically in the `results/` folder.

---

## Results

Evaluated on a held-out test set of **8,146 images** (15% stratified split).

| Metric | Custom CNN | Transfer Learning (EfficientNetB0) |
|---|:---:|:---:|
| Accuracy | 0.9130 | **0.9935** |
| Top-5 accuracy | 0.9971 | **1.0000** |
| Precision (macro) | 0.8762 | **0.9932** |
| Recall (macro) | 0.9249 | **0.9899** |
| F1-score (macro) | 0.8922 | **0.9914** |
| Precision (weighted) | 0.9265 | **0.9936** |
| Recall (weighted) | 0.9130 | **0.9935** |
| F1-score (weighted) | 0.9151 | **0.9935** |

### Observations

- Transfer learning improves macro F1 by about **10 points** over the CNN trained from scratch.
- The CNN has a noticeably lower macro precision (0.876) than recall (0.925). Its weakest classes are `Raspberry___healthy` (F1 0.70), `Grape___healthy` (0.72), `Apple___Apple_scab` (0.79) and `Potato___Late_blight` (0.79), where healthy-class predictions absorb errors from other classes.
- EfficientNetB0 has an F1 of 0.97 or higher on 37 of 38 classes. The hardest class is `Corn___Cercospora_leaf_spot Gray_leaf_spot` (F1 0.93), which it sometimes confuses with Northern Leaf Blight, a visually similar corn disease.

### Explainability (Grad-CAM)

Grad-CAM highlights the leaf regions that contributed most to each prediction, which helps check that the models focus on lesions rather than on the background.

<p align="center">
  <img src="results/gradcam/cnn_vs_transfer_gradcam.png" width="520" alt="Grad-CAM comparison">
</p>

---

## Dataset

- **Source:** [`BrandonFors/Plant-Diseases-PlantVillage-Dataset`](https://huggingface.co/datasets/BrandonFors/Plant-Diseases-PlantVillage-Dataset) on Hugging Face (PlantVillage, 38 classes, 256×256 RGB images).
- **Split:** the original splits are merged and re-split with stratification into **70% train / 15% validation / 15% test** (seed 42).
- **Class imbalance:** handled with balanced class weights during training.

---

## Models

### 1. Custom CNN
- Four convolutional blocks (32, 64, 128, 256 filters), two Conv layers each, with BatchNorm, ReLU and Dropout.
- Global average pooling, a Dense(256) layer, Dropout(0.5) and a 38-way output.
- Trained from scratch with Adam (lr 1e-3), early stopping and learning-rate reduction on plateau.

### 2. Transfer Learning (EfficientNetB0)
- ImageNet-pretrained EfficientNetB0 backbone with a global-average-pooling head.
- **Stage 1:** train the head with the backbone frozen (lr 1e-3).
- **Stage 2:** fine-tune the top 60 backbone layers with BatchNorm layers frozen (lr 1e-4).

### Shared training setup
- Input size 224×224, batch size 32.
- Data augmentation inside the model: random flips, rotation, zoom and contrast.
- Balanced class weights, early stopping on validation accuracy, ReduceLROnPlateau.

### 3. Grad-CAM
Gradients of the predicted class logit are taken with respect to the last convolutional feature map, averaged per channel and used to weight the feature maps. The result is passed through ReLU, normalized and overlaid on the input image.

---

## Project structure

```
.
├── Plant_Village_V2.ipynb        # full pipeline: data, CNN, transfer learning, Grad-CAM, comparison
├── plantvillage_pipeline.py      # the same pipeline as a standalone script
├── app.py                        # Streamlit demo
├── requirements.txt
├── README.md
└── results/
    ├── class_names.json
    ├── model_comparison.csv / .png
    ├── cnn/                      # curves, confusion matrices, reports, metrics.json
    ├── transfer/                 # curves, confusion matrices, reports, metrics.json
    ├── gradcam/                  # Grad-CAM figures
    └── models/                   # saved .keras models
```

---

## Getting started

### 1. Install

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
pip install -r requirements.txt
```

A GPU is strongly recommended for training.

### 2. Train and evaluate

Run the notebook `Plant_Village_V2.ipynb`, or run the script:

```bash
python plantvillage_pipeline.py
```

This downloads the dataset, trains both models, evaluates them, generates the Grad-CAM figures and writes everything to `results/`.

### 3. Save the class names (needed by the demo)

```python
import json
json.dump(class_names, open("results/class_names.json", "w"))
```

### 4. Run the demo

```bash
streamlit run app.py
```

The app lets you:
- choose between the CNN and the transfer-learning model,
- upload a leaf image or take a photo with the camera,
- see the predicted class, the top-3 probabilities and the Grad-CAM overlay.

---

## Tech stack

Python · TensorFlow / Keras · Hugging Face Datasets · scikit-learn · Matplotlib · Seaborn · Pandas · Streamlit

---

## Limitations

- PlantVillage images are leaves photographed in controlled conditions on a plain background. Accuracy on field photos with cluttered backgrounds, different lighting or other devices is likely to be lower.
- The train, validation and test sets come from a random split, so images of the same leaf may appear in more than one split. The reported scores are therefore likely optimistic compared with real-world use.
- The models cover only the 38 classes in the dataset and will give a wrong answer for any other plant or disease.
- This project is for research and education. It is not a replacement for professional agricultural diagnosis.

---

## Future work

- Test on in-the-wild datasets such as PlantDoc to measure real-world robustness.
- Compare more backbones (EfficientNetV2, ConvNeXt, MobileNetV3) and export a lightweight TFLite model for mobile use.
- Add an out-of-distribution check so the app can reject images that are not leaves.

---

## Acknowledgements

- PlantVillage dataset: Hughes & Salathé, *An open access repository of images on plant health to enable the development of mobile disease diagnostics* (2015).
- Hugging Face dataset: `BrandonFors/Plant-Diseases-PlantVillage-Dataset`.
- Grad-CAM: Selvaraju et al., *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization* (2017).

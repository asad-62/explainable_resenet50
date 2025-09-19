# explainable_resenet50

This project demonstrates basic explainable AI (XAI) techniques on a pretrained ResNet50 model using a sample ImageNet image (Bullmastiff). It includes implementations of Grad-CAM, LIME, and SHAP, each producing visual explanations to show which parts of the image influenced the model’s predictions.

## Contents

- `explainable_ai_gradcam.ipynb` — Grad-CAM implementation in Jupyter Notebook
- `lime_single_image.py` — LIME applied to a local image
- `shape_xai.py` — SHAP applied to the same image
- `Bullmastiff.jpg` — Sample ImageNet image used for testing
- `gradcam_results.png` — Output from Grad-CAM
- `lime_explanation.png` — Output from LIME
- `shap_output.png` — Output from SHAP
- `requirements.txt` — Required Python packages

## Requirements

Install the dependencies with:

```bash
pip install -r requirements.txt

## Reprodcuing the results
python lime_single_image.py
python shape_xai.py

### Gradcam > jupyter notebook explainable_ai_gradcam.ipynb

### Grad-CAM
![Grad-CAM](https://github.com/asad-62/explainable_resenet50/blob/main/gradcam_results.png?raw=true)


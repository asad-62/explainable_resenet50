
# Group imports
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import shap
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

# Use CPU only
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def load_class_names(url):
    try:
        with open(shap.datasets.cache(url)) as file:
            return [v[1] for v in json.load(file).values()]
    except Exception as e:
        print(f"Error loading class names: {e}")
        return []

def load_image(img_path):
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img).astype(np.uint8)
        return img_array
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

def show_predictions(model, X, top=4):
    preds = model.predict(X)
    decoded = decode_predictions(preds, top=top)[0]
    print("\nTop Predictions:")
    for i, (_, label, prob) in enumerate(decoded):
        print(f"{i+1}. {label} ({prob:.2%})")

def main():
    model = ResNet50(weights='imagenet')
    url = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
    class_names = load_class_names(url)

    img_path = "Bullmastiff.jpg"  
    img_array = load_image(img_path)
    if img_array is None:
        return

    original_img = img_array.copy()
    X = preprocess_input(np.expand_dims(img_array, axis=0))

    def model_fn(x):
        return model(preprocess_input(x.copy()))

    masker = shap.maskers.Image("inpaint_telea", original_img.shape)
    explainer = shap.Explainer(model_fn, masker, output_names=class_names)

    shap_values = explainer(
        np.array([original_img]),
        max_evals=300,
        batch_size=50,
        outputs=shap.Explanation.argsort.flip[:4]
    )

    shap.image_plot(shap_values)
    plt.savefig("shap_output.png", bbox_inches='tight', dpi=300)
    plt.close()

    plt.imshow(original_img.astype(np.uint8))
    plt.axis('off')
    plt.title("Original Image")
    plt.savefig("original_image_clean.png", bbox_inches='tight', dpi=300)
    plt.close()

    show_predictions(model, X, top=4)

if __name__ == "__main__":
    main()

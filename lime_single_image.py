
# Imports
import numpy as np
import torch
from PIL import Image
from torchvision import transforms, models
from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt

def get_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    return device

def load_model(device):
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model = model.to(device)
    model.eval()
    return model

def load_image(path):
    try:
        img = Image.open(path).convert('RGB')
        return img
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

def predict_fn(images, model, device):
    batch_tensors = []
    for img_array in images:
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float() / 255.0
        img_tensor = torch.nn.functional.interpolate(
            img_tensor.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False
        ).squeeze(0)
        img_tensor = img_tensor.to(device)
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1)
        img_tensor = (img_tensor - mean) / std
        batch_tensors.append(img_tensor)
    batch = torch.stack(batch_tensors)
    with torch.no_grad():
        outputs = model(batch)
    return outputs.cpu().numpy()

def explain_image(img, model, device, output_filename='lime_explanation.png'):
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        np.array(img),
        lambda imgs: predict_fn(imgs, model, device),
        top_labels=1,
        hide_color=0,
        num_samples=1000
    )
    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0],
        positive_only=True,
        num_features=30,
        hide_rest=False
    )
    plt.figure(figsize=(10, 6))
    plt.imshow(mark_boundaries(temp, mask))
    plt.axis('off')
    plt.title('LIME Explanation')
    plt.savefig(output_filename, bbox_inches='tight', dpi=300, pad_inches=0.1)
    print(f"LIME explanation saved as: {output_filename}")

def main():
    device = get_device()
    model = load_model(device)
    img_path = "Bullmastiff.jpg"  # Change to your image path
    img = load_image(img_path)
    if img is None:
        return
    explain_image(img, model, device)

if __name__ == "__main__":
    main()
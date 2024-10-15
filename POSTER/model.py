import cv2
import dlib
import os
import torch
from PIL import Image
from torchvision import transforms
import warnings
warnings.filterwarnings("ignore")
from utils import *
from models.emotion_hyp import pyramid_trans_expr

def align_face(image):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = np.array(image)
    detector = dlib.get_frontal_face_detector()  # 人臉檢測器
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # 68點特徵點檢測模型

    dets = detector(image, 1)
    if len(dets) > 0:
        shape = predictor(image, dets[0])
        aligned_face = dlib.get_face_chip(image, shape)  # 對齊臉部
        aligned_face_image = Image.fromarray(aligned_face)
        return aligned_face_image


def load_model(device):
    num_classes = 7
    checkpoint_path = "checkpoint/rafdb_best.pth"
    model = pyramid_trans_expr(img_size=224, num_classes=num_classes, type="large")
    print("Loading pretrained weights...", checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    checkpoint = checkpoint["model_state_dict"]
    model = load_pretrained_weights(model, checkpoint)
    model.to(device)  # Move the model to the specified device (GPU or CPU)
    model.eval()
    return model

def predict_image(model, image, device):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image = transform(image)
    image = image.unsqueeze(0)  # Add a batch dimension
    image = image.to(device)  # Move the image to the specified device (GPU or CPU)

    with torch.no_grad():
        outputs, features = model(image)
        _, predicted = torch.max(outputs, 1)
        predicted = predicted.cpu().numpy()[0]  # Move the result to CPU for further operations

    return predicted

def get_label(num):
    if num == 0:
        return "surprise"
    elif num == 1:
        return "fear"
    elif num == 2:
        return "disgust"
    elif num == 3:
        return "happy"
    elif num == 4:
        return "sad"
    elif num == 5:
        return "angry"
    else:
        return "neutral"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == 'cuda':
    print("Using GPU for inference")
else:
    print("Using CPU for inference")

model = load_model(device)


def model_pipeline(image : Image):
        aligned_face = align_face(image)
        predicted_class = predict_image(model, aligned_face, device)
        emotion = get_label(predicted_class)
        return emotion

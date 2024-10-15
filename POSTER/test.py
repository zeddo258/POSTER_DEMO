import cv2
import dlib
import os
import torch
from PIL import Image
from torchvision import transforms
import warnings
from collections import Counter # 計算情緒出現次數
warnings.filterwarnings("ignore")
from utils import *
from models.emotion_hyp import pyramid_trans_expr

def align_face(image):
    detector = dlib.get_frontal_face_detector()  # 人臉檢測器
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # 68點特徵點檢測模型

    dets = detector(image, 1)
    if len(dets) > 0:
        shape = predictor(image, dets[0])
        aligned_face = dlib.get_face_chip(image, shape)  # 對齊臉部
        aligned_face_image = Image.fromarray(aligned_face)
        return aligned_face_image

def extract_frames(video_path, interval=1):
    video = cv2.VideoCapture("video_reader/" + video_path)
    fps = int(video.get(cv2.CAP_PROP_FPS))
    frame_count = 0
    success, frame = video.read()
    frames = []  # 回傳

    while success:
        if frame_count % (fps * interval) == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # BGR to RGB
            aligned_face = align_face(rgb_frame) # 對齊臉部
            frames.append((aligned_face, frame_count // fps))

        success, frame = video.read()
        frame_count = frame_count + 1

    video.release()
    return frames

def get_mp4_files(dir):
    files = os.listdir(dir)
    mp4_files = [file for file in files if file.endswith(".mp4")]
    return mp4_files

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

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        print("Using GPU for inference")
    else:
        print("Using CPU for inference")

    model = load_model(device)
    mp4_files = get_mp4_files(dir="video_reader")
    
    # 處理每個mp4
    for mp4 in mp4_files:
        print(f"Processing {mp4}...")
        emotion_counter = Counter()  # 用來計算每個情緒出現次數
        frames = extract_frames(mp4) # 影片每一秒的人臉(已對齊)
        for aligned_face, frame_num in frames:
            # 對齊後的圖像丟給模型預測
            predicted_class = predict_image(model, aligned_face, device)
            emotion = get_label(predicted_class)
            emotion_counter[emotion] =  emotion_counter[emotion] + 1
            print(f"Frame {frame_num} in {mp4}: {emotion}")
        
        emotion_lst = sorted(emotion_counter.keys())
        peak_emotion = max(emotion_counter, key=emotion_counter.get)

        print(f"Emotion list for {mp4}: {emotion_lst}")
        print(f"Peak emotion for {mp4}: {peak_emotion}")

if __name__ == "__main__":
    main()

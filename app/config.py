import torch

class Config:
    UPLOAD_FOLDER = './uploaded_files'
    ALLOWED_EXTENSIONS = {'dcm'}
    MAX_IMAGES = 116
    IMAGE_SIZE = 224
    MODEL_PATH = "./app/static/models/breast-cancer-classifier-binary-v1.0-f1-0.7682-20250107.pth"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
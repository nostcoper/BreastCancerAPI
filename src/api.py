from flask import Flask, request, jsonify
import os
from werkzeug.utils import secure_filename
import numpy as np
import pydicom
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import ToPILImage
import torch.nn as nn
from torchvision import models
from typing import List, Tuple

class Config:
    UPLOAD_FOLDER = './uploaded_files'
    ALLOWED_EXTENSIONS = {'dcm'}
    MAX_IMAGES = 116
    IMAGE_SIZE = 224
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = Flask(__name__)
app.config.from_object(Config)


transform = transforms.Compose([
    transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])


def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def load_dicom_image(file_path: str) -> np.ndarray:
    try:
        dicom = pydicom.dcmread(file_path)
        if not hasattr(dicom, "pixel_array"):
            raise ValueError(f"El archivo {file_path} no contiene datos de imagen.")
        image = dicom.pixel_array.astype(np.float32)
        intercept = getattr(dicom, "RescaleIntercept", 0)
        slope = getattr(dicom, "RescaleSlope", 1)
        return image * slope + intercept
    except Exception as e:
        raise RuntimeError(f"Error al cargar la imagen DICOM ({file_path}): {e}")

def normalize_for_resnet(image: np.ndarray) -> torch.Tensor:
    p25, p95 = np.percentile(image, (25, 95))
    image = np.clip(image, p25, p95)
    image = (image - image.min()) / (image.max() - image.min()) if image.max() != image.min() else np.zeros_like(image)
    image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).repeat(3, 1, 1)
    return image_tensor

# Modelos y procesamiento de datos
class PreprocessingData(Dataset):
    def __init__(self, root_dir: str, max_images: int = Config.MAX_IMAGES, image_size: int = Config.IMAGE_SIZE):
        self.root_dir = root_dir
        self.max_images = max_images
        self.image_size = image_size
        self.transform = transform
        self.to_pil = ToPILImage()

    def __len__(self):
        return 1

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        dicom_files = sorted([f for f in os.listdir(self.root_dir) if f.endswith('.dcm')])

        images = []
        for file in dicom_files[:self.max_images]:
            image_path = os.path.join(self.root_dir, file)
            try:
                image = load_dicom_image(image_path)
                image = normalize_for_resnet(image)
                image = self.to_pil(image)
                if self.transform:
                    image = self.transform(image)
                images.append(image)
            except RuntimeError as e:
                app.logger.warning(e)

        sequence_length = len(images)

        if len(images) == 0:
            placeholder_image = torch.zeros((3, self.image_size, self.image_size))
            images = [placeholder_image] * self.max_images
        while len(images) < self.max_images:
            images.append(torch.zeros_like(images[0]))

        images_tensor = torch.stack(images)
        return images_tensor, sequence_length

class ResNetFeatureExtractor(nn.Module):
    def __init__(self):
        super(ResNetFeatureExtractor, self).__init__()
        resnet = models.resnet18(weights='IMAGENET1K_V1')
        self.features = nn.Sequential(*list(resnet.children())[:-2])

        for param in self.features[-2:].parameters():
            param.requires_grad = True

    def forward(self, x):
        features = self.features(x)
        return features.mean([2, 3])

class AttentionModule(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionModule, self).__init__()
        self.attention = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):
        scores = self.attention(x) 
        weights = torch.softmax(scores, dim=1)
        context = (x * weights).sum(dim=1)
        return context

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=True,
                            dropout=0.45)
        self.attention = AttentionModule(hidden_size)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        context = self.attention(lstm_out)
        output = self.classifier(context)
        return output

class SequenceClassificationModel(nn.Module):
    def __init__(self, lstm_hidden_size=512, lstm_num_layers=2):
        super(SequenceClassificationModel, self).__init__()
        self.feature_extractor = ResNetFeatureExtractor()
        self.sequence_classifier = LSTMClassifier(input_size=512, 
                                                  hidden_size=lstm_hidden_size,
                                                  num_layers=lstm_num_layers)
    def forward(self, x):
        batch_size, seq_len, c, h, w = x.size()
        x = x.view(batch_size * seq_len, c, h, w)
        features = self.feature_extractor(x)  
        features = features.view(batch_size, seq_len, -1)
        output = self.sequence_classifier(features)
        return output

# Cargar modelo preentrenado
model = torch.load("./model_f1_0.7682.pth", map_location=Config.DEVICE)
model.eval()

# Endpoint principal
@app.route('/upload', methods=['POST'])
def upload_and_classify():
    if 'files' not in request.files:
        return jsonify({'error': 'No files part in the request'}), 400

    files = request.files.getlist('files')
    if not files:
        return jsonify({'error': 'No files uploaded'}), 400

    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

    try:
        prediction = predict_single_patient(app.config['UPLOAD_FOLDER'])
        return jsonify({'message': 'Files uploaded and classified successfully.', 'prediction': prediction}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def predict_single_patient(root_dir: str) -> int:
    dataset = PreprocessingData(root_dir)
    dataloader = DataLoader(dataset, batch_size=1)

    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(Config.DEVICE)
            outputs = model(images)
            predicted = (outputs > 0.5).int()
            return predicted.item()

if __name__ == "__main__":
    app.run(debug=True)

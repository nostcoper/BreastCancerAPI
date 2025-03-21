import io
import numpy as np
import torch
import pydicom
from torchvision import transforms
from torchvision.transforms import ToPILImage
from app.config import Config

transform = transforms.Compose([
    transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

to_pil = ToPILImage()

def load_dicom_from_bytes(dicom_bytes: bytes) -> np.ndarray:
    try:
        with io.BytesIO(dicom_bytes) as dicom_buffer:
            dicom = pydicom.dcmread(dicom_buffer)
            if not hasattr(dicom, "pixel_array"):
                raise ValueError("El archivo DICOM no contiene datos de imagen.")
            image = dicom.pixel_array.astype(np.float32)
            intercept = getattr(dicom, "RescaleIntercept", 0)
            slope = getattr(dicom, "RescaleSlope", 1)
            return image * slope + intercept
    except Exception as e:
        raise RuntimeError(f"Error al cargar la imagen DICOM: {e}")

def normalize_for_resnet(image: np.ndarray) -> torch.Tensor:
    """Normaliza una imagen para procesarla con ResNet."""
    p25, p95 = np.percentile(image, (25, 95))
    image = np.clip(image, p25, p95)
    image = (image - image.min()) / (image.max() - image.min()) if image.max() != image.min() else np.zeros_like(image)
    image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).repeat(3, 1, 1)
    return image_tensor

def preprocess_dicom_image(file_path: str):
    """Preprocesa una imagen DICOM para el modelo."""
    image = load_dicom_from_bytes(file_path)
    image = normalize_for_resnet(image)
    image = to_pil(image)
    if transform:
        image = transform(image)
    return image
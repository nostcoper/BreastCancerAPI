import os
import torch
from torch.utils.data import Dataset
from typing import Tuple, List
import logging
from app.config import Config
from app.utils.data_utils import preprocess_dicom_image
from typing import List, Tuple, Dict, Any
from app.utils.loggin_config import logger


class PreprocessingData(Dataset):
    def __init__(self, dicom_files_data: List[Dict[str, Any]], max_images: int = Config.MAX_IMAGES, image_size: int = Config.IMAGE_SIZE):
        self.dicom_files_data = dicom_files_data
        self.max_images = max_images
        self.image_size = image_size
    def __len__(self):
        return 1

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        images = []
        
        # Procesar solo hasta el número máximo de imágenes
        for file_data in self.dicom_files_data[:self.max_images]:
            try:
                image = preprocess_dicom_image(file_data['bytes'])
                images.append(image)
            except RuntimeError as e:
                extra = getattr(self, 'extra', {})
                logger.warning(f"Error procesando imagen: {e}", extra=extra)

        sequence_length = len(images)

        # Manejar el caso de no tener imágenes o tener menos del máximo
        if len(images) == 0:
            placeholder_image = torch.zeros((3, self.image_size, self.image_size))
            images = [placeholder_image] * self.max_images
        while len(images) < self.max_images:
            images.append(torch.zeros_like(images[0]))

        images_tensor = torch.stack(images)
        return images_tensor, sequence_length
import torch
from torch.utils.data import DataLoader
from app.config import Config
from typing import List, Dict, Any
from app.services.preprocessing_service import PreprocessingData
from app.services.model_service import model
from app.utils.loggin_config import logger

def classsification_single_patient(dicom_files_data: List[Dict[str, Any]], request_id: str, client_ip: str) -> Dict:
    extra = {'request_id': request_id, 'client_ip': client_ip}
    logger.info(f"Iniciando predicción con {len(dicom_files_data)} archivos", extra=extra)
    
    dataset = PreprocessingData(dicom_files_data)
    dataloader = DataLoader(dataset, batch_size=1)

    with torch.no_grad():
        for images, sequence_length in dataloader:
            images = images.to(Config.DEVICE)
            logger.info(f"Imágenes cargadas en el dispositivo: {Config.DEVICE}", extra=extra)
            
            outputs = model(images)
            probability = torch.sigmoid(outputs).item()
            prediction = 1 if probability > 0.5 else 0
            
            logger.info(f"Resultado de la predicción: {prediction} (probabilidad: {probability:.4f})", extra=extra)
            
            return {
                "prediction": prediction,
                "probability": round(probability, 4),
                "sequence_length": sequence_length.item()
            }
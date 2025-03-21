import torch
from app.config import Config
from app.models.model import SequenceClassificationModel

def load_model(model_path=Config.MODEL_PATH):
    try:
        model = SequenceClassificationModel() 
        state_dict = torch.load(model_path, map_location=Config.DEVICE)
        model.load_state_dict(state_dict)
        model.eval()
        return model
    except Exception as e:
        raise RuntimeError(f"Error al cargar el modelo: {str(e)}")

model = load_model()
model_lock = torch.multiprocessing.Lock() if torch.multiprocessing.get_start_method() == 'spawn' else None
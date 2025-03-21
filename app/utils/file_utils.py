import os
import shutil
from werkzeug.utils import secure_filename
from app.config import Config

def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS
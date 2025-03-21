import uuid
from flask import Blueprint, request, jsonify
from app.services.classification_service import classsification_single_patient
from app.utils.file_utils import allowed_file
from app.utils.loggin_config import logger
from app.services.model_service import model_lock

class classification_controller: 
    def upload_and_classify(self):
        request_id = str(uuid.uuid4())[:8]
        client_ip = request.remote_addr
        extra = {'request_id': request_id, 'client_ip': client_ip}
        logger.info("Nueva petición recibida", extra=extra)

        if 'files' not in request.files:
            return jsonify({'error': 'No files part in the request'}), 400

        if 'files' not in request.files:
                logger.warning("No se encontró la parte 'files' en la petición", extra=extra)
                return jsonify({'error': 'No files part in the request'}), 400
        
        files = request.files.getlist('files')
        if not files:
            logger.warning("No se subieron archivos", extra=extra)
            return jsonify({'error': 'No files uploaded'}), 400
        
        dicom_files_data = []
        for file in files:
            if file and allowed_file(file.filename):
                try:
                    file_bytes = file.read()
                    dicom_files_data.append({
                        'filename': file.filename,
                        'bytes': file_bytes
                    })
                    logger.info(f"Archivo {file.filename} leído en memoria", extra=extra)
                except Exception as e:
                    logger.error(f"Error leyendo archivo {file.filename}: {e}", extra=extra)
                    return jsonify({'error': f'Error al procesar archivo {file.filename}: {str(e)}'}), 500

        try:
            if model_lock:
                with model_lock:
                    prediction = classsification_single_patient(dicom_files_data, request_id, client_ip)
            else:
                prediction = classsification_single_patient(dicom_files_data, request_id, client_ip)
                
            logger.info(f"Clasificación completada: {prediction}", extra=extra)
            return jsonify({'message': 'Files processed and classified successfully.', 'prediction': prediction}), 200
        except Exception as e:
            logger.error(f"Error en predicción: {e}", extra=extra)
            return jsonify({'error': str(e)}), 500


        
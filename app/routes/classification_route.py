from flask import Blueprint
from app.controllers.classification_controller import classification_controller
from flasgger import swag_from

classification_bp = Blueprint('classification_bp', __name__)
controller = classification_controller()

@classification_bp.route('/upload-and-classify', methods=['POST'])
@swag_from({
    'tags': ['Clasificación'],
    'summary': 'Carga y clasificación de múltiples imágenes DICOM',
    'description': (
        "Este endpoint permite subir hasta 116 archivos en formato DICOM (.dcm) y obtener una clasificación predictiva "
        "mediante un modelo de deep learning. La API categoriza las imágenes en dos clases: Benigno (BI-RADS 1-3) y "
        "Maligno (BI-RADS 4-6), proporcionando una probabilidad asociada a la predicción."
    ),
    'parameters': [
        {
            'name': 'files',
            'in': 'formData',
            'type': 'file',
            'required': True,
            'description': 'Archivos DICOM (.dcm) a clasificar (máximo 116)',
            'allowMultiple': True
        }
    ],
    'responses': {
        200: {
            'description': 'Clasificación exitosa',
            'schema': {
                'type': 'object',
                'properties': {
                    'message': {
                        'type': 'string',
                        'example': 'Clasificación realizada con éxito'
                    },
                    'results': {
                        'type': 'array',
                        'items': {
                            'type': 'object',
                            'properties': {
                                'file_name': {
                                    'type': 'string',
                                    'description': 'Nombre del archivo procesado'
                                },
                                'prediction': {
                                    'type': 'integer',
                                    'description': '0 = Benigno (BI-RADS 1-3), 1 = Maligno (BI-RADS 4-6)',
                                    'example': 1
                                },
                                'probability': {
                                    'type': 'number',
                                    'format': 'float',
                                    'description': 'Probabilidad de la predicción (0.0 - 1.0)',
                                    'example': 0.87
                                },
                                'sequence_length': {
                                    'type': 'integer',
                                    'description': 'Cantidad total de imágenes procesadas en la secuencia',
                                    'example': 100
                                }
                            }
                        }
                    }
                }
            }
        },
        400: {
            'description': 'Error en la solicitud',
            'schema': {
                'type': 'object',
                'properties': {
                    'error': {
                        'type': 'string',
                        'description': 'Motivo del error en la solicitud',
                        'example': 'Formato de archivo no válido'
                    }
                }
            }
        },
        500: {
            'description': 'Error interno del servidor',
            'schema': {
                'type': 'object',
                'properties': {
                    'error': {
                        'type': 'string',
                        'description': 'Descripción del error interno',
                        'example': 'Error al procesar la imagen'
                    }
                }
            }
        }
    }
})
def get_classification():
    """
    Carga y clasificación de imágenes DICOM.
    """
    return controller.upload_and_classify()

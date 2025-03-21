from flask import Flask, redirect, Blueprint
from app.config import Config
from app.utils.loggin_config import setup_logger
from flasgger import Swagger

def create_app():
    setup_logger()
    app = Flask(__name__)
    app.config.from_object(Config)
    
    swagger_config = {
        "headers": [],
        "specs": [
            {
                "endpoint": "apispec",
                "route": "/v1/apispec.json",  
                "rule_filter": lambda rule: True,
                "model_filter": lambda tag: True,
            }
        ],
        "static_url_path": "/v1/static", 
        "swagger_ui": True,
        "specs_route": "/v1/docs/" 
    }
    
    swagger_template = {
        "swagger": "2.0",
        "info": {
            "title": "INSIGHTPRISM - API de Clasificación de Cáncer de Mama",
            "description": (
                "INSIGHTPRISM es una API desarrollada por Carlos Serrato para el análisis del cáncer de mama mediante "
                "inteligencia artificial. Clasifica imágenes médicas DICOM en dos categorías: Benigno (BI-RADS 1-3) y "
                "Maligno (BI-RADS 4-6), con una precisión del 74%. Diseñada exclusivamente para estudios e investigación."
            ),
            "version": "1.0.0"
        },
        "schemes": ["http", "https"],
    }

    
    Swagger(app, config=swagger_config, template=swagger_template)
    base_prefix = '/v1/'
    from app.routes.classification_route import classification_bp
    app.register_blueprint(classification_bp, url_prefix=f'{base_prefix}')
    
    @app.route('/')
    def redirect_to_docs():
        return redirect('/v1/docs/')

    return app
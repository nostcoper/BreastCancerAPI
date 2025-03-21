import logging
from datetime import datetime

logger = None

class CustomFormatter(logging.Formatter):
    def format(self, record):
        record.timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        return super().format(record)

class ContextFilter(logging.Filter):
    def filter(self, record):
        record.request_id = getattr(record, 'request_id', 'N/A')
        record.client_ip = getattr(record, 'client_ip', 'N/A')
        return True

def setup_logger():
    global logger
    
    if logger is not None:
        return logger
        
    formatter = CustomFormatter('%(timestamp)s - %(levelname)s - [%(request_id)s] - [%(client_ip)s] - %(message)s')
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    for hdlr in logger.handlers:
        logger.removeHandler(hdlr)
    
    logger.addHandler(handler)
    logger.addFilter(ContextFilter())
    
    return logger
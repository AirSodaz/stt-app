import logging
import os
from logging.handlers import RotatingFileHandler
import sys

# Constants
LOG_DIR = "logs"
LOG_FILE = "app.log"
MAX_BYTES = 10 * 1024 * 1024  # 10 MB
BACKUP_COUNT = 5
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s"

def setup_logging(name=None):
    """
    Setup logging configuration.
    This should be called at the entry point of the application.
    """
    # Ensure log directory exists
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
        
    log_path = os.path.join(LOG_DIR, LOG_FILE)

    # Create formatters
    formatter = logging.Formatter(LOG_FORMAT)

    # File Handler
    file_handler = RotatingFileHandler(
        log_path, 
        maxBytes=MAX_BYTES, 
        backupCount=BACKUP_COUNT, 
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Remove existing handlers to avoid duplicates if setup is called multiple times
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    # Configure uvicorn loggers to use our handlers/format
    # This ensures uvicorn logs (access logs etc) go to our file as well
    logging.getLogger("uvicorn.access").handlers = [console_handler, file_handler]
    logging.getLogger("uvicorn.error").handlers = [console_handler, file_handler]
    
    # Return a logger for the specific module if name is provided
    if name:
        return logging.getLogger(name)
    else:
        return root_logger

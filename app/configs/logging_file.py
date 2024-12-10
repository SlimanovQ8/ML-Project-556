import logging


# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set the log level
    format='%(asctime)s - %(filename)s - Line: %(lineno)d - %(funcName)s() - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',  # Custom date format
)
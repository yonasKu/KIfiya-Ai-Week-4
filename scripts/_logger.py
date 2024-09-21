

import logging
import sys

# Configure logging to output to stdout for Jupyter notebooks
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

def get_logger():
    logger = logging.getLogger()
    return logger

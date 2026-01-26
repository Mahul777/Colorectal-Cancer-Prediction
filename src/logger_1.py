# This file configures and creates a logger for the project

# Import the logging module to track events, errors, and messages
import logging

# Import os module to work with directories and file paths
import os

# Import datetime to create time-based log filenames
from datetime import datetime

# Define the directory where log files will be stored
LOGS_DIR = "logs"

# Create the logs directory if it does not already exist
# exist_ok=True prevents an error if the directory already exists
os.makedirs(LOGS_DIR, exist_ok=True)

# Create a log file name with the current date
# Example: logs/log_2026-01-26.log
LOG_FILE = os.path.join(
    LOGS_DIR,
    f"log_{datetime.now().strftime('%Y-%m-%d')}.log"
)

# Configure the logging system
logging.basicConfig(
    # File where logs will be written
    filename=LOG_FILE,

    # Format of each log message:
    # timestamp - log level - actual message
    format='%(asctime)s - %(levelname)s - %(message)s',

    # Set the minimum log level to INFO
    # INFO, WARNING, ERROR, and CRITICAL will be logged
    level=logging.INFO,
)

# Function to create and return a logger for a given module/file
def get_logger(name):
    # Get a logger with the specified name (usually __name__)
    logger = logging.getLogger(name)

    # Set the logging level for this logger
    logger.setLevel(logging.INFO)

    # Return the configured logger
    return logger

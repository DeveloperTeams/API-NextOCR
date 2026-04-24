import logging

logger = logging.getLogger(__name__)

def log_info_green(message):
    green_color = "\033[92m"
    reset_color = "\033[0m"
    logger.info(f"{green_color}{message}{reset_color}")

def log_error_red(message):
    red_color = "\033[91m"
    reset_color = "\033[0m"
    logger.error(f"{red_color}{message}{reset_color}")

def log_warning_yellow(message):
    yellow_color = "\033[93m"
    reset_color = "\033[0m"
    logger.warning(f"{yellow_color}{message}{reset_color}")

def log_debug_blue(message):
    blue_color = "\033[94m"
    reset_color = "\033[0m"
    logger.debug(f"{blue_color}{message}{reset_color}")

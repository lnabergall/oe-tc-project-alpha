import sys

import jax

from loguru import logger


def setup_logging():
    config = {
        "handlers": [
            {"sink": sys.stdout, "level": "INFO", "format": "{time:YYYY-MM-D:HH:mm:ss.SSS} - {message}"},
            # {"sink": "logfile.log", "serialize": True, "rotation": "500 MB"}
        ]
    }
    logger.configure(**config)


def jax_log_info(fmt: str, *args, **kwargs):
    jax.debug.callback(lambda *args, **kwargs: logger.info(fmt, *args, **kwargs), *args, **kwargs)


def jax_log_debug(fmt: str, *args, **kwargs):
    jax.debug.callback(lambda *args, **kwargs: logger.debug(fmt, *args, **kwargs), *args, **kwargs)
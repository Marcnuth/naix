#!/usr/bin/python3
# coding=utf-8
import os
from pathlib import Path
import logging
from logging.config import dictConfig


DIR_BASE = Path(os.path.dirname(os.path.realpath(__file__))).parent
DIR_RESOURCES = DIR_BASE / 'resources'
DIR_ASSETS = DIR_RESOURCES / 'assets'
DIR_MODELS = DIR_RESOURCES / 'models'

FILE_LOGGING = Path('/tmp/naix/naix.log')
FILE_LOGGING.parent.mkdir(parents=True, exist_ok=True)


# Logging
dictConfig({
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "basic": {
            "format"  : "[%(asctime)s] [%(process)d:%(thread)d] [%(levelname)s] [%(name)s] %(filename)s:%(funcName)s:%(lineno)d %(message)s",
            "datefmt" : "%Y-%m-%d %H:%M:%S"
        }
    },
    "handlers": {
        "file": {
            "level": "INFO",
            "class": "logging.handlers.TimedRotatingFileHandler",
            "filename": FILE_LOGGING.absolute().as_posix(),
            "formatter": "basic",
            "encoding": "utf-8",
            "when": "midnight",
            "interval": 1,
            "backupCount": 7
        },
    },
    "loggers": {
        "naix" : {
            "handlers" : ["file"],
            "propagate": "true",
            "level"    : "INFO"
        }
    }
})

logger = logging.getLogger(__name__)
logger.info('naix logs will be output into %s', FILE_LOGGING.absolute().as_posix())


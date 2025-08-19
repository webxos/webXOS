import logging.config

logging_config = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        }
    },
    "handlers": {
        "file": {
            "class": "logging.FileHandler",
            "filename": "vial2/logs/vial2.log",
            "formatter": "default",
            "level": "INFO"
        },
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default",
            "level": "INFO"
        }
    },
    "loggers": {
        "": {
            "level": "INFO",
            "handlers": ["file", "console"]
        }
    }
}

logging.config.dictConfig(logging_config)

# xAI Artifact Tags: #vial2 #mcp #logging #config #neon_mcp

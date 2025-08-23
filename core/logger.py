import logging

class Logger:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Logger, cls).__new__(cls)
            cls._instance._setup_logger()
        return cls._instance

    def _setup_logger(self) -> None:
        self.logger = logging.getLogger("MediaRecommender")
        self.logger.setLevel(logging.INFO)

        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(message)s"
        )

        root_logger = logging.getLogger()
        if not any(isinstance(h, logging.StreamHandler) for h in root_logger.handlers):
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(formatter)
            root_logger.addHandler(console_handler)

        self.logger.propagate = True

    def get_logger(self, name: str = None) -> logging.Logger:
        """Get a logger with a specific name (e.g., module name), ensuring it propagates to root."""
        if name:
            logger = logging.getLogger(name)
            logger.setLevel(logging.INFO)
            logger.propagate = True
            return logger
        return self.logger

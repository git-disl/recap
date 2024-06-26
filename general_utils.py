import sys, os
import logging


class LogLevel:

    def __init__(self, level):
        self.loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
        self.old_levels = [logger.level for logger in self.loggers]
        self.level = level

    def __enter__(self):
        for logger in self.loggers:
            logger.setLevel(self.level)

    def __exit__(self, exc_type, exc_val, exc_tb):
        for i, logger in enumerate(self.loggers):
            logger.setLevel(self.old_levels[i])


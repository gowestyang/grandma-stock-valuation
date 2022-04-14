from .utils import logger
from logging import StreamHandler

HDLR = StreamHandler
FORMATTER = logger.SIMPLE_FORMATTER
NAME = None
KWARGS = {}

logger = logger.DefaultLogger(hdlr=HDLR, formatter=FORMATTER, name=NAME, **KWARGS)

from datasetsanity import get_logger, MissingValuesError

logger = get_logger()

try:
    raise MissingValuesError(columns=["a", "b"])
except MissingValuesError as e:
    logger.error(e)
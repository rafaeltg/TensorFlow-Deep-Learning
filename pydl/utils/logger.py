import logging


def get_logger(name, level):

    """
    :param name:
    :param level:
    """
    logger = logging.getLogger(name)

    if isinstance(level, int):
        log_level = logging.INFO if level == 1 else logging.WARNING
    else:
        log_level = level

    logger.setLevel(log_level)

    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(log_level)

    # create formatter
    formatter = logging.Formatter('%(levelname)s (%(name)s) - %(message)s')

    # add formatter to ch
    ch.setFormatter(formatter)

    # add ch to logger
    logger.addHandler(ch)
    return logger

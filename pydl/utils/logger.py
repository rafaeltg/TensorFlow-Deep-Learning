import logging


class Logger(object):

    def __init__(self, name, level):

        """
        :param name:
        :param level:
        """

        self.logger = logging.getLogger(name)

        if isinstance(level, int):
            log_level = logging.INFO if level == 1 else logging.WARNING

        else:
            log_level = level

        self.logger.setLevel(log_level)

        # create console handler and set level to debug
        ch = logging.StreamHandler()
        ch.setLevel(log_level)

        # create formatter
        formatter = logging.Formatter('%(levelname)s (%(name)s) - %(message)s')

        # add formatter to ch
        ch.setFormatter(formatter)

        # add ch to logger
        self.logger.addHandler(ch)

    def info(self, msg):

        """
        :param msg:
        :return:
        """

        self.logger.info(msg)

    def error(self, msg):

        """
        :param msg:
        :return:
        """

        self.logger.error(msg)

##Handling the exceptions
import sys
from src.logger import logging

class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = self._error_message_detail(error_message, error_detail)

    @staticmethod
    #@staticmethod is a Python decorator that makes a method in a class not depend on the instance or class itself.
    def _error_message_detail(error, error_detail: sys):
        _, _, exc_tb = error_detail.exc_info()
        file_name = exc_tb.tb_frame.f_code.co_filename
        error_message = (
            "Error occured in py script name [{0}] line number [{1}] with the error message : [{2}]"
            .format(file_name, exc_tb.tb_lineno, str(error))
        )
        return error_message

    def __str__(self):
        return self.error_message

if __name__ == "__main__":
    try:
        a = 1 / 0
    except Exception as e:
        logging.info("Divide by Zero")
        raise CustomException(e,sys)


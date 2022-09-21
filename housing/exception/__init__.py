import os
import sys


class Housing_Exception(Exception):

    def __init__(self, error_message: Exception, error_detail: sys):
        super().__init__(error_message)
        self.errr_message = Housing_Exception.get_detailed_error_message(
            error_message, error_detail)

    @staticmethod
    def get_detailed_error_message(error_message: Exception, error_detail: sys) -> str:
        """
        error_message: Exception object
        error_detail: object of sys module
        """
        _, _, traceback = error_detail.exc_info()
        file_name = traceback.tb_frame.f_code.co_filename
        exception_block_line_num = traceback.tb_frame.f_lineno
        try_block_line_num = traceback.tb_lineno
        error_message = f"""error occured in script:
        [{file_name}] at 
        try_block_line_num [{try_block_line_num}] and exception_block_line_num [{exception_block_line_num}]
        error message [{error_message}]"""

        return error_message

    def __str__(self):
        return self.errr_message

    def __repr__(self):
        return Housing_Exception.__name__.str()

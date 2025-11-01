import sys
from typing import Optional

def get_detailed_error_message(error: Exception, error_detail: sys) -> str:

    _, _, exc_tb = error_detail.exc_info()
    
    if exc_tb is not None:
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno
        error_message = (
            f"Error occurred in python script: [{file_name}] "
            f"at line number: [{line_number}] "
            f"error message: [{str(error)}]"
        )
    else:
        error_message = f"Error occurred: [{str(error)}]"
    
    return error_message


class SpaceshipTitanicException(Exception):

    def __init__(self, error_message: str, error_detail: sys = None):

        super().__init__(error_message)
        
        if error_detail is not None:
            self.error_message = get_detailed_error_message(
                error=Exception(error_message),
                error_detail=error_detail
            )
        else:
            self.error_message = error_message
    
    def __str__(self):
        return self.error_message

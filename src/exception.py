import sys

def format_error_message(error,error_detail:sys):
    _,_,exc_tb=error_detail.exc_info()
    filename=exc_tb.tb_frame.f_code.co_filename
    error_message=f"The error lies in file {filename} in line number {exc_tb.tb_lineno} and error message is {str(error)}"
    return error_message


class CustomException(Exception):
    def __init__(self,error_message,error_detail:sys):
        super().__init__(error_message)
        self.error_message=format_error_message(error_message,error_detail)

    def __str__(self):
        return self.error_message

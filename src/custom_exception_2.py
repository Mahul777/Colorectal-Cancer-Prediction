# This file defines a custom exception class
# It allows us to raise more informative errors in the project

# Import traceback module to extract detailed error information
import traceback

# Import sys module to access system-specific error details
import sys

# Define a custom exception class
# This allows us to raise more informative errors in the project
class CustomException(Exception):

    # Constructor for the custom exception
    def __init__(self, error_message, error_detail: sys):
        # Call the parent Exception class constructor
        super().__init__(error_message)

        # Generate a detailed error message with file name and line number
        self.error_message = self.get_detailed_error_message(
            error_message,
            error_detail
        )

    # Static method to extract detailed error information
    @staticmethod
    def get_detailed_error_message(error_message, error_detail: sys):

        # Get exception information:
        # exc_tb contains the traceback object
        _, _, exc_tb = traceback.sys.exc_info()

        # Get the file name where the exception occurred
        file_name = exc_tb.tb_frame.f_code.co_filename

        # Get the line number where the exception occurred
        line_number = exc_tb.tb_lineno

        # Return a formatted error message with file name and line number
        return f"Error in {file_name} , line {line_number} : {error_message}"

    # Define how the exception is displayed when printed or logged
    def __str__(self):
        return self.error_message

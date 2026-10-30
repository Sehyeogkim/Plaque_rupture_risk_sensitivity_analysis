import os
from datetime import datetime



def log_case_info(log_file, geo_num, case_params, duration, success, error_msg=None):
    """
    Log detailed information about a case execution
    """
    with open(log_file, 'a') as f:
        f.write(f"geo_id: {geo_num}\n")
        f.write(f"Duration: {duration}\n")
        f.write(f"Success: {success}\n")
        f.write(f"Error_msg: {error_msg}\n")

        # Log parameters in a more readable format
        f.write("\nParameters:\n")
        for key, value in case_params.items():
            f.write(f"  {key}: {value}\n")
        
        if error_msg:
            f.write(f"\nError Message:\n{error_msg}\n")
        
        f.write(f"{'='*50}\n\n")
        

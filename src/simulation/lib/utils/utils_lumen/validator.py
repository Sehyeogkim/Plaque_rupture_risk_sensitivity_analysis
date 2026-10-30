import os

def validate_slab_txt(slab_path: str):
    '''
    Validate the slab.txt file.

    if slab does not exist or contains NaN values, return False.
    otherwise, return True.
    '''
    if not os.path.exists(slab_path):
        print(f"Error: slab.txt not found at {slab_path}")
        return False

    with open(slab_path, "r") as f:
        lines = f.readlines()
        has_nan = any("nan" in line.lower() for line in lines)
        
        if has_nan:
            print("Warning: slab.txt contains NaN values.")
            return False
        else:
            print("No NaN values found in slab.txt.")
            return True
    

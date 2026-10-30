import os
from pathlib import Path
'''
paths.py includes the functions for the file operations.
1. copy and move the file
2. solver scripts revisement or generation.
'''

#copy mesh from from_folder to to_folder
def copy_mesh(which_mesh_folder:str, dest_folder:str):
    '''
    copy which_mesh_folder(./mesh-complete) to dest_folder
    '''
    import shutil
    mesh_dir = os.path.join(dest_folder, "mesh-complete")
    shutil.copytree(which_mesh_folder, mesh_dir, dirs_exist_ok=True)
    print(f"Copied mesh-complete directory as: {mesh_dir}")
    pass



#copy svpre from from_folder to to_folder
def copy_svpre(from_folder:str, to_folder:str):
    '''
    copy svpre from from_folder to to_folder
    '''
    import shutil
    from_file = os.path.join(from_folder, "model.svpre")
    to_file = os.path.join(to_folder, "model.svpre")
    shutil.copyfile(from_file, to_file)
    print(f"Copied svpre directory as: {to_file}")
    pass

def copy_solver_inp(from_folder:str, to_folder:str, ramp:bool = False):
    '''
    copy solver_inp from from_folder to to_folder
    '''
    import shutil
    if ramp:
        from_file = os.path.join(from_folder, "solver_ramp.inp")
        to_file = os.path.join(to_folder, "solver_ramp.inp")
    else:
        from_file = os.path.join(from_folder, "solver.inp")
        to_file = os.path.join(to_folder, "solver.inp")

    shutil.copyfile(from_file, to_file)
    print(f"Copied solver_inp directory as: {to_file}, ramp: {ramp}")
    pass

# create flow file
def create_flow_file(flow_file_path: str, Q: float):
    """
    Create a flow file with the following format:

    0.0 -Q_in
    1.0 -Q_in

    Input:
    - flow_file_path: str, path to the output flow file
    - Q_in: float, input flow value (will be negated in the file)
    
    """
    Q_in = -1 * Q

    with open(flow_file_path, "w") as f:
        f.write(f"0.0 {Q_in}\n")
        f.write(f"1.0 {Q_in}")
    
    print(f"Created flow file as: {flow_file_path}, Q: {Q}")
    pass

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

# revise the inp file
def revise_inp_file(input_file_path: str, changes: dict):
    """
    Modify specific numerical values in a .inp file.

    Parameters:
    - input_file_path: str, path to the original .inp file
    - changes: dict, key = keyword (string to search),
                     value = new number (float or int)
               Example: {"Number of Timesteps:": 5000,
                         "Time Step Size:": 0.001}
    """
    modified_lines = []
    with open(input_file_path, "r") as f:
        for line in f:
            new_line = line
            for key, new_value in changes.items():
                if line.strip().startswith(key):
                    parts = line.split(":")
                    if len(parts) >= 2:
                        new_line = f"{key} {new_value}\n"
            modified_lines.append(new_line)

    with open(input_file_path, "w") as f:
        f.writelines(modified_lines)

    print(f"Revised inp file as: {input_file_path}")
    pass


def create_svpre_file(svpre_file_path: Path, mesh_dir: Path):
    '''
    svpre_file_path: e.g) ~/peak/model.svpre
    mesh_dir: e.g) ~/home/jeff/40_1D_pratice/meshing
    '''

    mesh_dir = str(mesh_dir)
    svpre_file_path = str(svpre_file_path)

    svpre_content = f"""number_of_variables 5  #P, Vx, Vy, Vz

mesh_and_adjncy_vtu {mesh_dir}/mesh-complete/mesh-complete.mesh.vtu 
set_surface_id_vtp {mesh_dir}/mesh-complete/walls_combined.vtp 1
set_surface_id_vtp {mesh_dir}/mesh-complete/mesh-surfaces/inlet.vtp 2
set_surface_id_vtp {mesh_dir}/mesh-complete/mesh-surfaces/outlet.vtp 3

fluid_density 1.06
fluid_viscosity 0.04
initial_pressure 0
initial_velocity 0.0001 0.0001 0.0001

prescribed_velocities_vtp {mesh_dir}/mesh-complete/mesh-surfaces/inlet.vtp
pressure_vtp {mesh_dir}/mesh-complete/mesh-surfaces/outlet.vtp 0.0
noslip_vtp {mesh_dir}/mesh-complete/walls_combined.vtp #no-slip

bct_analytical_shape parabolic
bct_period 1.0
bct_point_number 2
bct_fourier_mode_number 15

bct_create {mesh_dir}/mesh-complete/mesh-surfaces/inlet.vtp inflow.flow

bct_merge_on

bct_write_dat bct.dat
bct_write_vtp bct.vtp

write_geombc geombc.dat.1
write_restart restart.0.1
write_numstart 0
"""
    
    # Write the svpre file
    with open(svpre_file_path, 'w') as f:
        f.write(svpre_content)
    
    print(f"model.svpre file created at: {svpre_file_path}")
    return


'''
functions to run the svpre and svSolver.
'''

import subprocess
import time
from pathlib import Path


def run_svpre(exe_dir: str, svpre_path: str):
    '''
    Run the svpre solver without changing directories.
    '''
    subprocess.run([svpre_path, "model.svpre"], cwd=exe_dir, check=True)
    print(f"svpre completed\n")
    return

def run_svsolver(exe_dir: str, svsolver_path: str, nproc: int, ramp = False):
    '''
    Run the svSolver solver without changing directories.
    '''
    start_time = time.time()
    argv = [
        'mpirun', '-n', str(nproc),
        svsolver_path,
        "solver_ramp.inp" if ramp else "solver.inp",
    ]
    subprocess.run(argv, cwd=exe_dir, check=True)
    end_time = time.time()
    print(f"svSolver completed in {end_time - start_time} seconds\n")
    return

def run_svpost(exe_dir: str, svpost_path: str, nproc: int, step_number: int, save_file_name: str):
    '''
        Run the svpost solver without changing directories.
                command = (
                f"{svpost_path} "
                f"-sn {step_number} "
                f"-vtp rst "
                f"-indir {nproc}-procs_case/ "
                f"-all 1 ") 
    '''
    start_time = time.time()
    argv = [
        svpost_path,
        "-sn", str(step_number),
        "-vtp", str(save_file_name),
        "-indir", f"{nproc}-procs_case/",
        "-all", "1",
    ]
    subprocess.run(argv, cwd=exe_dir, check=True)
    end_time = time.time()
    print(f"svpost completed in {end_time - start_time} seconds\n")
    return


# ---- Simmetrix meshing methods ----
def run_simmetrix_license(license_path: str):
    import os
    os.environ['SimModSuite_licenseFile'] = license_path
    print("\n\n--- License environment variable set ---\n\n")
    return

def run_simmetrix_meshing(vtp_path: str, exe_dir: str, simmetrix_path: str,
                        grid_size: float, boundary_layer_thickness: float, 
                        number_of_boundary_layers: int):

    try:
        argv = [
            simmetrix_path,
            "--input_model_file", str(vtp_path),
            "--grid_size", str(grid_size),
            "--generate_boundary_layers", "1",
            "--boundary_layer_thickness", str(boundary_layer_thickness),
            "--number_of_boundary_layers", str(number_of_boundary_layers),
            "--wall_surface_id", "1",
            "--output_mesh_file", "mesh-complete.mesh.vtu",
        ]
        subprocess.run(argv, cwd=exe_dir, check=True)
    
    except Exception as e:
        print(f"Error: {e}")
        return

def run_tree_pt_converter(exe_dir: str, pt_converter_dir: str, tree_path: str):

    argv = [
        pt_converter_dir,
        "--input_tree_file", tree_path
    ]
    subprocess.run(argv,    cwd=exe_dir,   check=True)
    
    output_tree_path = tree_path.replace(".dat", "_converted.dat") #Need to check this.

    return output_tree_path

def run_1d_simulation(exe_dir: str, pul_1d_solver_path: str, tree_path: str, bc_types_path: str, t_size: float, n_steps: int, save_dir):
    '''
    Run the 1D simulation.
    '''
    # Convert paths to strings if they are Path objects
    tree_path = str(tree_path)
    bc_types_path = str(bc_types_path)
    save_dir_path = Path(save_dir) if not isinstance(save_dir, Path) else save_dir
    output_file = str(save_dir_path / "result.json")
    
    print("tree_path: ", tree_path)
    print("bc_types_path: ", bc_types_path)
    print("save_dir: ", save_dir)
    print("t_size: ", t_size)
    print("n_steps: ", n_steps)

    argv = [
        pul_1d_solver_path,
        "--input_tree_file", tree_path,
        "--input_bc_file", bc_types_path,
        "--output_tree_file", output_file,
        "--prescribe_resistances", "1",
        "--time_step_size", str(t_size),
        "--output_frequency", "1",
        "--number_of_iterations", "10",
        "--number_of_time_steps", str(n_steps)
    ]
    subprocess.run(argv, cwd=exe_dir, check=True)
    
    return
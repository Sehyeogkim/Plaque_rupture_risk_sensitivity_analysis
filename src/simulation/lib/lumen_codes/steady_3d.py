import os
import json
import multiprocessing
import utils.utils_lumen.runner as runner
import os
import shutil
import utils.utils_lumen.paths as utils_paths
from utils.utils_lumen.extract_wall_csv import extract_wall_data
from pathlib import Path
'''
# Total 4 stages
    - prepare ( make 3 folders and copy mesh,svpre, inp file, create flow file ,  )
    - run svpre solver
    - run svsolver nproc / 3 each. at the same time (parallel process.)
    - post procesing(extract the wall_peak.csv, wall_low.csv, wall_av.csv)
'''

class Steady_3d():
    
    def __init__(self, parent):
        '''
        parent: lumen_simulation class instance(which is the highest level class)
        '''
        self.st_3d_dir = Path(parent.st_3d_dir) # folder dir
        self.mesh_complete_dir = Path(parent.mesh_complete_dir) #mesh dir

        #define the parent path
        self.parent_svpre_dir = Path(parent.scripts_dir) / "model.svpre"
        self.parent_inp_dir = Path(parent.scripts_dir) / "solver.inp"

        #Generate 3 folders and 1d json file path
        self.peak_dir = self.st_3d_dir / "peak"
        self.low_dir = self.st_3d_dir / "low"
        self.av_dir = self.st_3d_dir / "av"

        #define the wall csv path (will be saved as the final result wall csv path)
        self.wall_peak_csv = Path(parent.wall_folder) / "wall_peak.csv"
        self.wall_low_csv = Path(parent.wall_folder) / "wall_low.csv"
        self.wall_av_csv = Path(parent.wall_folder) / "wall_av.csv"

        #define the parameters
        self.num_timesteps = 75
        self.time_step_size = 0.0005
        self.file_name = "rst"
        self.nproc = 15
        
   
    def create_three_folders(self):
        '''
        create 3 folders.
        '''
        self.peak_dir.mkdir(parents=True, exist_ok=True)
        self.low_dir.mkdir(parents=True, exist_ok=True)
        self.av_dir.mkdir(parents=True, exist_ok=True)

    def copy_mesh(self):
        '''
        copy mesh_dir folder into peak_dir, low_dir, av_dir (preserve original folder name)
        '''
        peak_mesh = self.peak_dir / "mesh-complete"
        low_mesh = self.low_dir / "mesh-complete"
        av_mesh = self.av_dir / "mesh-complete"
        shutil.copytree(self.mesh_complete_dir, peak_mesh, dirs_exist_ok=True)
        shutil.copytree(self.mesh_complete_dir, low_mesh, dirs_exist_ok=True)
        shutil.copytree(self.mesh_complete_dir, av_mesh, dirs_exist_ok=True)

    def create_flowfile(self, case_id: str, _1d_rst_json: str):
        '''
        create flowfile in the three folders from the given Q_peak, Q_low, Q_av.
        '''

        #open the _1d_rst_json file
        with open(_1d_rst_json, "r") as f:
            all_data = json.load(f)
        case_data = all_data[str(case_id)]  # Convert case_id to string for key access
        Q_peak = case_data["systolic"]["Q"]
        Q_low = case_data["diastolic"]["Q"]
        Q_av = case_data["mean"]["Q"]
        print(f"Q_peak: {Q_peak}, Q_low: {Q_low}, Q_av: {Q_av}")

        peak_flow_dir = self.peak_dir / "inflow.flow"
        low_flow_dir = self.low_dir / "inflow.flow"
        av_flow_dir = self.av_dir / "inflow.flow"
        utils_paths.create_flow_file(str(peak_flow_dir), Q_peak)
        utils_paths.create_flow_file(str(low_flow_dir), Q_low)
        utils_paths.create_flow_file(str(av_flow_dir), Q_av)

    def prepare_svpre(self, mesh_dir: str):
        #copy svpre file to peak_dir, low_dir, av_dir
        peak_svpre_dir = self.peak_dir / "model.svpre"
        low_svpre_dir = self.low_dir / "model.svpre"
        av_svpre_dir = self.av_dir / "model.svpre"
        utils_paths.create_svpre_file(str(peak_svpre_dir), mesh_dir)
        utils_paths.create_svpre_file(str(low_svpre_dir), mesh_dir)
        utils_paths.create_svpre_file(str(av_svpre_dir), mesh_dir)

    def prepare_inp(self, case_id: int, _1d_rst_json: str):
        
        '''
            #Fixe this parameters for the 3d simulation.
            Number of Timesteps: step_number
            Time Step Size: 0.0005
            Number of Timesteps between Restarts: 10
        '''

        peak_svsolver_dir = self.peak_dir / "solver.inp"
        low_svsolver_dir = self.low_dir / "solver.inp"
        av_svsolver_dir = self.av_dir / "solver.inp"
        shutil.copyfile(self.parent_inp_dir, peak_svsolver_dir)
        shutil.copyfile(self.parent_inp_dir, low_svsolver_dir)
        shutil.copyfile(self.parent_inp_dir, av_svsolver_dir)

        #Open json file(got from 1D simulation) -> get R_peak, R_low, R_av
        import json
        with open(_1d_rst_json, "r") as f:
            all_data = json.load(f)
        case_data = all_data[str(case_id)]  # Convert case_id to string for key access
        R_peak = case_data["systolic"]["R"]
        R_low = case_data["diastolic"]["R"]
        R_av = case_data["mean"]["R"]

        #Modify the resistance values on the inp file.
        chages_peak = {"Resistance Values:": R_peak, "Number of Timesteps:": self.num_timesteps, "Time Step Size:": self.time_step_size}
        chages_low = {"Resistance Values:": R_low, "Number of Timesteps:": self.num_timesteps, "Time Step Size:": self.time_step_size}
        chages_av = {"Resistance Values:": R_av, "Number of Timesteps:": self.num_timesteps, "Time Step Size:": self.time_step_size}
        utils_paths.revise_inp_file(str(peak_svsolver_dir), chages_peak)
        utils_paths.revise_inp_file(str(low_svsolver_dir), chages_low)
        utils_paths.revise_inp_file(str(av_svsolver_dir), chages_av)
     
    def steady_3d_run_svpre(self, svpre_path):
        '''
        run three svpre.
        '''
        sv_pre_processes = []
        for exe_dir in [self.peak_dir, self.low_dir, self.av_dir]:
            p = multiprocessing.Process(target=runner.run_svpre, args=(str(exe_dir), svpre_path))
            p.start()
            sv_pre_processes.append(p)

        for p in sv_pre_processes:
            p.join()
            if p.exitcode != 0:
                raise RuntimeError(f"svpre failed in {exe_dir} with exitcode {p.exitcode}")
        
    def steady_3d_run_svsolver(self, svsolver_path, nproc):
        
        '''
        run three svsolver.
        '''
        sv_solver_processes = []
        for exe_dir in [self.peak_dir, self.low_dir, self.av_dir]:
            p = multiprocessing.Process(target=runner.run_svsolver, args=(str(exe_dir), svsolver_path, nproc))
            p.start()
            sv_solver_processes.append(p)
        
        for p in sv_solver_processes:
            p.join()
            if p.exitcode != 0:
                raise RuntimeError(f"svsolver failed in {exe_dir} with exitcode {p.exitcode}")

    def steady_3d_run_svpost(self, svpost_path, nproc):
        '''
        run three svpost.
        '''
        sv_post_processes = []
        for exe_dir in [self.peak_dir, self.low_dir, self.av_dir]:
            p = multiprocessing.Process(target=runner.run_svpost, args=(str(exe_dir), svpost_path, nproc, self.num_timesteps, self.file_name))
            p.start()
            sv_post_processes.append(p)
        
        for p in sv_post_processes:
            p.join()
            if p.exitcode != 0:
                raise RuntimeError(f"svpost failed in {exe_dir} with exitcode {p.exitcode}")
    
    def save_wall_csv(self, exe_dir, wall_csv_path, file_name: str, num_timesteps: int):
        '''
        extract the wall.csv from the vtp file(given file name and number of timesteps)
        '''
        # ex) vtp_path = "peak/rst_00250.vtp" (file_name = rst)
        exe_dir_path = Path(exe_dir)
        vtp_path = exe_dir_path / f"{file_name}_{num_timesteps:05d}.vtp"
        extract_wall_data(str(vtp_path), str(wall_csv_path))
        return

    def extract_wall_csv(self):
        '''
        extract the wall.csv from the vtp file(given file name and number of timesteps)
        '''
        self.save_wall_csv(self.peak_dir, self.wall_peak_csv, self.file_name, self.num_timesteps)
        self.save_wall_csv(self.low_dir, self.wall_low_csv, self.file_name, self.num_timesteps)
        self.save_wall_csv(self.av_dir, self.wall_av_csv, self.file_name, self.num_timesteps)
        return
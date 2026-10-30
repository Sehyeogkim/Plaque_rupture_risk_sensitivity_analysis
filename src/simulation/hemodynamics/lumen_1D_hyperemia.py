'''

    This is the main script for the whole lumen simulation.

    #Purpose:
        Along with the Morphology and hemodynamics parameters, 
        extract the systolic, diastolica and average Total traction on the wall.
        These total traction will be used for the solid elastic simulation.

    #Input:
        - Morphology parameters(TBD)
        - Hemodynamics parameters(TBD)

    #Output:
        - wall_peak.csv
        - wall_low.csv
        - wall_av.csv

    #Procedures:
        1. CAD operation w/ Autodesk Inventor -> lumen.stp
        2. Lumen meshing w/ SIMMETRIX -> mesh-complete file.
        3. prepare and run the Q_Ramped 3D simulation (xyzts.dat, tree.dat, model.svpre, .inp file)

        4. From stage 3) we get slab.txt and Flow_history.dat 
        5. Run regression.py and ROM.py(WJ codes) to get the abc written tree.dat
        6. convert the tree.dat -> converted_tree.dat (Pointconverter code in our Synthetic tree generator code)

        7. Prepare the 1D simulation (make bc_types.dat,Pin, Coronary BC, Pmyo)
        (P_myo should be calculated from the Elastance)
        8. Run the 1D simulation
        9. Extract Psys, Pdia, Pavg and Qsys, Qdia, Qavg
        
        10. NOw run the Steady 3D simulation
        11. Extract wall_peak.csv, wall_low.csv, wall_av.csv

'''

import os
import json
from lumen_codes.cad_meshing import VesselCADModel, LumenMeshing
from lumen_codes.pulsatile_1d import Pulsatile_1D
from lumen_codes.steady_3d import Steady_3d
import utils.utils_lumen.paths as paths
import utils.utils_lumen.runner as runner
import utils.utils_lumen.validator as validator
import utils.utils_lumen.one_d_post as utils_one_d_post

from pathlib import Path
import multiprocessing

class lumen_total_simulation():
    
    def __init__(self, case_index: int, exec_dir: Path, scripts_dir: Path, parameter_csv_path: Path, config_path: Path, nproc = 30, hyperemia = False):
        self.case_index = case_index
        self.exec_dir = exec_dir # e.g.) model0827/case_{case_index}
        self.scripts_dir = scripts_dir # all the scripts and pre-data are saved on this folder.
        self.parameter_csv_path = parameter_csv_path
        self.hyperemia = hyperemia

        # Load config file
        with open(config_path, "r") as f:
            self.config = json.load(f)
        
        ############################################################
        #Main folder path.
        self.meshing_dir = self.exec_dir / "meshing"
        self.q_ramp_dir = self.exec_dir / "q_ramp"
             
        #pulsatile 1D directory (hyperemia or not)
        if self.hyperemia:
            self.pul_1d_dir = self.exec_dir / "pulsatile_1d_hyperemia" 
        else:
            self.pul_1d_dir = self.exec_dir / "pulsatile_1d"

        self.st_3d_dir = self.exec_dir / "steady_3d"
        self.wall_folder = self.exec_dir / "wall" #final result wall csv path      
        for d in [self.meshing_dir, self.q_ramp_dir, self.pul_1d_dir, self.st_3d_dir, self.wall_folder]:
            os.makedirs(d, exist_ok=True)
        ############################################################


        ############################################################
        #Interchaging data path.
        self.mesh_complete_dir = self.meshing_dir / "mesh-complete"
        self.tree_path = self.exec_dir / "tree.dat"
        self.xyzt_path = self.exec_dir / "xyzts.dat" # only used in q_ramped_3d.py
        self._1d_rst_json = self.pul_1d_dir / "_1d_rst.json"
        ############################################################

        
        ####create 4 stage instances.
        self.stage3 = Pulsatile_1D(self)
        self.stage4 = Steady_3d(self)

    def q_ramped_3d_prepare(self):
        '''
            (not used in here)
            Required:
            - mesh-complete
            - xyzts.dat(made at the lumen_meshing step)
            - steady.flow(-0.371)
            - model.svpre
            - solver.inp(Q ramped included)
        '''
        #copy the mesh, svpre, solver_ramp.inp from the scripts_dir to the q_ramp_dir
        #paths.copy_mesh(which_mesh_folder = self.mesh_complete_dir, dest_folder = self.q_ramp_dir) # meshing -> q_ramp (move the mesh folder)
        paths.create_svpre_file(svpre_file_path = str(Path(self.q_ramp_dir) / "model.svpre"), mesh_dir = self.meshing_dir)
        paths.copy_solver_inp(from_folder = str(self.scripts_dir), to_folder = str(self.q_ramp_dir), ramp = True) # solver_ramp.inp
        paths.create_flow_file(flow_file_path = str(Path(self.q_ramp_dir) / "inflow.flow"), Q = 0.371)
        
        #Should modify the init file along with the patient's data.
        changes = {
            "Number of Timesteps:": 500,
            "Number of Timesteps between Restarts:": 10,
            "Time Step Size:": 0.0002,
            "Resistance Values:": 134487.69,
            "Start Timestep for Flow Control:": 150,
            "Flow Control Scale Factor:": 0.002,
            "Maximum Flow Control Scale Factor:": 1.5
        }
        paths.revise_inp_file(input_file_path = str(Path(self.q_ramp_dir) /"solver_ramp.inp"), changes = changes)
        return

    ############################################################
    def pul_1d_settings(self):
        
        import pandas as pd
        df = pd.read_csv(str(self.parameter_csv_path))
        self.stage3.SBP = df.loc[self.case_index, "SBP"]
        self.stage3.DBP = (df.loc[self.case_index, "SBP"] - df.loc[self.case_index, "PP"])
        self.stage3.tau = df.loc[self.case_index, "tau"]
        print(f"SBP: {self.stage3.SBP} mmHg, DBP: {self.stage3.DBP} mmHg, tau: {self.stage3.tau}")
        
        #fixed parameters
        self.stage3.HR = 60
        self.is_LCA = True

        #1D simulation setting
        self.stage3.n_steps = 500
        self.stage3.t_size = 0.01

        return
    
    def pul_1d_prepare_abc_tree(self, slab_path: str, flowhist_path: str):
        
        '''
            we need
            - slab.txt (read this)
            - FlowHist.dat (read this)
            - outlet_information.dat (will create)
            - solver_ramp.inp (read this)
            - tree_abc.dat (will create)
            - bc_types.dat(P_in.dat, P_myo.dat, R_total (P_Mean)) (will create)
        '''

        #Create a outlet_information.dat file.
        outlet_info_path = Path(self.pul_1d_dir) / "outlet_information.dat"
        Pulsatile_1D.create_outlet_information_file(outlet_info_path)

        #Derive the abc written tree.dat -> abc written tree.dat
        ramped_inp_path = Path(self.q_ramp_dir) / "solver_ramp.inp"
        self.abc_tree_path = Pulsatile_1D.Reduced_order_modeling(tree_path = self.tree_path, 
                                            slab_path = slab_path, 
                                            flowhist_path = flowhist_path, 
                                            ramped_inp_path = ramped_inp_path,
                                            outlet_info_path = outlet_info_path,
                                            save_dir = str(self.pul_1d_dir))
        return

    def pul_1d_prepare_bc_types(self):
        
        '''
        1. create new P_in.data and calculate p_mean
        2. scale up the P_myo_norm upto 1.05 * SBP * 1333.22mmHg
        2(RCA) scale up the P_myo_norm upto 0.3 * SBP * 1333.22mmHg
        3. save each P_in.dat and P_myo.dat at the pul_1d_dir
        4. Create bc_types.dat from above data(P_in.dat, P_myo.dat, P_mean)
        '''

        #create P_in.dat, P_myo.dat and bc_types.dat
        self.stage3.create_P_in_dat()
        self.stage3.create_P_myo_dat(myo_scale_factor = 1.0)
        self.stage3.create_bc_types(hyperimia = self.hyperemia)
        print("P in, myo dat and bc_types.dat created\n")

        #convert tree_abc.dat -> converted_tree.dat (Point converter)
        converter_path = self.config["pul_1d"]["converter"] # point converter command path from config.json
        self.stage3.converted_tree_path = runner.run_tree_pt_converter(str(self.pul_1d_dir), converter_path, str(self.abc_tree_path))
        print("converted_tree.dat created\n")
        return

    def pulsatile_1d_run(self):

        #run the 1D simulation
        self.stage3.n_steps = 500
        self.stage3.t_size = 0.01
        pul_1d_solver = self.config["pul_1d"]["pul_solver"]
        runner.run_1d_simulation(str(self.pul_1d_dir), pul_1d_solver, str(self.stage3.converted_tree_path), str(self.stage3.bc_save_path), self.stage3.t_size, self.stage3.n_steps, str(self.stage3.save_dir))
        return    
    
    def pulsatile_1d_post(self):
        '''
        FFR : Fractional Flow Reserve
        FFR = mean(P_distal) / mean(P_proximal)

        P_proximal = P_inlet
        P_distal = 2 ~ 3 cm distal to the stenosis.

        Reference:
        https://www.sciencedirect.com/science/article/pii/S0735109716334301?utm_source=chatgpt.com
        '''

        #calculate proximal and distal point id. from xyzts.dat
        import pandas as pd
        df = pd.read_csv(self.parameter_csv_path)
        lesion_length = df.loc[self.case_index, "lesion_length"]
        z_distal = (lesion_length / 2) + 3 # (unit: cm)
        
        #FFR (mean(distal P) / mean(proximal P))
        pt_id_proximal = 0
        pt_id_distal = utils_one_d_post.find_pt_id_from_xyzts(self.xyzt_path, z_distal)
        print(f"pt_id_proximal: {pt_id_proximal}, pt_id_distal: {pt_id_distal}")

        #post processing -> .json file.
        utils_one_d_post.post_processing_1d(self.stage3.n_steps, self.stage3.t_size, str(self.stage3.save_dir), str(self._1d_rst_json), t_cycle = 1.0, start_pt_id = 0, end_pt_id = 49)
        
        if self.hyperemia:
            utils_one_d_post.post_processing_FFR(self.stage3.n_steps, self.stage3.t_size, str(self.stage3.save_dir), str(self._1d_rst_json), t_cycle = 1.0, pt_id_proximal = pt_id_proximal, pt_id_distal = pt_id_distal)
     
        #put case id at the _1d_rst.json file.
        # Read current JSON content
        with open(str(self._1d_rst_json), "r") as f:
            current_data = json.load(f)
        
        # Wrap current data with case_id key
        wrapped_data = {
            str(self.case_index): current_data
        }
        
        # Write back to file
        with open(str(self._1d_rst_json), "w") as f:
            json.dump(wrapped_data, f, indent=4)
        
        return

    def remove_1d_result(self):
        '''
        remove all the result folder except _1d_rst.json, tree.dat, treedata_abc.dat
        '''
        import shutil
        
        for item in os.listdir(self.pul_1d_dir):
            if item not in ["_1d_rst.json", "tree.dat", "treedata_abc.dat"]:
                item_path = os.path.join(self.pul_1d_dir, item)
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                else:
                    os.remove(item_path)
        
        return


def run_single_case(args):
    """Worker function for parallel execution. Runs one case end-to-end."""
    i, current_dir, scripts_dir, parameter_csv_path, config_path, slab_dir, flowhist_dir, hyperemia = args

    exec_dir = current_dir / f"fluid_data/case_{i}"
    slab_path = slab_dir / f"slab_{i}.txt"
    flowhist_path = flowhist_dir / f"FlowHist_{i}.dat"

    if not slab_path.exists() or not flowhist_path.exists():
        print(f"[Case {i}] slab_path or flowhist_path does not exist: {slab_path} or {flowhist_path}")
        return (i, False, "missing input files")

    try:
        print(f"[Case {i}] slab_path: {slab_path}, flowhist_path: {flowhist_path}")

        cvbml = lumen_total_simulation(
            case_index=i,
            exec_dir=exec_dir,
            scripts_dir=scripts_dir,
            parameter_csv_path=parameter_csv_path,
            config_path=config_path,
            hyperemia=hyperemia,
        )
        # if cvbml._1d_rst_json.exists():
        #     print(f"1D simulation already exists: {cvbml._1d_rst_json}")
        #     return (i, False, "already exists")
        cvbml.q_ramped_3d_prepare()

        #step 3. 1D simulation (Pulsatile 1D simulation)
        cvbml.pul_1d_settings()
        cvbml.pul_1d_prepare_abc_tree(slab_path=str(slab_path), flowhist_path=str(flowhist_path))
        cvbml.pul_1d_prepare_bc_types()
        cvbml.pulsatile_1d_run()
        cvbml.pulsatile_1d_post()
        #cvbml.remove_1d_result() # remove except _1d_rst.json, tree.dat, treedata_abc.dat

        print(f"[Case {i}] completed successfully")
        return (i, True, "success")

    except Exception as e:
        print(f"[Case {i}] failed with error: {e}")
        return (i, False, str(e))


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description="Lumen 1D ROM runner (rest / hyperemia).")
    parser.add_argument("--hyperemia", type=lambda x: str(x).lower() in ("true", "1", "yes"),
                        default=False, help="True -> hyperemia (adenosine, FFR); False -> rest (default)")
    parser.add_argument("--cases", type=int, nargs="*", default=[0, 1, 2, 3, 4],
                        help="case ids (default: 0 1 2 3 4)")
    parser.add_argument("--workers", type=int, default=5, help="parallel workers (default 5)")
    cli = parser.parse_args()
    print(f"[config] hyperemia={cli.hyperemia}  cases={cli.cases}  workers={cli.workers}")

    NUM_WORKERS = cli.workers

    #Main. directory path.
    current_dir = Path(os.path.dirname(os.path.abspath(__file__)))  # project root (script at 54_analysis root)
    pre_data_dir = current_dir / "pre_data"
    post_data_dir = current_dir / "post_data"

    parameter_csv_path = pre_data_dir / "parameter_0728.csv"
    scripts_dir = pre_data_dir / "scripts"
    config_path = pre_data_dir / "solver_path.json"

    slab_dir = post_data_dir / "slab"
    flowhist_dir = post_data_dir / "flowhist"

    # Build argument tuples for each case
    case_args = [
        (i, current_dir, scripts_dir, parameter_csv_path, config_path, slab_dir, flowhist_dir, cli.hyperemia)
        for i in cli.cases
    ]

    # Run all cases in parallel
    with multiprocessing.Pool(processes=NUM_WORKERS) as pool:
        results = pool.map(run_single_case, case_args)

    # Collect failures
    lumen_meshing_failed = [case_i for case_i, success, msg in results if not success]
    if lumen_meshing_failed:
        print(f"Failed cases: {lumen_meshing_failed}")

    with open(current_dir / "lumen_failed_meshing.txt", "w") as f:
        f.write(f"{lumen_meshing_failed}")
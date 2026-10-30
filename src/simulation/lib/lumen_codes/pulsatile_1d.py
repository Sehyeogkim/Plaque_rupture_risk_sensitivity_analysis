'''
This is a script for the pulsatile 1D simulation.

3 convert tree.dat -> output_tree.dat(Syntehtic solver)

0. make a pulsatile 1D folder.
1. Move the tree.dat, slab.txt, FlowHist.dat to the pulsatile 1D folder.
1a make outlet_information.dat file just writen as 1 50 0.
2. Make a slopes.dat file. (regression_abc.py)
    (requires inp file had used on the Q_ramped 3D simulation)
4. Now run the Reduced order model.py -> get a,b,c written output_tree.dat

5. prepare bc_types.dat file.(P_in, Coronary Bc, P_myo)
- generate P_in.dat along with the hemodynamic parameters.
- generate Coronary Bc.dat along with the hemodynamic parameters.
- generate P_myo.dat from the Elastance.dat and the hemodynamic parameters.
- Combine them into bc_types.dat file.

6. run 1D simulation.

7. Post processing -> 1D post.py
'''


import os
import utils.utils_lumen.regression as utils_reg
import utils.utils_lumen.ROM as utils_ROM
from pathlib import Path
import numpy as np

class Pulsatile_1D():
    
    def __init__(self, parent):
        '''
        parent: lumen_simulation class instnaace(which is the highest level class)
        '''

        self.exe_dir = Path(parent.pul_1d_dir) # folder dir
        
        # the data will be calculated
        self.mean_pressure = 0 # mmHg.

        #Nomralized pre - data file path
        self.E_norm_path = parent.scripts_dir / "Elastance_norm.dat"
        self.inflow_path = parent.scripts_dir / "inflow.dat"

        # Heart model parameters (for P_myo ODE computation)
        self.E_max = 2.0 * 1333.22  # max elastance (mmHg -> dyne/cm^2)
        self.cardiac_output = 5 * 1000 / 60  # 5 L/min -> cc/s
        self.atrial_elastance = 250.0
        self.atrial_resistance = 1.0
        self.atrial_inductance = 10.0
        self.ventricular_resistance = 1.0
        self.ventricular_inductance = 5.0
        
        #Output file path
        self.P_in_data_path = self.exe_dir / "P_in.dat"
        self.P_myo_data_path = self.exe_dir / "P_myo.dat"
        self.bc_save_path = self.exe_dir / "bc_types.dat"

        #these files will be created.
        self.outlet_info_path = self.exe_dir / "outlet_information.dat"
        self.abc_tree_path = None
        self.converted_tree_path = None

        #default values will be changed in the lumen_main.py
        self.SBP = 120 * 1333.22 # mmHg
        self.DBP = 80 * 1333.22 # mmHg
        self.HR = 60 # bpm
        self.P_ratio = 1.0
        self.n_steps = 500
        self.t_size = 0.01
        self.tau = 0.2 # 0.1 ~ 0.3s

        #all the 1d simulation result will be saved in this directory
        self.save_dir = self.exe_dir / "result"
        os.makedirs(self.save_dir, exist_ok=True)

    @staticmethod
    def create_outlet_information_file(outlet_info_path):
        '''
        Make a outlet_information.dat file.
        '''
        with open(outlet_info_path, "w") as f:
            f.write("1 49 0")
        return

    @staticmethod
    def Reduced_order_modeling(tree_path, slab_path, flowhist_path, ramped_inp_path, outlet_info_path, save_dir):
        '''
        Run the Reduced order modeling -> create self.abc_tree_path
        '''
        slab_p_path, variable_path, slopes_path = utils_reg.regression(slab_path, flowhist_path, ramped_inp_path, save_dir)
        abc_tree_path = utils_ROM.ROM_main(tree_path, outlet_info_path, slab_p_path, variable_path, slopes_path, save_dir)
        return abc_tree_path
    
    def create_P_in_dat(self) -> None:
        '''
        Goal: create P_in.dat from the SBP, DBP, HR and inflow.dat
        '''
        from utils.utils_lumen.pressure_in_gen import generate_pressure_windkessel
        generate_pressure_windkessel(
            SBP=self.SBP, DBP=self.DBP, tau=self.tau, HR=self.HR,
            inflow_path=self.inflow_path,
            save_path=self.P_in_data_path,
            plot=False,
        )
        return 
    
   
    def create_P_myo_dat(self, myo_scale_factor: float = 0.8) -> None:
        '''
        Create P_myo.dat by solving the lumped-parameter heart ODE model.
        '''
        from utils.utils_lumen.pressure_myo_gen import SOLVE_HEART_ODE
        SOLVE_HEART_ODE(
            P_in_path=self.P_in_data_path,
            elastance_path=self.E_norm_path,
            myo_scale_factor=myo_scale_factor,
            P_myo_path=self.P_myo_data_path,
            plot=True,
        )
        return


    def create_bc_types(self, hyperimia = False) -> None:
        '''
            mean_pressure: mean pressure in Pa
            is_LCA: True -> LCA, False -> RCA
    
            This code is to set the bc_tyes.dat file for the 1D ROM simulation.
            bc_types.dat sturcture:
            1. P_inlet time series data -> should varry along the hemodynamic parameters.
            2. coronary BC data
            3. P_myo time series data
        
            1. read the P_in.dat
            2. open new .dat file and write 0 DirichletPressure 1001 1 0
            3. write the time series data to the file in a row.
            4. write 49 NeumannCoronary 1001 4 12
            5. write the calculated parameters
            6. write the P_myo time series data

            -> create self.bc_save_path

            #Reference for the Q mean = 29.7ml/min in the LAD (PC-MRI measurement)
            Johnson, K., Sharma, P., & Oshinski, J. (2007). 
            Coronary artery flow measurement using navigator echo gated phase contrast magnetic resonance velocity mapping at 3.0 T. 
            Journal of Biomechanics, 41(3), 595–602. https://doi.org/10.1016/j.jbiomech.2007.10.010
        '''

        # Calculate resistances and capacitances
        Q_mean = 0.495 # 29.7ml/min = 29.7/60 cc/s = 0.495 cc/s
        P_mean = 93 * 1333.22 # mmHg -> dynes/cm^2
        R_total = float(P_mean / Q_mean)
        C_total = 0.36 # per-outlet test (3.6e-5/~10 outlets)
        
        #adenosine protocol
        if hyperimia:
            R_total = R_total / 3.5
        else:
            R_total = R_total
        
        R_a, R_a_micro, R_v = 0.32 * R_total, 0.52 * R_total, 0.16 * R_total
        C_a, C_im = 0.11 * C_total, 0.89 * C_total
        R_venous = R_v_micro = R_v / 2

        # Load P_in.dat and P_myo.dat
        P_in_data = np.loadtxt(self.P_in_data_path, delimiter=' ', dtype=float)
        P_myo_data = np.loadtxt(self.P_myo_data_path, delimiter=' ', dtype=float)
        assert len(P_in_data) == len(P_myo_data), "P_in_data and P_myo_data must have the same length"

        # Open new boundary condition file
        with open(self.bc_save_path, 'w') as f:
            
            # Write inlet boundary condition (point 0)
            f.write("0 DirichletPressure 1001 1 0\n") # 1001 -> # of points, 1 -> dimension
            
            # Write P_in time series data
            for i in range(len(P_in_data)):
                f.write(f"{P_in_data[i,0]:.6f} {P_in_data[i,1]:.6f}\n")
            
            # Write outlet boundary condition (point 49)
            f.write("49 NeumannCoronary 1001 4 12\n")
            
            # Write the 12 coupled parameters
            f.write(f"arterial_resistance {R_a:.6f}\n")
            f.write(f"arterial_compliance {C_a:.4f}E-5\n")
            f.write(f"arterial_microcirculation_resistance {R_a_micro:.6f}\n")
            f.write(f"myocardial_compliance {C_im:.4f}E-5\n")
            f.write(f"venous_microcirculation_resistance {R_v_micro:.6f}\n")
            f.write(f"venous_resistance {R_venous:.6f}\n")
            f.write(f"venous_compliance 1.00E-5\n")
            f.write(f"venous_pressure 0\n") # 20000 -> 0
            f.write(f"total_compliance 0\n")
            f.write(f"arterial_resistance_ratio 0\n")
            f.write(f"venous_resistance_ratio 0\n")
            f.write(f"total_resistance 0\n")
            
            # Write P_myo time series data
            for i in range(len(P_myo_data)):
                f.write(f"{P_myo_data[i,0]:.6f} {P_myo_data[i,1]:.6f}\n")
        
        return

        
 
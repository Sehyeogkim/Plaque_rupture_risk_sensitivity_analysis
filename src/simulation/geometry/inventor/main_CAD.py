import math
import win32com.client as wc
import os
import sys
import pandas as pd
import traceback
from pathlib import Path

# Force UTF-8 console so the util print statements (which contain the "✅" emoji)
# don't crash on the Korean cp949 code page.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

PREEXISTING_LUMEN_ROOT = Path(
    r"C:\Users\jeff\project_inventor\30_sensitivity_analysis_0303\geo_0303"
)


class VesselModel:

    def __init__(self, 
                PI, 
                alpha, 
                lipid_length_ratio, 
                fc_av_th, 
                d_fc_ca, 
                fraction, 
                ca_axial_skewness, 
                ca_shoulder_skewness,
                ca_axial_strength, 
                ca_shoulder_strength,
                DOS, lesion_length, 
                lumen_axial_skewness,
                r_max = 0.1,  thick_ratio = 0.2, case_id = 1):
        
        '''
        @ Latest Update 0813 Jeff
        - Vornoli Tesslation method parameters(0731).
        - New Lumen Morphology parameters are added(0813).
        '''
        ###3
        self.case_id = case_id

        ############################################################
        #Lumen parameters.
        self.r_max = r_max
        self.r_min = (1 - DOS) * r_max # DOS : 1 - Dmin / Dmax (Diameter based)
        self.lesion_length = lesion_length
        self.lumen_skewness = lumen_axial_skewness  # [-1, 1]

        #Solid parameters.
        self.PI = PI 
        self.thick_ratio = thick_ratio # WALL_THICKNESS/ RADIUS = 0.2, DEFAULT

        #Lipid parameters.
        self.alpha = alpha
        self.lipid_length = lipid_length_ratio * self.lesion_length

        # Distance between the lipid circle at z = 0 and the solid wall. (for the calculatiozn of the stenosis area)
        self.wall_thickness = thick_ratio * self.r_max
        self.lipid_wall_thickness = self.wall_thickness #same as original wall thickness

        #Fibrous cap parameters.
        self.fc_av_th = fc_av_th 

        #Calcifcaiton parameters. (Vornoli Tesslation method)
        self.d_fc_ca = d_fc_ca
        self.fraction = fraction

        self.ca_axial_skewness = ca_axial_skewness
        self.ca_shoulder_skewness = ca_shoulder_skewness

        self.ca_axial_strength = ca_axial_strength
        self.ca_shoulder_strength = ca_shoulder_strength
        ############################################################




        #Variables for the lumen skewness.
        self.z_start = -2.0 #cm 
        self.z_end   =  8.0 #cm
        self.z_lipid  = self.lipid_length / 2
        self.z_lesion = self.lesion_length / 2
        self.z_peak  = self.lumen_skewness * self.z_lesion # min R, z coordinate (varies depend on the lumen skewness)

        self.T        = self.lesion_length #Period of the sinousal functions that frequently defined in the CAD codes.
        self.r_ex     = self.r_max * (1 + self.thick_ratio) # Normal vessel Radius.
        self.r_pos    = math.sqrt(self.PI) * self.r_ex      # Abnormal vesssel Radius by PI.
        
        pass

    def connect_to_inventor(self):

        ''' !!!!Important!!!!  Make sure Autodesk inventor is operating bf running this python script'''
        #Connect to Autdesk inventor
        self.inv = wc.GetActiveObject('Inventor.Application')
        self.close_CAD = True

        if self.inv.Visible:
            print("Autodesk inventor is running.")
        else:
            print("Error: Autodesk inventor is not running.")
            raise RuntimeError("Autodesk inventor is not running.")
        
        pass


#updated 0826 jeff
def CAD_instance_from_idx(case_idx: int, case_csv_path: str):
    '''
    Create a CAD instance on the sepcific case_idx
    return the CAD instance.
    '''
    para_dict = pd.read_csv(case_csv_path).iloc[case_idx].to_dict()

    stenosis_model = VesselModel(
        PI = para_dict["PI"], 
        alpha=para_dict["alpha"], 
        lipid_length_ratio=para_dict["lipid_length_ratio"], 
        fc_av_th=para_dict["fc_av_th"],  
        d_fc_ca=para_dict["d_fc_ca"], 
        fraction=para_dict["fraction"], 
        ca_axial_skewness=para_dict["ca_axial_skewness"], 
        ca_shoulder_skewness=para_dict["ca_shoulder_skewness"],
        ca_axial_strength=para_dict["ca_strength_ratio"], 
        ca_shoulder_strength = 1.0,
        DOS=para_dict["DOS"],
        lesion_length=para_dict["lesion_length"],
        lumen_axial_skewness=para_dict["lumen_axial_skewness"],
        r_max = 0.1,  thick_ratio = 0.2, case_id = case_idx
    )        

    return stenosis_model

#run cad operation.
def Autodesk_inventor_operation(CAD_instance: VesselModel, save_folder_path: str, close_CAD: bool = True):

    from utils.solid_centerline import solid_step
    from utils.fc import fc_step, fc_offset_step
    from utils.lipid_main import lipid_step
    from utils.lumen_centerline import lumen_step

    # 1) lumen: already generated in the lumen-only pass -> REUSE existing lumen.ipt
    #    (do NOT regenerate; that would re-break the 24 guideline-rescued cases)
    lumen_ipt_path = os.path.join(save_folder_path, "lumen.ipt")
    CAD_instance.lumen_path = lumen_step(CAD_instance, save_folder_path)
    if not os.path.exists(lumen_ipt_path):
        raise FileNotFoundError(f"existing lumen.ipt not found (run lumen first): {lumen_ipt_path}")

    # 2) solid (centerline)
    CAD_instance.solid_path = solid_step(CAD_instance, save_folder_path)

    # 3) fc  (lumen offset by fc_av_th) -> produces fc.ipt required by lipid
    CAD_instance.fc_path = fc_step(CAD_instance, save_folder_path, lumen_ipt_path)

    # 4) fc_offset  (lumen offset by d_fc_ca + fc_av_th)
    CAD_instance.fc_offset_path = fc_offset_step(CAD_instance, save_folder_path, lumen_ipt_path)

    # 5) lipid  (lofted, then boolean-cut by fc.ipt)
    CAD_instance.lipid_path = lipid_step(CAD_instance, save_folder_path)

    return None



if __name__ == "__main__":

    #Define folder name.
    current_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir1 = os.path.join(current_dir, "geo_0723")
    os.makedirs(save_dir1, exist_ok=True)

    #read case csv (full 384-case Sobol DOE)
    case_csv_path = os.path.join(current_dir, "parameter_0728.csv")
    num_cases = len(pd.read_csv(case_csv_path))
    error_index_list = []

    START_CASE = 5   # cases 0..4 already done; continue from 5 through the end
    
    
    for i in [438]:
        #CAD_main
        try:
            CAD_instance = CAD_instance_from_idx(i, case_csv_path) # Create a CAD instance
            CAD_instance.connect_to_inventor() # connect to inventor

            save_case_folder_path = os.path.join(save_dir1, f"case_{i}")
            os.makedirs(save_case_folder_path, exist_ok=True)

            Autodesk_inventor_operation(CAD_instance, save_case_folder_path, close_CAD=True) # run the CAD operation
            print(f"Case {i} completed")
            CAD_instance.inv.Documents.CloseAll(True)
        except Exception as e:
            print(f"Error: {e}")
            print(f"Error: {traceback.format_exc()}")
            CAD_instance.inv.Documents.CloseAll(True)
            error_index_list.append(i)
        print(f"SO FAR FAILED CASES: {error_index_list}")
    print("ALL COMPLETE")

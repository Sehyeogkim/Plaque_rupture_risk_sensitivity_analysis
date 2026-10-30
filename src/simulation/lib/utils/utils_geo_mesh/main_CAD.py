import math   
import pandas as pd

class VesselModel:

    def __init__(self, 
                PI, alpha, 
                lipid_length_ratio, fc_av_th, 

                d_fc_ca, fraction, 
                ca_axial_skewness, ca_shoulder_skewness,

                ca_axial_strength, ca_shoulder_strength,
                
                #New Lumen Morphology parameters
                DOS, lesion_length, lumen_axial_skewness,

                r_max = 0.1,  thick_ratio = 0.2):

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
        
        from utils.utils_geo_mesh.utils_CAD import find_ab
        self.z_lipid_L, self.z_lipid_R = find_ab(self) # Function in the utils.CAD

        pass

#updated 0826 jeff
def CAD_instance_from_idx(case_idx: int, parameter_csv_path: str):
    '''
    Create a CAD instance on the sepcific case_idx
    return the CAD instance.
    '''
    para_dict = pd.read_csv(parameter_csv_path).iloc[case_idx].to_dict()

    stenosis_model = VesselModel(
        PI = float(para_dict["PI"]), 
        alpha = float(para_dict["alpha"]), 
        lipid_length_ratio = float(para_dict["lipid_length_ratio"]), 
        fc_av_th = float(para_dict["fc_av_th"]),  
        d_fc_ca = float(para_dict["d_fc_ca"]), 
        fraction = float(para_dict["fraction"]), 
        ca_axial_skewness = float(para_dict["ca_axial_skewness"]), 
        ca_shoulder_skewness = float(para_dict["ca_shoulder_skewness"]),
        ca_axial_strength = float(para_dict["ca_strength_ratio"]), 
        ca_shoulder_strength = 1.0,
        DOS = float(para_dict["DOS"]),
        lesion_length = float(para_dict["lesion_length"]),
        lumen_axial_skewness = float(para_dict["lumen_axial_skewness"]),
        r_max = 0.1,  thick_ratio = 0.2
    )        

    return stenosis_model
    






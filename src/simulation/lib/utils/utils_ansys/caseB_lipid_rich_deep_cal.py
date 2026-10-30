import os
import pyvista as pv
import pandas as pd
import math
import utils.utils_geo_mesh.utils_CAD as utils_CAD
import numpy as np

'''
caseB: lipid rich w/ deep calcification
'''


def Voronoi_tesselation_KDTREE(model, mesh, save_cal_vtu = True, console = True):
    '''
    KDTREE algorithm.
    input:
    mesh: read by meshio
    model: class instance having (case_index and parameter_csv_path)
        self.case_index = case_index # case id, should be int.
        self.parameter_csv_path = parameter_csv_path

    output: new unstructured grid with calcification with physical tag 8
    '''

    #read the parameters from the vessel_model instance.
    fraction = model.vessel_model.fraction
    skew_axial = model.vessel_model.ca_axial_skewness
    skew_shoulder = model.vessel_model.ca_shoulder_skewness
    fc_av_th = model.vessel_model.fc_av_th
    d_fc_ca = model.vessel_model.d_fc_ca
    strength_axial = model.vessel_model.ca_axial_strength
    strength_circum = model.vessel_model.ca_shoulder_strength

    print("================================================")
    print(f"fraction: {fraction}")
    print(f"skew_axial: {skew_axial}")
    print(f"skew_shoulder: {skew_shoulder}")
    print(f"fc_av_th: {fc_av_th}")
    print(f"d_fc_ca: {d_fc_ca}")
    print(f"strength_axial: {strength_axial}")
    print(f"strength_circum: {strength_circum}")
    print("================================================")

    #tag for the subdomain, lipid, and calcification, defined in gmsh.
    subdomain_tag = 10 #subdomain that calcfication will propagate inside.
    lipid_tag = 2
    ca_tag = 8
    
    subdomain_mask = (mesh.cell_data["gmsh:physical"] == subdomain_tag) & (mesh.celltypes == 24)
    subdomain_cells_indices  = np.where(subdomain_mask)[0]
    
    #2. Now derive (# of cal) = (# of tetra) * fraction
    cal_num = round(len(subdomain_cells_indices) * fraction)

    #3. get the z_max of the subdomain from the gmsh.
    submesh = mesh.extract_cells(subdomain_cells_indices)
    
    #inside the subdomain where ca can exist, normalize (z_min, z_max) -> (0, 1)
    z_min = submesh.points[:, 2].min()
    z_max = submesh.points[:, 2].max()
    z_seed = z_min + (z_max - z_min) * skew_axial

    #4. calculate θ_seed from lipid_half_angle(z = z_seed) and the distance from the lumen_centere
    θ_lipid_arc_half = utils_CAD.alpha_theta(model.vessel_model, z_seed) #rad
    θ_seed_half = θ_lipid_arc_half * skew_shoulder # this value can be negative.

    #5. calculate the x, y coordinate of the seed_point
    # distance between the lumen_center and the seed_point 
    # (big assumption seed point is the closests to the lumen_center in the subdomain, divided by d_fc_ca)
    d_lumen_seed = utils_CAD.radius_lumen(model.vessel_model, z_seed) + fc_av_th + d_fc_ca
    x_seed = d_lumen_seed * math.sin(θ_seed_half)
    y_seed = d_lumen_seed * math.cos(θ_seed_half) + utils_CAD.y_center_lumen(model.vessel_model, z_seed)
    
    if console:
        print(f"x_seed: {x_seed}, y_seed: {y_seed}, z_seed: {z_seed}\n")
    
    #7. Find the tetra element that contains the seed_point.
    seed_point = np.array([x_seed, y_seed, z_seed])
    seed_cell_id = -1
    status = ""

    if submesh.n_cells > 0:
        local_cell_id = submesh.find_containing_cell(seed_point)
        original_ids_in_submesh = submesh.cell_data['vtkOriginalCellIds']
        
        if local_cell_id != -1:
            status = "point is in the subdomain"
            seed_cell_id = original_ids_in_submesh[local_cell_id]
        else:
            status = "closest cell in the subdomain"
            local_closest_cell_id = submesh.find_closest_cell(seed_point)
            seed_cell_id = original_ids_in_submesh[local_closest_cell_id]

    if console:
        print(f"SEED_CELL_ID: {seed_cell_id}")
        print(f"SEED_STATUS: {status}")
    
    '''
    so far we confirm the seed_point location and the seed_cell_id which contains the seed_point.
    '''



    ############################################################
    ##### Propagation algorithm starts from the KDTREE #########
    ############################################################
    
    import time
    from scipy.spatial import cKDTree
    start_time = time.time()
    calibrated_cells_original_ids = []


    if cal_num > 0:
        print(f"\nStarting KD-Tree based propagation...")

        # --- Parameters to control propagation shape ---
        # When weight is larger, distance is scaled down in that direction,
        # leading to more propagation in that direction
        # e.g) for more axial propagation,  axial_weight_factor should be larger.
        axial_weight_factor = strength_axial
        circum_weight_factor = strength_circum

        # --- 1. prepare the data: extract the cell centers of the subdomain ---
        subdomain_cell_centers = mesh.cell_centers().points[subdomain_cells_indices]

        # --- 2. transform the coordinates: apply the weight to the coordinates ---
        # to make the general euclidean distance search to be like the weighted distance search,
        # we scale the coordinates by sqrt(weight)
        # Distance^2 ≈ w_c*(dx^2+dy^2) + w_a*dz^2
        w_axial_sqrt = np.sqrt(axial_weight_factor)
        w_circum_sqrt = np.sqrt(circum_weight_factor)
        
        transformed_centers = np.copy(subdomain_cell_centers)
        transformed_centers[:, 0:2] *= w_circum_sqrt # x, y (circumferential direction)
        transformed_centers[:, 2]   *= w_axial_sqrt   # z (axial direction)

        # --- 3. create the KDTree ---
        # create the KDTree with the transformed cell centers
        kdtree = cKDTree(transformed_centers)

        # --- 4. transform the seed point and query ---
        # transform the seed point with the same weight
        transformed_seed_point = np.copy(seed_point)
        transformed_seed_point[0:2] *= w_circum_sqrt
        transformed_seed_point[2]   *= w_axial_sqrt
        
        # query the cal_num nearest neighbors in the KD-Tree
        distances, indices = kdtree.query(transformed_seed_point, k=cal_num)

        # --- 5. map the result: convert the original cell IDs ---
        valid_indices = indices[np.isfinite(distances)] # filter the valid indices
        calibrated_cells_original_ids = subdomain_cells_indices[valid_indices]

        end_time = time.time()
        print(f"KD-Tree search completed in {end_time - start_time:.4f} seconds.")
        print(f"Selected {len(calibrated_cells_original_ids)} cells for calcification.")
        
        # Assign calcification tag (8) to the selected cells
        mesh.cell_data["gmsh:physical"][calibrated_cells_original_ids] = ca_tag # allocate new ca_index = 8


    else:
        raise ValueError("cal_num is zero. No cells will be calcified.")



    #9. First, we set the subdomain tag as 10, and then we divide the domain into two parts: calcification and lipid.
    # Now, we need to set the corresponding tag for the lipid and cal as lipid_tag = 2,  cal_tag = 8
    lipid_mask = mesh.cell_data["gmsh:physical"] == subdomain_tag # 10
    mesh.cell_data["gmsh:physical"][lipid_mask] = lipid_tag       # 2

    #####################################################
    ##### 10. Calculate the dependent variables #########
    #####################################################

    #10.1. Total fraction
    ca_mask = (mesh.cell_data["gmsh:physical"] == ca_tag) & (mesh.celltypes == 24) # Calcification mask
    ca_cells_indices  = np.where(ca_mask)[0]
    ca_mesh = mesh.extract_cells(ca_cells_indices)
    if save_cal_vtu:
        ca_mesh.save(os.path.join(model.ansys_exec_dir, f"case_B_cal_mesh.vtu"))
    
    lipid_mask = (mesh.cell_data["gmsh:physical"] == lipid_tag) & (mesh.celltypes == 24) # Lipid mask
    lipid_cells_indices = np.where(lipid_mask)[0]

    total_fraction = len(ca_cells_indices) / (len(ca_cells_indices) + len(lipid_cells_indices))
    if console:
        print(f"total_fraction: {total_fraction}")

    
    #10.2. get the z_max of the subdomain from the gmsh.
    ca_mesh = mesh.extract_cells(ca_cells_indices)
    z_max = ca_mesh.points[:, 2].max()
    z_min = ca_mesh.points[:, 2].min()
    ca_length = z_max - z_min
    if console:
        print(f"ca_length: {ca_length}")


    #10.3. Maximum cal arc angle
    '''
    change the algorithm 20251010
    
    so, there are calcification cells already defined. we are going to slice the cal_mesh parallel to the xy plane along the z axis.
    z_slice = np.linspace(z_min, z_max, 100)

    and then, calculate the max arc angle on each sliced plane.
    the funcion f(z) will derive the arc angle.

    and then, find the z value with the maximum arc angle, z_max_arc
    and the dependent variable that we are looking for is f(z_max_arc) = ca_arc_max
    '''
    
    def f(z):
        '''
        output the arc angle for the given z plane parallel to the x-y plane
        '''
        z_filter = np.abs(ca_mesh.points[:, 2] - z) < 0.01
        filtered_points = ca_mesh.points[z_filter]
        if not len(filtered_points) > 0:
            raise ValueError("No points found within z tolerance for cal arc angle calculation")
        
        lumen_center = np.array([0, utils_CAD.y_center_lumen(model.vessel_model, z), z])
        
        #create vectors from lumen_center to filtered points
        vectors = filtered_points - lumen_center
        vectors_2d = vectors[:, :2]  # z coordinate is removed

        #the angle between the vectors and the y_axis
        angles = np.arctan2(vectors_2d[:, 0], vectors_2d[:, 1])
        arc_angle = np.max(angles) - np.min(angles)
        return arc_angle
    
    z = np.linspace(z_min, z_max, 100)
    arc_angles = np.array([f(z_val) for z_val in z])
    
    # Find the z value with maximum arc angle
    max_arc_idx = np.argmax(arc_angles)
    z_max_arc = z[max_arc_idx]
    ca_arc_max = arc_angles[max_arc_idx]
    ca_arc_max_degrees = np.degrees(ca_arc_max)
    
    if console:
        print(f"z_max_arc: {z_max_arc:.4f}")
        print(f"ca_arc_max: {ca_arc_max_degrees:.4f} degrees")
        lumen_center = np.array([0, utils_CAD.y_center_lumen(model.vessel_model, z_max_arc), z_max_arc])
        print(f"lumen_center: {lumen_center}")

    

    ################################################
    ##### 11. Save the dependent variables #########
    ################################################
    json_path = os.path.join(model.ansys_exec_dir, f"dependent_variables.json")
    # Save total_fraction, ca_length, and ca_arc to a JSON file at model.json_path
    import json
    result_dict = {
        "total_fraction": round(total_fraction, 4),
        "ca_length": round(ca_length, 4),
        "ca_arc": round(ca_arc_max_degrees, 4)
    }
    with open(json_path, "w") as file:
        json.dump(result_dict, file, indent=4)

    return mesh
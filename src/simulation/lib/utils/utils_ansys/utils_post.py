import numpy as np
import math
import meshio
import pyvista as pv

'''
@updated 0905 jeff
- Multiply the amp_vtu and peak_vtu for the R3 and R4 analysis.
'''

def generate_mean_vtu(peak_vtu_path: str, low_vtu_path: str, mean_vtu_path: str):
    '''
    generate amp_vtu file from peak_vtu and low_vtu.
    amp_vtu has nodal EQV data and element EQV data as well.

    !!!! amplitude is only relevant for the Stress not the strain.


    point_data={"Group": tags, "EQV_strain": nodal_eqv_strain, "Principal_strain": nodal_princ_strain, 
                "Displacement": nodal_disp, "EQV_stress": nodal_eqv_stress, "Principal_stress": nodal_princ_stress}
    '''
    point_array_name_list = ["Principal_strain", "Principal_stress", "EQV_strain", "EQV_stress"]
    peak_mesh = pv.read(peak_vtu_path)
    low_mesh = pv.read(low_vtu_path)

    mean_mesh = peak_mesh.copy(deep=True)

    for name in point_array_name_list:

        #Make sure peak and low mesh have all the point_array_name_list
        if name not in peak_mesh.point_data or name not in low_mesh.point_data:
            raise KeyError(f"Missing point array '{name}' in peak or low mesh.")
    
        mean_vals = (np.asarray(peak_mesh.point_data[name]) + np.asarray(low_mesh.point_data[name])) / 2
        mean_mesh.point_data[name] = mean_vals   # overwrite with mean value under the SAME name


    #Save the amp_vtu file
    mean_mesh.save(mean_vtu_path)

    return mean_vtu_path

def generate_amp_vtu(peak_vtu_path: str, low_vtu_path: str, amp_vtu_path: str):
    '''
    generate amp_vtu file from peak_vtu and low_vtu.
    amp_vtu has nodal EQV data and element EQV data as well.

    !!!! amplitude is only relevant for the Stress not the strain.


    point_data={"Group": tags, "EQV_strain": nodal_eqv_strain, "Principal_strain": nodal_princ_strain, 
                "Displacement": nodal_disp, "EQV_stress": nodal_eqv_stress, "Principal_stress": nodal_princ_stress}
    '''
    point_array_name_list = ["Principal_strain", "Principal_stress", "EQV_strain", "EQV_stress"]
    peak_mesh = pv.read(peak_vtu_path)
    low_mesh = pv.read(low_vtu_path)

    amp_mesh = peak_mesh.copy(deep=True)   # start from peak, then overwrite arrays with amplitudes
    #Make sure peak and low mesh have all the point_array_name_list
    for name in point_array_name_list:
        if name not in peak_mesh.point_data or name not in low_mesh.point_data:
            raise KeyError(f"Missing point array '{name}' in peak or low mesh.")

        amp_vals = np.asarray(peak_mesh.point_data[name]) - np.asarray(low_mesh.point_data[name])
        amp_mesh.point_data[name] = amp_vals   # overwrite with amplitude under the SAME name

    
    #Save the amp_vtu file
    amp_mesh.save(amp_vtu_path)

    return amp_vtu_path

def Extract_Nodal_EQV(mapdl, cm_name):
    '''
    @update 05/28 jeff
    Input: mapdl instance, and component name that nodes are grouped.

    Output: node_ids, node_coord, nodal_eqv_stress

    !!Caution, 
    - qudaratic nodes do not have EQV stress.
    '''

    #Node info
    mapdl.cmsel(type_= 'S', name = cm_name, entity = 'NODE')
    node_ids = np.array(mapdl.mesh.nnum) #wall node ids.
    node_coord = np.array(mapdl.mesh.nodes) #wall node point coordinates.
    nodal_eqv_stress = np.array(mapdl.post_processing.nodal_eqv_stress())

    # Get mask of non-zero stress values
    nonzero_mask = nodal_eqv_stress != 0
    
    # Apply mask to all arrays to keep only nodes with non-zero stress(Eliminate middle points info)
    node_ids = node_ids[nonzero_mask]
    node_coord = node_coord[nonzero_mask]
    nodal_eqv_stress = nodal_eqv_stress[nonzero_mask]

    print("node_ids", node_ids)
    print("node_coord", node_coord)
    print("nodal_eqv_stress", nodal_eqv_stress)

    return node_ids, node_coord, nodal_eqv_stress



def get_node_index(node_ids, node_id):
    """Return the index of node_id in node_ids array."""
    return np.where(node_ids == node_id)[0][0]

def triangle_area(coords):
    """Calculate area of a triangle given 3 (x, y, z) coordinates."""
    a = coords[1] - coords[0]
    b = coords[2] - coords[0]
    return 0.5 * np.linalg.norm(np.cross(a, b))

def element_centroid(coords):
    """Calculate centroid of a triangle given 3 (x, y, z) coordinates."""
    return np.mean(coords, axis=0)

def eid_list_sel(mapdl, eid_list):
    '''
    @update 05/27 jeff
    (usd on post procssing part.)
    This function is defined in order to select the element IDs in the eid_list.
    '''
    commands = []
    for i,eid in enumerate(eid_list):   
        if i == 0:
            cmd = f"esel, 'S', 'ELEM', ,{eid}"
        else:
            cmd = f"esel, 'A', 'ELEM', ,{eid}"
        commands.append(cmd)
    cmd_string = "\n".join(commands)
    mapdl.input_strings(cmd_string)
    return


from utils.utils_geo_mesh.main_CAD import VesselModel
import utils.utils_geo_mesh.utils_CAD as utils_CAD

def extract_surfdata_from_vtu(CAD_instance: VesselModel, vtu_path: str, point_array_name: str, n: int, surface_taglist: list):
    """
    @update 0820 jeff
    - z_lipid: lipid length defined by the parameter(lipid_ratio)
    - vtu_path: we will analyze the surface data on the vtu file.
    - point_array_name: inside the vtu file there are various point_array data, among them we will analyze specific one.
         #point_array_name: EQV_strain, EQV_stress, Principal_strain, Principal_stress

    - n: we are going to divide the domain into n regions along the z axis. from - z_lipid to z_lipid.

    Now on each domain inside the analyze_vtu_path we need to extract Three data:
        - p95: 95% scalar value of the point data(strain or stress) 
        - T5%: Mean of Top 5% scalar value of the point data(strain or stress)
        - mean: Overall Mean of the point data(strain or stress)

    #Algorithm
    - Our goal is to derive the surface average point array data on each surface ( tag = 5 and 7)
    - For each tag, only consider triangles where all three nodes have the tag.

    For each region (1,2,3,...,n):
        - Compute area-weighted average and max surface elemental EQV (mean of nodal values per triangle).

    Returns:
        dict: For each tag (5, 7), a list:
            [1_avg, 2_avg, 3_avg, ..., n_avg]

    #Important
    - Nodal data is defined by the user. - point_array_name
    e.g.) point_array_name = "EQV_strain", "EQV_stress", "Principal_strain", "Principal_stress", "Displacement"
    """

    # 1) read file and extract surface.
    mesh = pv.read(vtu_path)

    #Extract surface
    surf = mesh.extract_surface()
    tris = surf.faces.reshape(-1, 4)[:, 1:]  # (n_tri, 3)
    pts = surf.points
    eqv = surf.point_data[point_array_name]

    tags = surf.point_data["Group"]
    
    #Final reulst dict.
    results = {}

    for tag in surface_taglist: # surface_taglist = [5,7] actually.
        
        # Only triangles where all three nodes have the tag
        mask = np.all(tags[tris] == tag, axis=1)
        tris_tag = tris[mask]

        # Prepare region stats
        region_stats = {}
        # Loop all triangels and allocate it on the region.
        for tri in tris_tag:

            # Check for zero values in nodal eqv data for this triangle
            tri_eqv = eqv[tri]
            if np.any(tri_eqv == 0):
                print(f"Warning: Zero value found in triangle {tri} with nodal values: {tri_eqv}")

            #tri = [n1, n2, n3] -> area, element eqv
            xyz = pts[tri]
            x_cent = xyz[:, 0].mean()
            y_cent = xyz[:, 1].mean()
            z_cent = xyz[:, 2].mean()
            
            def allocate_axial_region(z_cent):
                '''
                Divide the [0,1] into n domain -> index1 :[0, 1/n], index2 :[1/n, 2/n], ..., indexn :[(n-1)/n, 1] and find the index.
                return the index among 1 to n.(1 - most proximal to the lumen, n - most distal to the lumen)
                '''
                z_lipid_L = CAD_instance.z_lipid_L
                z_lipid_R = CAD_instance.z_lipid_R
                #Normalize the domain form [-z_lipid_L, z_lipid_R] to [0, 1]
                norm_z = (z_cent - z_lipid_L) / (z_lipid_R - z_lipid_L)

                #Divide the [0,1] into n domain -> index1 :[0, 1/n], index2 :[1/n, 2/n], ..., indexn :[(n-1)/n, 1] and find the index.
                for i in range(n):
                    if norm_z >= i / n and norm_z < (i+1) / n:
                        return i+1
                return None
            
            def allocate_circumferential_region(x_cent, y_cent, z_cent):
                '''
                Allocate the point center of the triangle into 3 regions.
                - Shoulder_left
                - Shoulder_right
                - Center

                #Algorithm  
                일단 먼저 엑스 값이 0보다 큰 녀석들만 분석한다.
                Circumferential dvidie the region into Shoulder and center (only two regions)
                루만 센터와 셀 중심점을 연결한 벡터와 와이 축이 이루는 각도를 구한다.
                그 각도가 특정 각도 이상이면 숄더_left, 이하이면 중심으로 간주한다.

                그리고 엑스값이 작은 녀석들 중에서
                같은 방식으로 특정각도가 이상이면 숄더_right, 이하이면 중심으로 간주한다.
                '''

                θ_crit = utils_CAD.alpha_theta(CAD_instance,z_cent) / 3 # lipid arc angle at z = z_cent
                y_lumen_center = utils_CAD.y_center_lumen(CAD_instance, z_cent)
                vector1 = np.array([x_cent, y_cent - y_lumen_center, 0]) # vector 1 = cell center point - lumen center point
                vector2 = np.array([0, 1, 0]) # y axis

                # Calculate angle between vectors using dot product
                dot_product = np.dot(vector1, vector2)
                v1_norm = np.linalg.norm(vector1)
                v2_norm = np.linalg.norm(vector2)
                
                # Avoid division by zero
                if v1_norm == 0 or v2_norm == 0:
                    angle = 0
                
                else:
                    cos_angle = dot_product / (v1_norm * v2_norm)
                    # Handle numerical errors that could make cos_angle slightly outside [-1,1]
                    cos_angle = np.clip(cos_angle, -1.0, 1.0)
                    angle = np.arccos(cos_angle)

                #now let us distribute the cell into three regions with the criterion
                if x_cent >= 0: # x > 0
                    if angle >= θ_crit:
                        region = "SR" # Shoulder_right
                    else:
                        region = "C" # Center
                else: # x < 0
                    if angle >= θ_crit:
                        region = "SL" # Shoulder_left
                    else:
                        region = "C" # Center
                
                return region
            
            axial_region = allocate_axial_region(z_cent)
            circumferential_region = allocate_circumferential_region(x_cent, y_cent, z_cent)
            region_key =  (axial_region, circumferential_region)

            #Pass the none case
            if axial_region is None: continue

            #Calculate area and eqv
            v1 = xyz[1] - xyz[0]
            v2 = xyz[2] - xyz[0]
            area = 0.5 * np.linalg.norm(np.cross(v1, v2))
            eqv_val = eqv[tri].mean()  # or use centroid interpolation if you prefer

            if region_key not in region_stats:
                region_stats[region_key] = {"area": [], "data": []}
            region_stats[region_key]["area"].append(area)
            region_stats[region_key]["data"].append(eqv_val)
        # DEGUGGING
        # Print the number of data points for each region_key for verification
        # for region_key, stats in region_stats.items():
        #     print(f"Region {region_key}: {len(stats['data'])} data points")
        '''
            Now the data is saved on the region_stats dict for instance as follows:
            e.g.) n = 3 case
            region_stats = {
                (1, "SR"): {"area": [area1, area2, ...], "data": [eqv1, eqv2, ...]},
                (1, "C"): {"area": [area1, area2, ...], "data": [eqv1, eqv2, ...]},
                (1, "SL"): {"area": [area1, area2, ...], "data": [eqv1, eqv2, ...]},
                (2, "SR"): {"area": [area1, area2, ...], "data": [eqv1, eqv2, ...]},
                (2, "C"): {"area": [area1, area2, ...], "data": [eqv1, eqv2, ...]},
                (2, "SL"): {"area": [area1, area2, ...], "data": [eqv1, eqv2, ...]},    
                (3, "SR"): {"area": [area1, area2, ...], "data": [eqv1, eqv2, ...]},
                (3, "C"): {"area": [area1, area2, ...], "data": [eqv1, eqv2, ...]},
                (3, "SL"): {"area": [area1, area2, ...], "data": [eqv1, eqv2, ...]},
            }

            from this we are going to calculate three diiffrent types of result.
            - p95: 95% scalar value of the point data
            - T5%: Mean of Top 5% scalar value of the point data
            - mean: Overall Area average Mean of each domain.

        ''' 
        #1st save cut off 95% on each domain - p95
        p95_list = {}
        for region in region_stats.keys():
            eqvs = np.array(region_stats[region]["data"])
            p95 = np.percentile(eqvs, 95)
            p95_list[region] = p95
        
        #2nd save Top 5% on each domain - T5%
        T5_list = {}
        for region in region_stats.keys():
            eqvs = np.asarray(region_stats[region]["data"])
            cut = np.percentile(eqvs, 95)
            T5 = eqvs[eqvs >= cut].mean()
            T5_list[region] = T5

        #3rd save mean on each domain
        mean_list = {}
        for region in region_stats.keys():
            areas = np.array(region_stats[region]["area"])
            eqvs = np.array(region_stats[region]["data"])
            mean = np.sum(eqvs * areas) / np.sum(areas)
            mean_list[region] = mean

        #4th save the result on the results dict
        results[tag] = {"p95": p95_list, "T5%": T5_list, "mean": mean_list}

    return results



#@updated 0801 jeff
def vtu_to_volume_eqv(z_lipid: float, vtu_path: str):
    """
    input: 
    - vtu_path to read 
    - z_lipid to classify the region.

    @update 0801 jeff
    Analyze volume tetrahedra in a Fibrous cap vtu file.
    
    #Algorithm
    For each region (pro1, pro2, mid1, mid2, dis1, dis2, by centroid z):
        - Compute volume and max surface elemental EQV (mean of nodal values per triangle).
    On top of that, find the node with the maximum nodal EQV for each tag and its region.

    Returns: single list
    [pro1_avg, pro2_avg, mid1_avg, mid2_avg, dis1_avg, dis2_avg]

    """

    mesh = pv.read_meshio(vtu_path)
    eqv = mesh.cell_data["EQV"]
    tetras = mesh.cells_dict[10]
    pts = mesh.points
   
    # Prepare region stats
    region_dict = {
        "pro1": {"volume": [], "eqv": []},
        "pro2": {"volume": [], "eqv": []},
        "mid1": {"volume": [], "eqv": []},
        "mid2": {"volume": [], "eqv": []},
        "dis1": {"volume": [], "eqv": []},
        "dis2": {"volume": [], "eqv": []},
    }

    #loop all tetra and calculate save the EQV data on each domain.
    for i, tetra in enumerate(tetras):
        #tetra = [n1, n2, n3, n4] -> volume, element eqv
        xyz = pts[tetra] # (4,3) each tetra node's point coordinates.
        z_cent = xyz[:, 2].mean() 

        #Determine the region along the z axis.
        if   z_cent > 0.75 * z_lipid:   region = None
        elif z_cent >  0.5 * z_lipid:   region = "dis2"
        elif z_cent > 0.25 * z_lipid:   region = "dis1"
        elif z_cent > 0:                region = "mid2"
        elif z_cent > -0.25 * z_lipid:  region = "mid1"
        elif z_cent >  -0.5 * z_lipid:  region = "pro2"
        elif z_cent > -0.75 * z_lipid:  region = "pro1"
        else: region = None

        #Pass the none case
        if region is None: continue

        #get the volume of the tetra
        v0, v1, v2, v3 = xyz[0], xyz[1], xyz[2], xyz[3]
        volume = abs(np.linalg.det([v1-v0, v2-v0, v3-v0])) / 6

        # Get stress value for this tetrahedron
        stress = eqv[i]

        region_dict[region]["volume"].append(volume)
        region_dict[region]["eqv"].append(stress)



    #Now calculate the average and max eqv for each region.
    avg_list = []
    for region in ("pro1", "pro2", "mid1", "mid2", "dis1", "dis2"):
        volume = np.array(region_dict[region]["volume"])
        eqvs = np.array(region_dict[region]["eqv"])
        if volume.size > 0:
            avg = round(np.sum(eqvs * volume) / np.sum(volume), 2)
        else:
            avg = 0.00
        avg_list.append(avg)



    return avg_list





# updated 0906 jeff
def generate_multiplied_vtu(
    amp_vtu_path: str,
    peak_vtu_path: str,
    amp_array_name: str,
    peak_array_name: str,
    output_vtu_path: str,
    new_array_name: str,
):
    """
    Create a new VTU where each node has a point array "R3" defined as:

        R3(node) = principal_strain(node) * principal_stress(node)

    Assumptions:
    - amp_vtu and peak_vtu share the same nodes in the same order.
    - `strain_array_name` exists in amp_vtu point data.
    - `stress_array_name` exists in peak_vtu point data.

    Parameters
    ----------
    amp_vtu_path : str
        Path to the VTU containing principal strain (point data).
    peak_vtu_path : str
        Path to the VTU containing principal stress (point data).
    strain_array_name : str
        Name of the strain point-data array in `amp_vtu_path`.
    stress_array_name : str
        Name of the stress point-data array in `peak_vtu_path`.
    output_vtu_path : str
        Destination path for the new VTU (copied from peak mesh) with R3 added.
    """
    amp_mesh = pv.read(amp_vtu_path)
    peak_mesh = pv.read(peak_vtu_path)

    if amp_array_name not in amp_mesh.point_data:
        raise KeyError(f"Point array '{amp_array_name}' not found in amp_vtu: {amp_vtu_path}")
    if peak_array_name not in peak_mesh.point_data:
        raise KeyError(f"Point array '{peak_array_name}' not found in peak_vtu: {peak_vtu_path}")

    if amp_mesh.n_points != peak_mesh.n_points:
        raise ValueError(
            f"Point count mismatch: amp_vtu has {amp_mesh.n_points}, peak_vtu has {peak_mesh.n_points}"
        )
    
    #Reshape the array to 1D for the multiplication.
    amp_vals = np.asarray(amp_mesh.point_data[amp_array_name]).reshape(-1)
    peak_vals = np.asarray(peak_mesh.point_data[peak_array_name]).reshape(-1)

    if amp_vals.shape[0] != peak_vals.shape[0]:
        raise ValueError(
            f"Array length mismatch: amp({amp_vals.shape[0]}) vs peak({peak_vals.shape[0]})"
        )

    muptliply_vals = amp_vals * peak_vals

    out_mesh = peak_mesh.copy(deep=True)
    out_mesh.point_data[new_array_name] = muptliply_vals
    out_mesh.save(output_vtu_path)
    print(f"[Completed] Wrote {new_array_name} VTU: {output_vtu_path}")
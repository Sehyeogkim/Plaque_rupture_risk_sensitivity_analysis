import numpy as np
import meshio   
import gmsh
import sys

'''
updated on @0630 jeff
-Calcifcation added.

updated on @0718 jeff
-Added save_log_mesh_quality function.
'''

def slash_lines():
    print("\n---------------------------------------\n")

def section_title(title):
    width = 50
    title_length = len(title)
    left_padding = (width - title_length) // 2
    right_padding = width - title_length - left_padding
    
    print("\n" + "-" * width)
    print("-" * width)
    print(f"{'-' * left_padding}{title}{'-' * right_padding}")
    print("-" * width)
    print("-" * width + "\n")

def gmsh_display(exit = False):
    
    wireframe = True
    gmsh.option.setNumber("Geometry.Surfaces", 1) # 0 hide, 1 show
    gmsh.option.setNumber("Geometry.Volumes", 1) # 0 hide, 1 show
    gmsh.option.setNumber("Geometry.SurfaceType", 1 if wireframe else 2) # 1 wireframe, 2 solid surface
    gmsh.model.occ.synchronize()
    
    if '-nopopup' not in sys.argv:
        gmsh.fltk.run()
    
    if exit:
        gmsh.finalize()
        sys.exit()
    return

def find_wall_surfaces(surface_tags, lumen_vol_tags):
    '''
    Find the wall surfaces' tag that are adjacent to the lumen volumes.
    '''
    wall = []
    for surf in surface_tags:
        vols = gmsh.model.getAdjacencies(2, surf)[0]
        if any(v in lumen_vol_tags for v in vols):
            wall.append(surf)
    
    return wall

def rearrange_cell_blocks(mesh):

    '''
    In order to generate mesh by Simmetrix, we need to specifiy the wall's face id as '1'.

    By Experience, when we put lumen.stl format as input to the 'Synthetic/mesh generator', 
    the wall's face id is automatically defined as the first cell block of the 'mesh.cells'
    (mesh.cells is a list of cell blocks obtained when reading a .msh file with meshio)

    Therefore, we need to rearrange the cell blocks to make (wall cell block) at 1st index.
    
    [Algorithm]
    1. Extract only triangle cell blocks (since there are vertex, line cell blocks in the mesh.cells)
    2. Find the largest triangle cell block.
    3. Rearrange the cell blocks to make the largest triangle cell block at the first index.

    Input : raw mesh (Generated on gmsh and read from Meshio)
    Output : rearranged mesh (wall cell block at first index + only triangle cell blocks are included)

    '''
    
    #Extract triangle cell blocks
    tri_blocks = [blk for blk in mesh.cells if blk.type == "triangle"] #3 triangle cell blocks
    counts = [blk.data.shape[0] for blk in tri_blocks]
    max_idx = int(np.argmax(counts)) #index of the largest block inside of the tri_blocks


    # Rearrange tri_blocks to put the largest block at front.
    largest_block = tri_blocks.pop(max_idx)  # Remove the largest block
    tri_blocks.insert(0, largest_block)      # Insert it at the beginning(1st index)


    #Defome face_id cell_data array
    face_id_arrays = []
    next_id = 2
    for i, blk in enumerate(tri_blocks):
        n = len(blk.data)
        if i == 0:
            face_id_arrays.append(np.ones(n, dtype=np.int32))
        else:
            face_id_arrays.append(np.full(n, next_id, dtype=np.int32))
            next_id += 1

    
    #Define fluid_mesh
    fluid_mesh = meshio.Mesh(
        points=mesh.points,
        cells=tri_blocks,
        cell_data={"FaceID": face_id_arrays}
    )


    print("fluid_mesh:",fluid_mesh)

    return fluid_mesh

def save_log_mesh_quality(log_file_path):
    '''
    Save the mesh quality information to a log file.
    Input: log_file_path
    '''
    # Open log file in append mode
    with open(log_file_path, "a") as f:
        f.write("\nMesh Statistics(GMSH):\n")
        f.write("-" * 40 + "\n")
        
        # Get total number of nodes
        node_tags, _, _ = gmsh.model.mesh.getNodes()
        f.write(f"Number of nodes: {len(node_tags)}\n")
        
        # Get all elements
        element_types, element_tags, node_tags = gmsh.model.mesh.getElements()
        
        # Map element type to name
        type_names = {
            1: "2-node Line",
            2: "3-node Triangle", 
            3: "4-node Quadrangle",
            4: "4-node Tetrahedron",
            5: "8-node Hexahedron",
            6: "6-node Prism",
            8: "3-node 2nd order line",
            9: "6-node 2nd order Triangle",
            11: "10-node 2nd order Tetrahedron",
            15: "1-node Point",
            16: "8-node 2nd order Quadrangle",
            17: "20-node 2nd order Hexahedron"
        }
        
        # Write element counts by type
        total_elements = 0
        for i, element_type in enumerate(element_types):
            num_elements = len(element_tags[i])
            total_elements += num_elements
            type_name = type_names.get(element_type, f"Type {element_type}")
            f.write(f"{type_name}: {num_elements}\n")
            
        f.write(f"Total number of elements: {total_elements}\n")
        
        # Get quality statistics (only for 2D and 3D elements)
        f.write("\nMesh Quality Statistics (2D and 3D elements only):\n")
        f.write("-" * 40 + "\n")
        
        all_qualities = []
        for i, element_type in enumerate(element_types):
            # Skip 1D elements (lines) and points
            if element_type in [1, 8, 15]:  # Skip lines and points
                continue
                
            if len(element_tags[i]) > 0:
                try:
                    qualities = gmsh.model.mesh.getElementQualities(element_tags[i])
                    all_qualities.extend(qualities)
                except Exception as e:
                    f.write(f"Warning: Could not compute quality for element type {type_names.get(element_type, element_type)}\n")
                    continue
        
        if all_qualities:
            qualities = np.array(all_qualities)
            f.write(f"Minimum quality: {np.min(qualities):.3f}\n")
            f.write(f"Maximum quality: {np.max(qualities):.3f}\n")
            f.write(f"Average quality: {np.mean(qualities):.3f}\n")
            f.write(f"RMS quality: {np.sqrt(np.mean(qualities**2)):.3f}\n")
            f.write(f"Number of elements with quality < 0.3: {np.sum(qualities < 0.3)}\n")
            f.write(f"Number of elements with quality < 0.5: {np.sum(qualities < 0.5)}\n")
            f.write(f"Number of elements with quality < 0.7: {np.sum(qualities < 0.7)}\n")
            f.write(f"Number of elements with quality < 0.9: {np.sum(qualities < 0.9)}\n")
            
            # Find worst elements
            if len(qualities) > 0:
                worst_indices = np.argsort(qualities)[:5]
                f.write("\nWorst Elements (top 5):\n")
                f.write("-" * 40 + "\n")
                for i, idx in enumerate(worst_indices):
                    f.write(f"Element {i+1}: Quality {qualities[idx]:.3f}\n")
        else:
            f.write("No elements found for quality computation\n")

def check_mesh_quality(quality_type = 0):
    '''
    Check mesh quality using Gmsh's built-in analysis tools.
    
    quality_type:
    0: scaled jacobian (default)
    1: aspect ratio
    2: minimum angle
    3: minimum angle
    '''
    
    # Set the quality type
    gmsh.option.setNumber("Mesh.QualityType", quality_type)
    
    print("\n\nMesh Statistics(GMSH):")
    print("-" * 40)
    
    # Get total number of nodes
    node_tags, _, _ = gmsh.model.mesh.getNodes()
    print(f"Number of nodes: {len(node_tags)}")
    
    # Get all elements
    element_types, element_tags, node_tags = gmsh.model.mesh.getElements()
    
    # Map element type to name
    type_names = {
        1: "2-node Line",
        2: "3-node Triangle",
        3: "4-node Quadrangle",
        4: "4-node Tetrahedron",
        5: "8-node Hexahedron",
        6: "6-node Prism",

        8: "3-node 2nd order line",
        9: "6-node 2nd order Triangle",

        11: "10-node 2nd order Tetrahedron",

        15: "1-node Point",
        16: "8-node 2nd order Quadrangle",
        17: "20-node 2nd order Hexahedron"
    }
    
    # Print element counts by type
    total_elements = 0
    for i, element_type in enumerate(element_types):
        num_elements = len(element_tags[i])
        total_elements += num_elements
        type_name = type_names.get(element_type, f"Type {element_type}")
        print(f"Number of {type_name}: {num_elements}")
    
    print(f"Total number of elements: {total_elements}")
    
    # Get quality statistics (only for 2D and 3D elements)
    print("\nMesh Quality Statistics (2D and 3D elements only):")
    print("-" * 40)
    
    all_qualities = []
    for i, element_type in enumerate(element_types):
        # Skip 1D elements (lines) and points
        if element_type in [1, 8, 15]:  # Skip lines and points
            continue
            
        if len(element_tags[i]) > 0:
            try:
                qualities = gmsh.model.mesh.getElementQualities(element_tags[i])
                all_qualities.extend(qualities)
            except Exception as e:
                print(f"Warning: Could not compute quality for element type {type_names.get(element_type, element_type)}")
                continue
    
    if all_qualities:
        qualities = np.array(all_qualities)
        print(f"Minimum quality: {np.min(qualities):.3f}")
        print(f"Maximum quality: {np.max(qualities):.3f}")
        print(f"Average quality: {np.mean(qualities):.3f}")
        print(f"RMS quality: {np.sqrt(np.mean(qualities**2)):.3f}")
        print(f"Number of elements with quality < 0.3: {np.sum(qualities < 0.3)}")
        print(f"Number of elements with quality < 0.5: {np.sum(qualities < 0.5)}")
        print(f"Number of elements with quality < 0.7: {np.sum(qualities < 0.7)}")
        print(f"Number of elements with quality < 0.9: {np.sum(qualities < 0.9)}")
        
        # Find worst elements
        if len(qualities) > 0:
            worst_indices = np.argsort(qualities)[:5]
            print("\nWorst Elements (top 5):")
            print("-" * 40)
            for i, idx in enumerate(worst_indices):
                print(f"Element {i+1}: Quality {qualities[idx]:.3f}")
    else:
        print("No elements found for quality computation")

def get_mesh_info(path):
    '''
    Print the type of the cells in the mesh.
    Input: .msh file path
    '''

    mesh = meshio.read(path)
    try:
        cell_dict = mesh.cells_dict
    except AttributeError:
        cell_dict = {block.type: block.data for block in mesh.cells}
    
    print("\nMesh Element Information(MESHIO):")
    print("-" * 40)
    print(f"{'Cell Type':<20} {'Element Count':>15}")
    print("-" * 40)
    for cell_type, data in cell_dict.items():
        print(f"{cell_type:<20} {len(data):>15}")
    print("-" * 40)


import math
def y_center_lumen(z, r_max, r_min, lesion_length):
    '''
    y coordinate of the center of the lumen along the z axis.
    
    #cos function is used to model the lumen.
    T = model.lesion_length
    amplitude = -(model.r_max - model.r_min)
    '''
    A = -(r_max - r_min)
    if abs(z) <= lesion_length/2:
        T = lesion_length
        return A / 2 + (A / 2) * math.cos(2 * math.pi * z / T)
    else:
        return 0
    

def radius_lumen(z, r_max, r_min, lesion_length):
    '''
    radius of the lumen along the z axis.
    '''
    if abs(z) <= lesion_length/2:
        A = r_min - r_max
        B = r_min
        r = A * math.cos(2 * math.pi * z / lesion_length) + B -  y_center_lumen(z, r_max, r_min, lesion_length)
    else:
        r = r_max
    return r

def alpha_theta(z, alpha, z_lipid):
    '''
    return, the θ(z) (******half value, radian)

    Boundary condition:
    - θ(z = 0) = model.alpha
    - θ(|z| = z_lipid ) = 0
    - dramatic drop near |z| = z_lipid

    
    By experiment, sqrt(cosx) would be the appropriate function.
    '''
    #1. sinusoidal function.

    alpha_rad = math.radians(alpha) #change it as radian.

    #SQRT COS FUNCTION
    T = 4 * z_lipid
    theat_SQRT = (alpha_rad / 2 ) * math.sqrt(math.cos(2 * math.pi * z/ T))

    #COS FUNCTION
    T = 4 * z_lipid
    theat_COS = (alpha_rad / 2 ) * math.cos(2 * math.pi * z/ T)

    #2. just linear function.
    a = alpha_rad / (2 * z_lipid)
    b = alpha_rad / 2
    y = - a * abs(z) + b

    #by experiment, sinusoidal function is most appropriate
    return theat_SQRT

def y_center_lesion(z, r_pos, r_ex, T, lesion_length):
    '''
    y coordinate of the center of the external vessle(circle),
    included the effect of PI.
    '''
    if abs(z) <= lesion_length:

        A = r_pos - r_ex
        return A/2 + (A/2) * math.cos(2 * math.pi * z/ T)
    else:
        return 0

def radius_lesion(z, r_pos, r_ex, T, lesion_length):
    '''
    radius of the external vessle(circle),
    included the effect of PI.
    '''
    if abs(z) <= lesion_length:
        A = r_pos - r_ex
        B = r_pos
        return (A) * math.cos( 2 * math.pi * z / T) + B - y_center_lesion(z, r_pos, r_ex, T, lesion_length)
    else:
        return r_ex

import random
def check_sphere_inside_volume(sphere_tag, volume_tag, r_ca):
    center = gmsh.model.occ.getCenterOfMass(3, sphere_tag)
    cx, cy, cz = center[0], center[1], center[2]

    num_points = 100
    test_points = []
    for _ in range(num_points):
        theta = random.uniform(0, 2 * math.pi)  # 경도 (0~2π)
        phi = random.uniform(0, math.pi)        # 위도 (0~π)

        x = cx + r_ca * math.sin(phi) * math.cos(theta)
        y = cy + r_ca * math.sin(phi) * math.sin(theta)
        z = cz + r_ca * math.cos(phi)

        test_points.append([x, y, z])

    #check if all of the points are inside the volume
    for point in test_points:
        if gmsh.model.isInside(3, volume_tag, point) <= 0:
            return False
    return True

def ensure_dim_tag_list(entities, dim=3):
    # entities가 [(3, 12), (3, 13)] 형태면 그대로 반환
    # entities가 [12, 13] 또는 np.array([12, 13])이면 (dim, tag)로 변환
    # entities가 12 또는 np.int32(12)면 [(dim, 12)]로 변환
    if entities is None:
        return []
    if isinstance(entities, (list, tuple, np.ndarray)):
        if len(entities) == 0:
            return []
        if isinstance(entities[0], (list, tuple)) and len(entities[0]) == 2:
            return list(entities)
        # [12, 13] 또는 np.array([12, 13])
        return [(dim, int(tag)) for tag in entities]
    # 단일 값
    return [(dim, int(entities))]









# @0708 jeff Following functions are for the Macro Calcification - Method1
def extract_centroids_certain_tag_meshio(mesh, index):
    result = []
    # mesh.cells: list of CellBlock, e.g. [CellBlock('tetra10', array(...)), ...]
    # mesh.cell_data['gmsh:physical']: list of arrays, one per cell block
    for block_idx, cell_block in enumerate(mesh.cells):
        if cell_block.type in ("tetra", "tetra10"):
            physical_tags = mesh.cell_data['gmsh:physical'][block_idx]
            target_indices = np.where(physical_tags == index)[0]
            if len(target_indices) == 0:
                continue
            # node indices for this cell block
            block_nodes = cell_block.data
            for i in target_indices:
                # For tetra10, use only the first 4 nodes (corners)
                if block_nodes.shape[1] >= 4:
                    centroid = np.mean(mesh.points[block_nodes[i, :4]], axis=0)
                else:
                    centroid = np.mean(mesh.points[block_nodes[i]], axis=0)
                result.append((block_idx, i, centroid))
    print(f"Total centroids calculated: {len(result)}")
    return result

def macro_calcification_meshio(mesh, index, ca_tag):
    result = extract_centroids_certain_tag_meshio(mesh, index)

    # z 필터
    lesion_length = 1.0
    r_max = 0.1
    r_min = 0.05
    z_lipid = 0.3 * lesion_length
    alpha = 220.0

    z_middle = -0.5 * z_lipid
    ca_length = 0.05
    z_min = z_middle - 0.5 * ca_length
    z_max = z_middle + 0.5 * ca_length

    # z 필터 적용
    for block_idx, elem_idx, centroid in result:
        if z_min <= centroid[2] <= z_max:
            mesh.cell_data['gmsh:physical'][block_idx][elem_idx] = ca_tag

    # theta 필터
    y_lumen = y_center_lumen(z_middle, r_max, r_min, lesion_length)
    lipid_theta = alpha_theta(z_middle, alpha, z_lipid)
    center = np.array([0, y_lumen, z_middle])
    y_axis = np.array([0, 1, 0])
    shoulder = True
    cal_arc = np.deg2rad(150.0)

    for block_idx, elem_idx, centroid in result:
        if mesh.cell_data['gmsh:physical'][block_idx][elem_idx] == ca_tag:
            v = centroid - center
            v[2] = 0
            norm = np.linalg.norm(v)
            if norm == 0:
                continue
            v_norm = v / norm
            y_norm = y_axis / np.linalg.norm(y_axis)
            cos_theta = np.dot(v_norm, y_norm)
            theta_val = np.arccos(np.clip(cos_theta, -1.0, 1.0))
            if shoulder:
                if centroid[0] < 0:
                    if theta_val > cal_arc/2 - lipid_theta/2:
                        mesh.cell_data['gmsh:physical'][block_idx][elem_idx] = 1
            else:
                if theta_val > cal_arc/2:
                    mesh.cell_data['gmsh:physical'][block_idx][elem_idx] = 1
    return mesh


# @0708 (Unstructured) jeff Following functions are for the Macro Calcification - Method1
def extract_centroids_certain_tag_unstructured(mesh, lipid_index):
    '''
    Input: pyvista mesh
    index: physical tag of the target cells(lipid core)

    Output: list of tuples, (cell_index,centroid)
    '''
    result = []

    # Get cell centers and cell data
    cell_centers = mesh.cell_centers().points
    physical_tags = mesh.cell_data['gmsh:physical']  # For unstructured grid

    # Find cells with matching physical tag
    target_indices = np.where(physical_tags == lipid_index)[0]
    
    # Get centroids for matching cells
    for i in target_indices:
        centroid = cell_centers[i]
        result.append((i, centroid))  

    return result

def macro_calcification_centroid_unstructured(model, mesh, index, ca_tag):
    '''
    There are 4 calcifcation parameters.s   
    1. z_middle
    2. ca_length
    3. shoulder or center
    4. cal_arc
    5. Distance from the center of the lumen.

    index: lipid tag number.

    save fraction as a depent variable.
    '''

    # Calculate initial count of lipid cells (tag = index)
    initial_lipid_count = np.sum(mesh.cell_data['gmsh:physical'] == index)
    
    result = extract_centroids_certain_tag_unstructured(mesh, index) #contain, cell_idx, centroid
    
    ############################################################

    #Rule1: z filter
    z_min = model.z_middle - 0.5 * model.ca_length
    z_max = model.z_middle + 0.5 * model.ca_length

    for cell_idx, centroid in result:
        if z_min <= centroid[2] <= z_max:
            
            mesh.cell_data['gmsh:physical'][cell_idx] = ca_tag

            z = centroid[2] # z coordinate of the element's centroid.
            y_lumen = y_center_lumen(z, model.r_max, model.r_min, model.lesion_length)
            lipid_theta = alpha_theta(z, model.alpha, model.z_lipid)
            center = np.array([0, y_lumen, z])
            y_axis = np.array([0, 1, 0])
            
            v = centroid - center
            v[2] = 0
            norm = np.linalg.norm(v)
            
            if norm == 0:
                continue
            
            v_norm = v / norm
            y_norm = y_axis / np.linalg.norm(y_axis)
            cross_z = np.cross(y_norm, v_norm)[2]  # sin(theta)
            dot = np.dot(y_norm, v_norm) #cos(theta)
            theta_val = np.arctan2(cross_z, dot) #radian angle with the y axis.

            if model.shoulder == 1.0:
                if theta_val > (model.cal_arc - lipid_theta/2):
                    mesh.cell_data['gmsh:physical'][cell_idx] = index
            else:
                if abs(theta_val) > model.cal_arc/2:
                    mesh.cell_data['gmsh:physical'][cell_idx] = index

            #rule3: distance from the center of the lumen. if the distance is less than d, then the cell is lipid core.
            radius = radius_lumen(z, model.r_max, model.r_min, model.lesion_length)
            distance = np.linalg.norm(centroid - center)
            
            if distance - radius - model.fc_av < model.d:
                mesh.cell_data['gmsh:physical'][cell_idx] = index

    # Calculate final count of calcification cells (tag = ca_tag)
    final_ca_count = np.sum(mesh.cell_data['gmsh:physical'] == ca_tag)
    
    # Calculate fraction
    fraction = final_ca_count / initial_lipid_count if initial_lipid_count > 0 else 0.0
    
    # print(f"Initial lipid cells (tag {index}): {initial_lipid_count}")
    # print(f"Final calcification cells (tag {ca_tag}): {final_ca_count}")
    print(f"Calcification Fraction: {fraction:.4f}")

    return mesh, fraction

def two_sides_physical_tag(solid_dim_tag, physical_tag_num = 6, tag_name = "Two_sides"):
    
    all_solid_surfaces = gmsh.model.getBoundary([solid_dim_tag], oriented=False)

    # Get z coordinates of surface centers to identify inlet/outlet
    surface_centers = []
    for surface in all_solid_surfaces:
        com = gmsh.model.occ.getCenterOfMass(surface[0], abs(surface[1]))
        surface_centers.append((surface, com[2]))  # Store surface and its z coordinate

    # Sort by z coordinate
    surface_centers.sort(key=lambda x: x[1])
    
    # First surface (minimum z) is inlet (side1), last surface (maximum z) is outlet (side2)
    side1_surface = surface_centers[0][0] # (dim = 2 , tag)
    side2_surface = surface_centers[-1][0] # (dim = 2 , tag)

    # Add physical groups for inlet and outlet
    gmsh.model.addPhysicalGroup(2, [abs(side1_surface[1]), abs(side2_surface[1])], 
                                    tag = physical_tag_num, name=tag_name)
    gmsh.model.occ.synchronize()

    return



def check_mesh_size(vtu_mesh_path):
    '''
    Check the size of the mesh.
    Input: vtu file path
    Output: # of tetra10 cells.
    '''
    import meshio
    try:
        mesh = meshio.read(vtu_mesh_path)
    except:
        print(f"Vtu file not found or error: {vtu_mesh_path}")
        return None
    return len(mesh.cells_dict['tetra10'])


def check_lipid(lipid_stp_path):
    
    '''
    Check whether the lipid is void or twisted.
    return bool: True if the lipid is void or twisted.
    '''

    gmsh.initialize()
    gmsh.model.occ.importShapes(lipid_stp_path)
    gmsh.model.occ.synchronize()
    # Get number of 3D cells (volumes)
    vol_num = len(gmsh.model.getEntities(3))
    surf_num = len(gmsh.model.getEntities(2))
    if vol_num == 0:
        gmsh.finalize()
        print(f"Void lipid: {lipid_stp_path}")
        return True

    if surf_num >= 3:
        gmsh.finalize()
        print(f"Twisted lipid: {lipid_stp_path}")
        return True

    gmsh.finalize()
    return False
import numpy as np
import pyvista as pv
import os
from collections import defaultdict

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

def surface_nid_dic_bf(mesh, Linear = True):
    '''
    @update 05/20 jeff
    This function is defined in order to extract the boundary node IDs from the unstructred mesh 
    (for the future bc settings, for instance, pressure load or traction load).
    
    While Transforming the unstructrued quadratic mesh(.msh) to ansys.cdb(.cdb),
    Quadratic surface meshes(triangles) are removed and only the volume elements survive.
    (#importnat: only when the surface mesh is 'qudartic triangle', since still ansys doesn't support them)

    Therefore, we need to extract the boundary node IDs, which were defined as physical tags on gmsh,

    
    Input:
    - mesh instance: pyvista.UnstructuredGrid
    - Linear: bool (False, if the mesh is quadratic)

    Output: two dictonaries
    - Nid_dictonary 
        {Physical group tag defined on gmsh: List of node ids}
    '''


    # print(mesh)
    # print(mesh.cell_data["gmsh:physical"]) #[1,1,3,5,1,1,...7,6,5]
    # print(mesh.cells_dict)
    # print(mesh.celltypes) #[22,24,22,22,......]


    # VTK_QUADRATIC_TRIANGLE(6 Nodes) = 22 (surface), VTK_Linear_TRIANGLE(3 Nodes) = 5 
    # VTK_QUADRATIC_TETRA(10 Nodes) = 24 (volume), VTK_Linear_TETRA(4 Nodes) = 10

    type = 5 if Linear else 22
    
    # 1) Boolean mask and cell data preparation
    mask = (mesh.celltypes == type)
    tags = mesh.cell_data["gmsh:physical"][mask]
    cells = mesh.cells_dict[type]


    # 2) dict initialization
    nodes_by_tag = defaultdict(set)    # tag → {node1, node2, ...}
    conns_by_tag = defaultdict(list)   # tag → [ [n1,n2,n3], [n4,n5,n6], ... ]

    # 3) accumulate data by tag
    for tag, conn in zip(tags, cells):
        # Increment all node indices by 1
        conn_plus1 = [n + 1 for n in conn]
        nodes_by_tag[tag].update(conn_plus1)
        conns_by_tag[tag].append(conn_plus1)

    # 4) (optional) convert set to list
    nodes_by_tag = {tag: list(nodes) for tag, nodes in nodes_by_tag.items()}

    # check results.
    print("Nodes by tag:")
    for tag, nds in nodes_by_tag.items():
        print(f"  Tag {tag}: {len(nds)} nodes")

    print("\nConnectivity by tag:")
    for tag, conns in conns_by_tag.items():
        print(f"  Tag {tag}: {len(conns)} elements connectivities")

    return nodes_by_tag, conns_by_tag

def surface_nid_dic(mesh, Linear=True, Terminal_display = False):
    """
    Fast extraction of boundary node IDs and element connectivities from a pyvista mesh, grouped by physical tag.
    Returns:
        nodes_by_tag: dict {tag: [node ids]}
        conns_by_tag: dict {tag: [[conn1], [conn2], ...]}
    """
    type_id = 5 if Linear else 22 # linear or quadratic triangle.
    
    mask = (mesh.celltypes == type_id)
    tags = mesh.cell_data["gmsh:physical"][mask]
    cells = mesh.cells_dict[type_id]

    tags = np.asarray(tags)
    cells = np.asarray(cells)

    cells_plus1 = cells + 1

    nodes_by_tag = defaultdict(set)
    conns_by_tag = defaultdict(list)

    for tag in np.unique(tags):
        tag_mask = (tags == tag)
        # Flatten all node indices for this tag
        node_ids = cells_plus1[tag_mask].ravel()
        nodes_by_tag[tag].update(node_ids)
        # Add connectivities for this tag
        tag_conns = cells_plus1[tag_mask]
        for conn in tag_conns:
            conns_by_tag[tag].append(list(conn))

    nodes_by_tag = {tag: list(nodes) for tag, nodes in nodes_by_tag.items()}

    if Terminal_display:
        print("Nodes by tag:")
        for tag, nds in nodes_by_tag.items():
            print(f"  Tag {tag}: {len(nds)} nodes")

        print("\nConnectivity by tag:")
        for tag, conns in conns_by_tag.items():
            print(f"  Tag {tag}: {len(conns)} elements connectivities")

    return nodes_by_tag, conns_by_tag

def define_mat(mapdl, mat_id, rho, E, v):
    '''
    @update 05/21 jeff
    '''
    mapdl.mp('DENS', mat_id, rho)  # Density in g/cm³
    mapdl.mp('EX', mat_id, E)  # Elastic modulus in dynes/cm² (e.g., 1 MPa = 10^6 dynes/cm²)
    mapdl.mp('NUXY', mat_id, v)  # Poisson's ratio
    
    return

def nid_list_sel(mapdl, nid_list):
    '''
    @update 05/21 jeff
    This function is defined in order to select the node IDs in the nid_list.
    '''
    commands = []
    for i,nid in enumerate(nid_list):
        if i == 0:
            cmd = f"nsel, 'S', 'NODE', ,{nid}"
        else:
            cmd = f"nsel, 'A', 'NODE', ,{nid}"
        commands.append(cmd)
    cmd_string = "\n".join(commands)
    mapdl.input_strings(cmd_string)
    return

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

def Create_cm_from_nid_list(mapdl, node_id_list, cm_name, Display = True):
    '''
    @update 05/20 jeff

    This function is defined in order to generate 'a node component'
    from the node_id list.

    #input:
    - node_id list  e.g.) [nid1, nid2, nid3, ...]
    - cm_name to be defined. e.g.) "side1"

    #output:
    -> Generate a node component has 'cm_name' with the node_id list.
    

    Procedures:
    1. Select each of the node_id in the list by using nsel 
    2. Stack the commands by using nsel, 'A', 'NODE', ,{nid}
    3. Execute the commands by using input_strings
    4. Plot the component by using nplot if Display is True
    5. Create the component by using cm
    '''

    commands = []
    for i,nid in enumerate(node_id_list):
        if i == 0:
            cmd = f"nsel, 'S', 'NODE', ,{nid}"
        else:
            cmd = f"nsel, 'A', 'NODE', ,{nid}"
        commands.append(cmd)
    cmd_string = "\n".join(commands)
    mapdl.input_strings(cmd_string)

    mapdl.nplot(background='w', plot_bc = True, title = cm_name) if Display else None
    mapdl.cm(cm_name, "NODE")
    mapdl.allsel()
    return

def fix_bc_from_nid_list(mapdl, nid_list):
    '''
    @update 05/21 jeff

    This function allocate fixed boundary condition 
    on all of the node IDs in the nid_list.

    '''
    commands = []
    for nid in nid_list:
        cmd = f"d, {nid}, all"
        commands.append(cmd)
    cmd_string = "\n".join(commands)
    mapdl.input_strings(cmd_string)
    return

def check_selected_mat_etype(mapdl, name):
    '''
    function to check the material and element type id of the selected region.

    #Caution
    "Selected region" by esel
    '''
    max_element_number = mapdl.get(entity="ELEM", entnum=0, item1="NUM", it1num="MAX") #print(max_element_number)

    mat_id = mapdl.get(entity="ELEM", entnum = max_element_number, item1 = "ATTR", it1num = "MAT")
    type_num = mapdl.get(entity="ELEM", entnum = max_element_number, item1 = "ATTR", it1num = "TYPE")
    print(f"Material ID of {name} :", mat_id )
    print(f"Element type of {name} :", type_num)
    return

def mesh_eplot(mapdl, title):
    '''
    function to plot the mesh.
    '''
    mapdl.eplot(
        style = "wireframe",
        background='w',
        show_edges=True,
        smooth_shading=True,
        title = title
        )

    return

def wipe_out_useless_data(ansys_exec_dir):
    '''
    Delete all files in ansys_exec_dir except:
    .csv, .stp, .msh, _0.err, .vtp, .vtu, .stl, .json, .vtk
    (.rst and .db are removed)
    '''
    keep = ('.csv', '.stp', '.msh', '_0.err', '.vtp', '.vtu', '.stl', '.json', '.vtk')
    for file in os.listdir(ansys_exec_dir):
        fpath = os.path.join(ansys_exec_dir, file)
        if not os.path.isfile(fpath):
            continue
        if not file.endswith(keep):
            os.remove(fpath)
    return


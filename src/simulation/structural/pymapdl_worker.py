# -*- coding: utf-8 -*-
from tkinter.constants import FALSE
import ansys.mapdl.core as pymapdl
import os
import numpy as np
import pyvista as pv
import time
from ansys.mapdl.reader import save_as_archive
import meshio
from pathlib import Path
import pandas as pd
from utils.utils_geo_mesh.main_CAD import CAD_instance_from_idx
import utils.utils_ansys.utils_prep as utils_prep
import utils.utils_ansys.utils_bc as utils_bc


'''
Goal of this  ver2 is change the post processing part.
1. save the time
2. save the data size.

from .rst -> .vtu it took too long time.
also currently we are saving two diff .vtu file peak and low but
I would like to save on .vtu file and point array as peak_~~~~ and low_~~~~
(I remeber currnelty saveing various parametrs such as EQV_strain, Principal_strain, etc.)

so I think we better test fisrt with a simple simulation cases and then implement to my real cases.
cuz other wise simulation time will be too long.

test inside the ./solid_data/case_0_test plz.
(just a single case)
'''

class PYMAPDL_worker():

    def __init__(self, case_index: int, wall_folder_path: str, ansys_exec_dir: str, 
                parameter_csv_path: str, port_num: int = 50052, nproc: int = 10):
        
        self.case_index = case_index # case id, should be int.
        self.wall_folder_path = wall_folder_path
        self.ansys_exec_dir = ansys_exec_dir
        self.parameter_csv_path = parameter_csv_path
        self.port_num = port_num
        self.nproc = nproc

        #will be defined further.
        self.bc_type = ""
        self.casename = ""
        self.cdb_path = ""

        #Create a CAD instance from the parameter csv file and the case index.
        self.vessel_model =  CAD_instance_from_idx(self.case_index, self.parameter_csv_path)
        
    def pymapdl_launch(self, display_info = True):
    
        #Launch MAPDL
        self.mapdl = pymapdl.launch_mapdl(
                jobname = f"{self.case_index}",
                exec_file="/opt/cvbml/softwares/ansys_inc/v251/ansys/bin/ansys251",
                run_location = self.ansys_exec_dir,
                loglevel="ERROR", #Print out only ERROR
                override = True,
                port = None,
                nproc = self.nproc
            )
        print(f"New MAPDL launched successfully on port {self.mapdl.port}")
        self.mapdl.units("CGS")
        
        #MAPDL INFO
        if display_info:
            utils_prep.section_title(f"ANSYS MAPDL VER: {self.mapdl.version}")
            print(self.mapdl)
            print("current direcotry:", self.mapdl.directory)
            print(f"Number of CPU: {self.mapdl.get_value('ACTIVE', 0, 'NUMCPU')}")
            print("mapdl ip:", self.mapdl.ip)
            print("mapdl port:", self.mapdl.port)
            utils_prep.slash_lines()
        
        return None
    
    def pymapdl_msh_to_cdb(self, gmsh_path: Path, type: int = 1):
        '''
            Convert .msh file to .cdb file.

            @jeff 0708
            if self.macro_cal is True, the macro calcification will be applied to the mesh.
        '''


        print("======= .msh to .cdb ========")
        if not gmsh_path.exists() or gmsh_path.stat().st_size == 0:
            raise ValueError(f"Invalid mesh file: {gmsh_path} is missing or empty")

        try:
            gmsh_mesh = meshio.read(gmsh_path, file_format="gmsh")
        except Exception as e:
            raise ValueError(f"Failed to read gmsh mesh {gmsh_path}: {e}") from e

        mesh = pv.from_meshio(gmsh_mesh)
        tags = np.array(mesh.cell_data["gmsh:physical"])
        print("gmsh:physical tags:", set(tags))
        
        
        # only for the type2 mesh. for the type1 mesh. [1,2,3,4,5,6,7]
        # for the type2 mesh. [1,2,3,4,5,6,7,8]
        tag_max = 7 if type == 1 else 8
    
        for i in range(1, tag_max + 1): # 1~8 should be all in the tags
            if i not in tags:
                print(f"gmsh:physical tag {i} is not found in the mesh.")
                raise Exception(f"gmsh:physical tag {i} is not found in the mesh.")
        print("gmsh:physical tags are all found in the mesh.")

        mesh.points *= 0.1 #scale down cuz of inventor.
        self.nodes_by_tag, _ = utils_prep.surface_nid_dic(mesh, Linear = False, Terminal_display = False)
        
        cdb_path = self.ansys_exec_dir / f"total_solid{type}.cdb"
        if cdb_path.exists():
            print(f"CDB file already exists: {cdb_path}. pass")
            return str(cdb_path)
        
        save_as_archive(cdb_path, mesh)
        return cdb_path

    def safe_mapdl_exit(self):
        if not hasattr(self, "mapdl"):
            return

        try:
            self.mapdl.clear()
        except Exception as e:
            print(f"MAPDL clear failed: {e}")

        try:
            self.mapdl.exit()
        except Exception as e:
            print(f"MAPDL exit failed: {e}")

    def pymapdl_prep_bc(self, wall_csv_path: Path):

        #start the prep7 read the .cdb file.
        self.mapdl.prep7()
        from pathlib import Path
        cdb_path_obj = Path(self.cdb_path)
        cdb_name = str(cdb_path_obj.name)
        self.mapdl.cdread("db", cdb_name)
        self.mapdl.nlgeom("OFF")
        self.mapdl.shpp("OFF")

        #check geo and mesh info.
        utils_prep.section_title("Geo and Mesh info")
        print(self.mapdl.mesh)
        self.mapdl.allsel()

        print("set(self.mapdl.mesh.material_type):", set(self.mapdl.mesh.material_type))
        print("set(self.mapdl.mesh.etype):", set(self.mapdl.mesh.etype))

        wall_in_vessel_tag = 4
        wall_in_fc_tag = 5
        sides_tag = 6
        lipid_in_fc_tag = 7 # new tag updated 0528 jeff


        #Fixed support BC on the two sides which were tagged as 6. (0.15s)
        sides_tag = 6
        fix_time = time.time()
        utils_prep.fix_bc_from_nid_list(self.mapdl, self.nodes_by_tag[sides_tag])
        print(f"Fixed support BC time: {time.time() - fix_time:.2f} seconds")


        #generate Component block for each surface.
        cm_display = False
        cm_time = time.time()
        utils_prep.Create_cm_from_nid_list(self.mapdl, self.nodes_by_tag[wall_in_vessel_tag], "WALL_IN_VESSEL", cm_display)
        utils_prep.Create_cm_from_nid_list(self.mapdl, self.nodes_by_tag[wall_in_fc_tag], "WALL_IN_FC", cm_display)
        utils_prep.Create_cm_from_nid_list(self.mapdl, self.nodes_by_tag[lipid_in_fc_tag], "LIPID_IN_FC", cm_display) # updated 0528 jeff
        print(f"Create CM for wall_in_vessel, wall_in_fc, and lipid_in_fc time: {time.time() - cm_time:.2f} seconds")
        
        #Define Element type 7 for Total Traction.
        self.mapdl.et(7,"SURF154")
        self.mapdl.keyopt(7,2,1) # face,1,2,3 -> local x,y,z, direction
        self.mapdl.keyopt(7,4,0) # 8 nodes.
        self.mapdl.keyopt(7,7,1)
        self.mapdl.keyopt(7,11,2)

        #Define a new local coordinate(num 14, arbitrary) for etype 7(element local coordinate)
        self.mapdl.local(14, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        self.mapdl.csys(0) 

        #Generate ETYPE 7 surface Mesh in order to apply Total traction.
        walls = ["WALL_IN_FC", "WALL_IN_VESSEL"]
        for wall in walls:
            self.mapdl.cmsel(type_= 'S', name = wall, entity = 'NODE')
            self.mapdl.type(7)
            self.mapdl.esurf()

        self.mapdl.esel('S', 'TYPE', '', 7)
        self.mapdl.emodif('all', 'ESYS', 14)
        wall_elem_array = self.mapdl.mesh.elem

        self.mapdl.nsle() #select all nodes attached to the currently selected element.
        wall_node_ids = self.mapdl.mesh.nnum #wall node ids.
        wall_node_coord = self.mapdl.mesh.nodes #wall node point coordinates.
        self.mapdl.allsel()

        #Apply Total traction to the wall.
        '''
        1. Interpolate the wall data.
        2. Apply the traction to the wall.
        '''
        print(f"\nApplying {self.bc_type} BC...")
        print("wall_csv_path: ", wall_csv_path)
        wall_interpol_df = utils_bc.Interpolate_wall_data(wall_csv_path, wall_node_ids, wall_node_coord)

        apply_time = time.time()
        txt_path = os.path.join(self.wall_folder_path, f"apply_traction_{self.bc_type}.txt")
        utils_bc.Apply_Traction(self.mapdl, wall_interpol_df, wall_elem_array, txt_path = txt_path)
        print(f"Apply traction {self.bc_type} time: {time.time() - apply_time:.2f} seconds")
        self.mapdl.finish() #finish the prep7

        #Save the mesh and BC info.
        self.mapdl.allsel()
        self.mapdl.cdwrite('All',f'{self.casename}','cdb')

        return
    
    def pymapdl_prep_mat(self, material_json_path: str):
    
        '''
        Step6. Define Material Properties
        - read the material.json file and allocate the material properties.
        - E must be got from the parameter csv file.
        '''
        self.mapdl.prep7() #Must restart the prep7 after defining new jobname.
        self.mapdl.nlgeom("OFF")
        self.mapdl.shpp("OFF")

        import json
        with open(material_json_path, 'r') as f:
            material_dict = json.load(f)

        for name, mat_dict in material_dict.items():
            mat_id = mat_dict["mat_id"]
            #E = mat_dict["E"] Note we are not using this anymore. 20260401 jeff

            parameter_df = pd.read_csv(self.parameter_csv_path)
            E = parameter_df.loc[self.case_index, f"E_{name}"]
            
            #debugging: print the E value.
            print(f"E_{name}: {E}")

            rho = mat_dict["rho"]
            v = mat_dict["v"]
            self.mapdl.mp('DENS', mat_id, rho)  # Density in g/cm³
            self.mapdl.mp('EX', mat_id, E)  # Elastic modulus in dynes/cm² (e.g., 1 MPa = 10^6 dynes/cm²)
            self.mapdl.mp('NUXY', mat_id, v)  # Poisson's ratio
        
        #Check the material properties.
        self.mapdl.allsel()
        utils_prep.section_title(f"Material ID of {self.casename}")
        print(self.mapdl.mplist())
        self.mapdl.finish() #finish the prep
        return
    
    def pymapdl_solve(self):
        '''
        static solver.
        '''
        start_solver_time = time.time()
        utils_prep.section_title(f"Solve {self.casename}") #self.casename = case_{index}_{bc_type}
        
        self.mapdl.allsel() 
        self.mapdl.run("/SOLU", verbose = True) #start the solver
        self.mapdl.antype("STATIC")
        self.mapdl.nlgeom("OFF") #turm off the non-linear
        
        # Set out-of-core mode for sparse solver to reduce memory usage
        # This is important because if PCG fails, ANSYS will automatically switch to sparse solver
        # Required memory: ~84GB (out-of-core) vs ~285GB (in-core)        
        # PCG solver: tries to converge with less memory
        # If PCG fails, ANSYS automatically switches to sparse direct solver (LU decomposition)
        # The sparse solver will then use out-of-core mode (set above) to reduce memory usage
        # Relaxed tolerance (1e-6 instead of 1e-8) to improve convergence chance
        self.mapdl.eqslv(lab = 'PCG', toler = 1e-5)
        
        self.mapdl.time(1) # set the end_time
        self.mapdl.autots("ON") #auto time stepping.
        self.mapdl.nsubst(nsbstp = 1, nsdbmn = 1, nsbmx = 10, carry = "OFF")
        self.mapdl.kbc(0) #ramped Load
        self.mapdl.allsel()

        self.mapdl.solve(verbose = True)
        self.mapdl.finish()

        print(f"Total simulation time: {time.time() - start_solver_time:.2f} seconds")
        
        try: 
            self.mapdl.db.save(f'{self.casename}.db')
        except Exception as e:
            try:
                self.mapdl.db.save(f'{self.casename}.db')
            except Exception as e:
                print(f"Failed to save .db file: {e}")
                raise Exception(f"Failed to save .db file: {e}")

        #clean up the useless data.
        #utils_prep.wipe_out_useless_data(self.ansys_exec_dir)

    #Post processing functions.
    def _extract_fc_results(self):
        '''
        Extract FC results from the live MAPDL session and return as a dict of numpy arrays.
        Does NOT build mesh or write .vtu — that is done by _build_fc_vtu.
        '''
        start_time = time.time()
        self.mapdl.post1()
        self.mapdl.set(1,1)

        #Extract the FC element data.
        self.mapdl.esel('S', 'MAT', '', 3) # Select MAT ID 3, which is FC.
        elem_fc = np.array(self.mapdl.mesh.elem)
        elem_eqv_fc = np.array(self.mapdl.post_processing.element_stress('EQV'))
        elem_princ_fc = np.array(self.mapdl.post_processing.element_stress('1'))

        #Extract the FC node data.
        self.mapdl.nsle()
        nids_fc = np.array(self.mapdl.mesh.nnum)
        ncoords_fc = np.array(self.mapdl.mesh.nodes)
        nodal_disp_fc = np.array(self.mapdl.post_processing.nodal_displacement('ALL'))
        nodal_eqv_strain_fc = np.array(self.mapdl.post_processing.nodal_total_eqv_strain())
        nodal_princ_strain_fc = np.array(self.mapdl.post_processing.nodal_total_principal_strain('1'))
        nodal_eqv_stress_fc = np.array(self.mapdl.post_processing.nodal_eqv_stress())
        nodal_princ_stress_fc = np.array(self.mapdl.post_processing.nodal_principal_stress('1'))

        #node ids of two surfaces of FC (WALL_IN_FC, LIPID_IN_FC)
        self.mapdl.cmsel(type_= 'S', name = "WALL_IN_FC", entity = 'NODE')
        nids_5 = np.array(self.mapdl.mesh.nnum) #physical tag 5
        self.mapdl.cmsel(type_= 'S', name = "LIPID_IN_FC", entity = 'NODE')
        nids_7 = np.array(self.mapdl.mesh.nnum) #physical tag 7
        self.mapdl.finish()

        print(f"Extracted FC results for {self.bc_type} in {time.time() - start_time:.2f} seconds")

        return {
            "elem_fc": elem_fc, "nids_fc": nids_fc, "ncoords_fc": ncoords_fc,
            "elem_eqv_stress": elem_eqv_fc, "elem_princ_stress": elem_princ_fc,
            "nodal_disp": nodal_disp_fc, "nodal_eqv_strain": nodal_eqv_strain_fc,
            "nodal_princ_strain": nodal_princ_strain_fc,
            "nodal_eqv_stress": nodal_eqv_stress_fc, "nodal_princ_stress": nodal_princ_stress_fc,
            "nids_wall_in_fc": nids_5, "nids_lipid_in_fc": nids_7,
        }

    def _build_fc_vtu(self, data: dict, output_path: str):
        '''
        Build a single FC-only .vtu file for one BC type.
        '''
        start_time = time.time()

        nids_fc = data["nids_fc"]
        ncoords_fc = data["ncoords_fc"]
        elem_fc = data["elem_fc"]

        index_map = np.empty(nids_fc.max()+1, dtype=int)
        index_map[nids_fc] = np.arange(len(nids_fc))
        corners = np.stack([elem_fc[:, -10], elem_fc[:, -9], elem_fc[:, -8], elem_fc[:, -7]], axis=1)
        cell_nodes = index_map[corners]

        tags = np.zeros(len(nids_fc), dtype=int)
        for nid in data["nids_wall_in_fc"]:
            tags[index_map[nid]] = 5
        for nid in data["nids_lipid_in_fc"]:
            tags[index_map[nid]] = 7

        point_data = {
            "Group": tags,
            "Displacement": data["nodal_disp"][index_map[nids_fc]],
            "EQV_strain": data["nodal_eqv_strain"][index_map[nids_fc]],
            "Principal_strain": data["nodal_princ_strain"][index_map[nids_fc]],
            "EQV_stress": data["nodal_eqv_stress"][index_map[nids_fc]],
            "Principal_stress": data["nodal_princ_stress"][index_map[nids_fc]],
        }
        cell_data = {
            "EQV_stress": [data["elem_eqv_stress"]],
            "Principal_stress": [data["elem_princ_stress"]],
        }

        mesh = meshio.Mesh(
            points=ncoords_fc,
            cells=[("tetra", cell_nodes)],
            cell_data=cell_data,
            point_data=point_data,
        )

        meshio.write(output_path, mesh, file_format="vtu")
        print(f"Saved FC VTU: {output_path} in {time.time() - start_time:.2f} seconds")



if __name__ == "__main__":

    import pathlib
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("start", type=int, help="Start case index (inclusive)")
    parser.add_argument("end", type=int, help="End case index (exclusive)")
    parser.add_argument("nproc", type=int, help="Number of processors")
    parser.add_argument("--test", action="store_true", help="Test mode: process only case_0_test with case_index=0")
    args = parser.parse_args()

    working_dir = pathlib.Path(__file__).parent
    pre_data_dir = working_dir / "pre_data"
    post_data_dir = working_dir / "post_data"
    material_json_path = pre_data_dir / "material.json"
    parameter_csv_path = pre_data_dir / "parameterB_new.csv" #revised 20260401 jeff 
    wall_dir = post_data_dir / "wall_expB_new" #revised 20260401 jeff 
    solid_dir = working_dir / "solid_data"
    
    no_msh = []
    too_large_mesh = []


    case_list = list(range(args.start, args.end))
    case_dir_map = {i: solid_dir / f"case_{i}" for i in case_list}

    for case_index in case_list:

        case_dir = case_dir_map[case_index]
        ansys_dir = case_dir / "ansys_results_type2_expB_new"
        ansys_dir.mkdir(parents=True, exist_ok=True)
        ansys_exec_dir = ansys_dir


        #Quick check if wall csv path exists or not

        wall_peak_csv_path = wall_dir / "peak" / f"wall_peak_case_{case_index}.csv"
        wall_low_csv_path = wall_dir / "low" / f"wall_low_case_{case_index}.csv"
        if not wall_peak_csv_path.exists() or not wall_low_csv_path.exists():
            print(f"wall_peak_csv_path or wall_low_csv_path does not exist: {wall_peak_csv_path} or {wall_low_csv_path}. pass")
            continue

        #msh or result exists? -> pass
        msh_path = case_dir / "total_solid_type2.msh" # Type1->2 msh changed.
        if not msh_path.exists():
            no_msh.append(case_index)
            print(f"msh_path does not exist: {msh_path}")
            continue

        rst_peak_path = ansys_dir / f"FC_case_{case_index}_peak.vtu"
        rst_low_path = ansys_dir / f"FC_case_{case_index}_low.vtu"
        if rst_peak_path.exists() and rst_low_path.exists():
            print(f"rst_peak_path and rst_low_path exist: {rst_peak_path} and {rst_low_path}. pass")
            continue

        print(f"\n\n************Processing case {case_index}:************\n")
        #1. define the pymapdl_worker instance and launch the mapdl instance.
        pymapdl_worker = PYMAPDL_worker(case_index, wall_dir, ansys_exec_dir, parameter_csv_path, nproc=args.nproc)

        #2. Convert .msh to .cdb + save the surface mesh info for the future bc setting.
        try:
            cdb_path = pymapdl_worker.pymapdl_msh_to_cdb(msh_path, type = 2)
            pymapdl_worker.cdb_path = cdb_path

            #3. Launch MAPDL ONCE, solve both peak and low, then exit.
            pymapdl_worker.pymapdl_launch()
            for bc_type in ["peak", "low"]:
                pymapdl_worker.mapdl.clear()       # reset DB, keep process alive
                pymapdl_worker.mapdl.units("CGS")   # must re-set after clear()

                pymapdl_worker.bc_type = bc_type
                pymapdl_worker.casename = f"case_{pymapdl_worker.case_index}_{pymapdl_worker.bc_type}"
                pymapdl_worker.mapdl.jobname = pymapdl_worker.casename

                #Pre processing
                wall_csv_path = wall_dir / f"{bc_type}" / f"wall_{bc_type}_case_{case_index}.csv"
                pymapdl_worker.pymapdl_prep_bc(wall_csv_path)
                pymapdl_worker.pymapdl_prep_mat(material_json_path)

                #Solve the case.
                pymapdl_worker.pymapdl_solve()

                fc_result = pymapdl_worker._extract_fc_results()
                output_vtu_path = ansys_dir / f"FC_case_{case_index}_{bc_type}.vtu"
                pymapdl_worker._build_fc_vtu(fc_result, str(output_vtu_path))

            pymapdl_worker.safe_mapdl_exit()
            print("="*50)
            print("no mesh:", no_msh)
            print("too large mesh:", too_large_mesh)
            print("="*50)
            print(f"************ Finished case: {case_index}************")
            utils_prep.wipe_out_useless_data(ansys_exec_dir)

        except Exception as e:
            print(f"Error: {e}")
            pymapdl_worker.safe_mapdl_exit()
            continue

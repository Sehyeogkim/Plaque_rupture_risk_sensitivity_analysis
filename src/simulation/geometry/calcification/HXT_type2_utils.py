'''
updated by 1113 jeff
Total process for the type 2. (smooth cal inside the lipid core).
1. lipid - fc_offset으로 distance멀어진 거리의 공간 확보.
2. gmsh에서 볼륨메쉬짜고.msh 로 추출.
3. meshio 에서 vtu로 변환
4. SEED location of the Calcification을 파라메터로 삽입,
5. 그리고 이제, seed에서 progate해서 vtu 생성 cal_mesh.vtu로 output
6. output한 cal.vtu → cal_surface.stl
7. cal_surface.stl  → resample - image  → smoothing 진행 (VTK tool kit 이용)
8. smoothed_cal_surface.stl → cal.step (Free CAD 이용)

1. 이제 다시 gmsh시작한다.
2. import (lipid, fc, lumen, solid) + cal.step까지,
3. cal.step intersect lipid core 하여, lipid core내부에 cal이 존재하도록.
4. gmsh에서 메쉬 제작.


@updated 0316 jeff
- smoothing option avaiable (taubin, laplacian)

- Calcfiication volume is save on the .json file.
'''
import gmsh, meshio
from pathlib import Path
import utils.utils_geo_mesh.utils_gmsh as utils_gmsh
import time
from multiprocessing import Process
from scipy.spatial import cKDTree
import os
import pyvista as pv
import pandas as pd
import math
import utils.utils_geo_mesh.utils_CAD as utils_CAD
import numpy as np
from utils.utils_geo_mesh.main_CAD import CAD_instance_from_idx
from utils.utils_geo_mesh.smooth_vtu_to_stl import _smoothing_vtu
import subprocess


TIMEOUT_SECONDS = 1800

class HXT_mesh_II():
    
    def __init__(self, case_index: int, nproc: int = 12):

        self.case_index = case_index
        self.cur_dir = Path(__file__).parent
        self.nproc = nproc

        #Stored data directory.
        #self.fluid_data_dir = self.cur_dir / "fluid_data"
        self.geo_dir = self.cur_dir / "geo_0605"
        self.solid_dir = self.cur_dir / "solid_data"

        #csv file path.
        # Use the same case-indexed design as the published analysis.
        self.para_csv_path = (Path(__file__).resolve().parents[4]
                              / "data" / "inputs" / "lhs_design_1000.csv")
        self.scripts_dir = self.cur_dir / "pre_data" / "scripts"

        #create CAD instance
        self.vessel_model =  CAD_instance_from_idx(int(case_index), self.para_csv_path)

        #total save directory
        self.case_dir = self.solid_dir / f"case_{self.case_index}"
        self.rst_dir = self.case_dir / "type2_mesh"
        self.rst_dir.mkdir(parents=True, exist_ok=True)

        #step file path
        self.lipid_path = self.case_dir / "lipid.stp"
        self.solid_path = self.case_dir / "solid.stp"
        self.fc_offset_path = self.geo_dir / f"case_{self.case_index}" / "fc_offset.stp"
        self.fc_path = self.case_dir / "fc.stp"
        self.lumen_path = self.case_dir / "lumen.stp"
        
        #Created file path.
        self.cal_dep_json_path = self.rst_dir / "cal_dependent_variables.json" # for the dependent variables of the calcification.

        self.raw_cal_msh_path = self.rst_dir / "cal_subdomain.msh"
        self.cal_vtu_path = self.rst_dir / "cal_subdomain.vtu"
        self.offset_cal_stl_path = self.rst_dir / "offset_cal_subdomain.stl"

        self.prog_cal_vtu_path = self.rst_dir / "cal_propa.vtu"
        self.smooth_cal_stl_path = self.rst_dir / "smo_cal.stl"
        self.scaled_smo_cal_stl_path = self.rst_dir / "smo_cal_scaled.stl"
        self.smooth_cal_step_path = self.rst_dir / "smo_cal.step"
        self.lipid_stl_path = self.rst_dir / "true_lipid.stl"
        self.cal_intersect_lipid_stl_path = self.rst_dir / "ca_lipid_intersect.stl"

        self.final_solid_msh_path = self.rst_dir / "solid_type_2_quad.msh"


    @staticmethod
    def create_cal_mesh(lipid_path: Path, fc_offset_path: Path, save_msh_path: Path, nproc: int = 12):
        
        gmsh.initialize()

        #step1. lipid - fc_offset
        lipid = gmsh.model.occ.import_shapes(str(lipid_path))[0]
        gmsh.model.occ.synchronize()

        fc_offset = gmsh.model.occ.import_shapes(str(fc_offset_path))[0]
        gmsh.model.occ.synchronize()

        gmsh.model.occ.cut([lipid], [fc_offset], removeObject = True, removeTool = True)
        gmsh.model.occ.synchronize()

        gmsh.model.occ.removeAllDuplicates()
        gmsh.model.occ.synchronize()

        #utils_gmsh.gmsh_display(exit = False)

        #create physical tag for lipid as 2.
        volumes = gmsh.model.getEntities(3)
        gmsh.model.addPhysicalGroup(3, [volumes[0][1]], tag=2, name="subdomain")
        print(f"Number of volumes: {len(volumes)}")
        print(f"Volumes: {volumes}")


        #3D meshing
        # Set the global mesh characteristic size
        mesh_size = 0.025 # you may customize this value
        print(f"Number of going to be used CPU cores on gmshing: {nproc}")
        gmsh.option.setNumber("General.NumThreads", nproc)  # Leave some cores free for system
        gmsh.option.setNumber("Mesh.MaxNumThreads2D", nproc)  # Leave some cores free for system
        gmsh.option.setNumber("Mesh.MaxNumThreads3D", nproc)  # Use 12 threads for 3D meshing
        #gmsh.option.setNumber("Mesh.HighOrderOptimize", 1)  # No optimization while meshing.
        gmsh.model.occ.synchronize()
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_size)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size)
        gmsh.option.setNumber("Mesh.Algorithm"          , 5)   # 2D Delaunay 5
        gmsh.option.setNumber("Mesh.Algorithm3D"        , 10)  # 3D Delaunay 1, HXT 10
        gmsh.model.mesh.generate(3)

        gmsh.write(str(save_msh_path))
        print(f"Mesh saved to: {save_msh_path}")
        
        gmsh.finalize()
        return
    
    def Voronoi_tesselation_KDTREE(self, gmsh_path: Path, distance_threshold: float = 0.01, console = True):
        '''
        KDTREE algorithm.
        input:
        gmsh_path:
        console: whether to print the console output
        output: new unstructured grid with calcification with physical tag 8
        '''

        mesh = pv.read_meshio(str(gmsh_path))
        mesh.points *= 0.1  # scale down cuz of inventor.
        print("Cell types in mesh:", set(mesh.celltypes))
        # mesh.save(str(self.save_dir / "raw_cal_mesh.vtu"))

        # read the parameters from the vessel_model instance.
        fraction        = self.vessel_model.fraction
        skew_axial      = self.vessel_model.ca_axial_skewness
        skew_shoulder   = self.vessel_model.ca_shoulder_skewness
        fc_av_th        = self.vessel_model.fc_av_th
        d_fc_ca         = self.vessel_model.d_fc_ca
        strength_axial  = self.vessel_model.ca_axial_strength
        strength_circum = self.vessel_model.ca_shoulder_strength

        if console:
            print("================================================")
            print(f"fraction: {fraction}")
            print(f"skew_axial: {skew_axial}")
            print(f"skew_shoulder: {skew_shoulder}")
            print(f"fc_av_th: {fc_av_th}")
            print(f"d_fc_ca: {d_fc_ca}")
            print(f"strength_axial: {strength_axial}")
            print(f"strength_circum: {strength_circum}")
            print("================================================")

        # 1. lipid domain - only physical tag 2 & tetra cells
        subdomain_mask = (mesh.cell_data["gmsh:physical"] == 2) & (mesh.celltypes == 10)
        all_tetra_indices = np.where(subdomain_mask)[0]
        if console:
            print(f"all_tetra_indices: {len(all_tetra_indices)}")

        # 2. 이들만 뽑아서 별도 subdomain mesh 만들기
        subdomain_mesh = mesh.extract_cells(all_tetra_indices)

        # 3. lipid core 표면 추출 및 표면 점들 추출
        surface = subdomain_mesh.extract_surface()
        surface_points = surface.points  # 표면의 모든 점들 (N x 3)
        
        # 표면 점이 너무 많으면 샘플링하여 KDTree 생성 속도 향상
        # (거리 계산 정확도는 거의 동일하지만 속도는 훨씬 빠름)
        max_surface_points = 50000  # 최대 표면 점 개수
        if len(surface_points) > max_surface_points:
            # 균등하게 샘플링
            indices = np.linspace(0, len(surface_points) - 1, max_surface_points, dtype=int)
            surface_points = surface_points[indices]

        # 4. 각 tetra의 cell center를 구하고, KDTree를 사용하여 표면까지의 거리 계산
        #    -> KDTree가 implicit distance보다 훨씬 빠름
        tet_centers = subdomain_mesh.cell_centers().points  # (n_cells x 3) numpy array
        
        # 표면 점들로 KDTree 생성 (한 번만 생성하면 됨)
        surface_kdtree = cKDTree(surface_points)
        
        # 각 cell center에서 가장 가까운 표면 점까지의 거리 계산
        # (벡터화된 연산으로 매우 빠름)
        dist_to_surface, _ = surface_kdtree.query(tet_centers, k=1)

        # 5. 거리 버퍼 기준 설정:
        safe_mask_subdomain = dist_to_surface >= distance_threshold

        # 6. safe_mask_subdomain은 subdomain_mesh의 cell 인덱스 기준이므로,
        #    이를 원래 mesh의 cell 인덱스로 되돌림
        subdomain_cell_indices = all_tetra_indices[safe_mask_subdomain]

        if console:
            n_total   = len(all_tetra_indices)
            n_kept    = np.count_nonzero(safe_mask_subdomain)
            n_removed = n_total - n_kept
            print("Distance-based erosion method:")
            print(f"  Total tetrahedra (physical=2): {n_total}")
            print(f"  Kept internal (d >= {distance_threshold}): {n_kept}")
            print(f"  Removed near-surface (d < {distance_threshold}): {n_removed}")

        # 이후 로직: subdomain_cell_indices만 가지고 seed propagation, calcification 태깅 진행

            
        #2. Now derive (# of cal) = (# of tetra) * fraction
        total_cell_num = len(subdomain_cell_indices)
        cal_num = round(total_cell_num * fraction)
        print(f"total_cell_num: {total_cell_num}, cal_num: {cal_num}") if console else None

        #3. get the z_max of the subdomain from the gmsh.
        submesh = mesh.extract_cells(subdomain_cell_indices)
        
        #inside the domain where ca can exist, normalize (z_min, z_max) -> (0, 1)
        z_min = submesh.points[:, 2].min()
        z_max = submesh.points[:, 2].max()
        z_seed = z_min + (z_max - z_min) * skew_axial

        #4. calculate θ_seed from lipid_half_angle(z = z_seed) and the distance from the lumen_centere
        θ_lipid_arc_half = utils_CAD.alpha_theta(self.vessel_model, z_seed) #rad
        θ_seed_half = θ_lipid_arc_half * skew_shoulder # this value can be negative.

        #5. calculate the x, y coordinate of the seed_point
        # distance between the lumen_center and the seed_point 
        # (big assumption seed point is the closests to the lumen_center in the subdomain, divided by d_fc_ca)
        d_lumen_seed = utils_CAD.radius_lumen(self.vessel_model, z_seed) + fc_av_th + d_fc_ca
        x_seed = d_lumen_seed * math.sin(θ_seed_half)
        y_seed = d_lumen_seed * math.cos(θ_seed_half) + utils_CAD.y_center_lumen(self.vessel_model, z_seed)
        
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
        
        start_time = time.time()
        calibrated_cells_original_ids = []


        if cal_num > 0:
            print(f"\nStarting KD-Tree based propagation...")

            # --- Parameters to control propagation shape ---
            subdomain_cell_centers = mesh.cell_centers().points[subdomain_cell_indices]

            # --- 2. transform the coordinates: apply the weight to the coordinates ---
            # we scale the coordinates by sqrt(weight)
            # Distance^2 ≈ w_c*(dx^2+dy^2) + w_a*dz^2
            w_axial_sqrt = np.sqrt(strength_axial)
            w_circum_sqrt = np.sqrt(strength_circum)
    
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
            calibrated_cells_original_ids = subdomain_cell_indices[valid_indices]

   
            print(f"KD-Tree search completed in {time.time() - start_time:.4f} seconds.")
            print(f"Selected {len(calibrated_cells_original_ids)} cells for calcification.")
            
            # Assign calcification tag (8) to the selected cells
            mesh.cell_data["gmsh:physical"][calibrated_cells_original_ids] = 8 # allocate new ca_index = 8

        else:
            raise ValueError("cal_num is zero. No cells will be calcified.")



        #####################################################
        ##### 10. Calculate the dependent variables #########
        #####################################################

        #10.1. Total fraction(not in here)
        ca_mask = (mesh.cell_data["gmsh:physical"] == 8) & (mesh.celltypes == 10) # Calcification mask
        ca_cells_indices  = np.where(ca_mask)[0]
        ca_mesh = mesh.extract_cells(ca_cells_indices)
        ca_mesh.save(str(self.prog_cal_vtu_path))
        ca_volume = ca_mesh.volume
        # total_fraction = len(ca_cells_indices) / (len(subdomain_cell_indices))
        # if console:
        #     print(f"total_fraction: {total_fraction}")

        
        #10.2. get the z_max of the subdomain from the gmsh.
        ca_mesh = mesh.extract_cells(ca_cells_indices)
        z_max = ca_mesh.points[:, 2].max()
        z_min = ca_mesh.points[:, 2].min()
        ca_length = z_max - z_min
        if console:
            print(f"ca_length: {round(ca_length, 4)}")


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
            
            lumen_center = np.array([0, utils_CAD.y_center_lumen(self.vessel_model, z), z])
            
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
            lumen_center = np.array([0, utils_CAD.y_center_lumen(self.vessel_model, z_max_arc), z_max_arc])
            print(f"lumen_center: {lumen_center}")
        
        ################################################
        ##### 11. Save the dependent variables #########
        ################################################
        json_path = str(self.cal_dep_json_path)
        # Save total_fraction, ca_length, and ca_arc to a JSON file at model.json_path
        import json
        result_dict = {
            "ca_volume": round(ca_volume, 8),
            "ca_length": round(ca_length, 4),
            "ca_arc": round(ca_arc_max_degrees, 4)
        }
        with open(json_path, "w") as file:
            json.dump(result_dict, file, indent=4)

        return mesh
    
    @staticmethod
    def smooth_vtu_to_stl(vtu_path: Path, stl_path: Path, smooth_mode: str = "taubin"):
        '''
        Smooth VTU to STL with selectable smoothing mode.
        smooth_mode: "taubin" (recommended, volume-preserving) or "laplacian" (stronger, may shrink)
        '''
        if smooth_mode == "laplacian":
            _smoothing_vtu(vtu_path, stl_path,
                           RESOLUTION=[20, 20, 20],
                           ISO_VALUE=0.5,
                           SMOOTH_MODE="laplacian",
                           SMOOTH_STRENGTH=0.1,
                           NUM_ITERATIONS=500)
        elif smooth_mode == "taubin":
            _smoothing_vtu(vtu_path, stl_path,
                           RESOLUTION=[20, 20, 20],
                           ISO_VALUE=0.5,
                           SMOOTH_MODE="taubin",
                           SMOOTH_STRENGTH=0.01,
                           NUM_ITERATIONS=500)
        else:
            raise ValueError(f"Unknown smooth_mode: '{smooth_mode}'. Use 'taubin' or 'laplacian'.")
        return
    
    @staticmethod
    def scale_stl(bf_stl_path: Path, scaled_stl_path: Path, scale_factor: float = 10.0):
        '''
        Scale the STL file by the given scale factor.
        '''
        mesh = pv.read(str(bf_stl_path))
        mesh.points *= scale_factor
        mesh.save(str(scaled_stl_path))
        return mesh

    @staticmethod
    def does_ca_inside_lipid(lipid_stl_path: Path, ca_stl_path: Path,
                             tol: float = 1e-6) -> bool:
        """
        Check if CA mesh is inside lipid mesh using select_enclosed_points.
        
        Args:
            lipid_stl_path: Path to lipid STL file
            ca_stl_path: Path to CA STL file
            tol: Tolerance for boundary point detection
            
        Returns:
            True if all CA points are inside lipid, False otherwise
        """
        # Load and prepare meshes
        lipid = pv.read(str(lipid_stl_path)).triangulate().clean()
        ca = pv.read(str(ca_stl_path)).triangulate()
        lipid = lipid.compute_normals(auto_orient_normals=True)

        # Check if CA points are inside lipid surface
        enclosed = ca.select_enclosed_points(lipid, tolerance=tol, check_surface=False)
        mask = np.asarray(enclosed.point_data["SelectedPoints"]).astype(np.bool_)
        all_inside = mask.all()

        print(f"CA is {'inside' if all_inside else 'NOT fully inside'} the lipid.")
        return all_inside
        
    @staticmethod
    def get_lipid_stl(lipid_stp_path: Path, fc_stp_Path: Path, lipid_stl_path: Path):
        '''
        Convert lipid STEP file to STL file using Gmsh.
        '''
        #lipid stp -> stl
        gmsh.initialize()
        lipid = gmsh.model.occ.import_shapes(str(lipid_stp_path))[0]
        gmsh.model.occ.synchronize()

        fc = gmsh.model.occ.import_shapes(str(fc_stp_Path))[0]
        gmsh.model.occ.synchronize()

        gmsh.model.occ.cut([lipid], [fc], removeObject = True, removeTool = True)
        gmsh.model.occ.synchronize()

        gmsh.model.occ.removeAllDuplicates()
        gmsh.model.occ.synchronize()

        #charcateristic length
        mesh_size = 0.05
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_size)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size)
        gmsh.model.mesh.generate(2)
        gmsh.write(str(lipid_stl_path))
        gmsh.finalize()

        import meshio
        mesh = meshio.read(str(lipid_stl_path))
        mesh.points *= 0.1
        meshio.write(str(lipid_stl_path), mesh)
        print(f"Lipid STL saved to: {lipid_stl_path}")

        return
    
    @staticmethod
    def msh_to_vtu(msh_path: Path):
        '''
        Convert msh file to vtu file.
        '''
        msh_path = str(msh_path)
        vtu_path = msh_path.replace('.msh', '.vtu')
        mesh = meshio.read(msh_path)
        mesh.points *= 0.1
        
        # VTU format doesn't support cell_sets, so remove them before writing
        # This prevents the IndexError that occurs during automatic conversion
        if mesh.cell_sets:
            mesh.cell_sets = {}
        
        meshio.write(vtu_path, mesh)
        print(f"Msh file saved to: {vtu_path}")
        return
    
    @staticmethod
    def stl_to_step(stl_path: Path, step_path: Path, script_dir: Path):
        """
        Convert STL file to STEP file using FreeCAD.
        
        Args:
            stl_path: Path to input STL file
            step_path: Path to output STEP file
            script_dir: Path to the directory containing the stl_to_step.py script
        """
        # Path to stl_to_step.py script
        stl_to_step_script = script_dir / "stl_to_step.py"
        
        if not stl_to_step_script.exists():
            raise FileNotFoundError(f"stl_to_step.py script not found at: {stl_to_step_script}")
        
        if not stl_path.exists():
            raise FileNotFoundError(f"STL file not found: {stl_path}")
        
        
        print(f"Converting STL to STEP: {stl_path} -> {step_path}")

        # Execute FreeCAD command using subprocess
        # freecadcmd -c "import sys; sys.path.append('script_dir'); exec(open('stl_to_step.py').read())"
        # But we need to pass arguments, so we use a different approach
        cmd = [
            "freecadcmd",
            "-c",
            f"import sys; import os; sys.path.insert(0, '{script_dir}'); "
            f"sys.argv = ['stl_to_step.py', '{stl_path}', '{step_path}']; "
            f"exec(open('{stl_to_step_script}').read())"
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            print(result.stdout)
            if result.stderr:
                print("Warnings:", result.stderr)
            print(f"Successfully converted to STEP: {step_path}")
        except subprocess.CalledProcessError as e:
            print(f"Error during conversion: {e}")
            print(f"stdout: {e.stdout}")
            print(f"stderr: {e.stderr}")
            raise
        except FileNotFoundError:
            raise RuntimeError(
                "freecadcmd not found. Please ensure FreeCAD is installed and available in PATH.\n"
                "Installation: conda install -c conda-forge freecad"
            )
        
        return
   
    def stl_boolean(self):
      
        # Load STL files using PyVista
        lipid_mesh = pv.read(str(self.lipid_stl_path))
        cal_mesh = pv.read(str(self.smooth_cal_stl_path))
        
        print(f"Lipid mesh: {lipid_mesh.n_points} points, {lipid_mesh.n_cells} faces")
        print(f"Cal mesh: {cal_mesh.n_points} points, {cal_mesh.n_cells} faces")
        
        # Ensure meshes are clean (remove degenerate cells, etc.)
        lipid_mesh = lipid_mesh.clean(tolerance=1e-6)
        cal_mesh = cal_mesh.clean(tolerance=1e-6)
        
        # Perform boolean intersection operation (cal ∩ lipid)
        # This keeps only the overlapping region between cal and lipid
        try:
            result_mesh = cal_mesh.boolean_intersection(lipid_mesh)
            print(f"Boolean intersection operation completed successfully")
            print(f"Result mesh: {result_mesh.n_points} points, {result_mesh.n_cells} faces")
            
            # Save the result
            result_mesh.save(str(self.cal_intersect_lipid_stl_path))
            print(f"Result saved to: {self.cal_intersect_lipid_stl_path}")
            
        except Exception as e:
            print(f"Error during boolean operation: {e}")
            print("Trying alternative method with VTK...")
            
            # Alternative: Use VTK directly
            import vtk
            reader1 = vtk.vtkSTLReader()
            reader1.SetFileName(str(self.lipid_stl_path))
            reader1.Update()
            lipid_polydata = reader1.GetOutput()
            
            reader2 = vtk.vtkSTLReader()
            reader2.SetFileName(str(self.smooth_cal_stl_path))
            reader2.Update()
            cal_polydata = reader2.GetOutput()
            
            # Clean the polydata
            cleaner1 = vtk.vtkCleanPolyData()
            cleaner1.SetInputData(lipid_polydata)
            cleaner1.Update()
            
            cleaner2 = vtk.vtkCleanPolyData()
            cleaner2.SetInputData(cal_polydata)
            cleaner2.Update()
            
            # Boolean operation - Intersection
            boolean_filter = vtk.vtkBooleanOperationPolyDataFilter()
            boolean_filter.SetOperationToIntersection()  # Intersection operation
            boolean_filter.SetInputData(0, cleaner2.GetOutput())  # cal first
            boolean_filter.SetInputData(1, cleaner1.GetOutput())  # lipid second
            boolean_filter.Update()
            
            result_polydata = boolean_filter.GetOutput()
            
            # Save result
            writer = vtk.vtkSTLWriter()
            writer.SetFileName(str(self.cal_intersect_lipid_stl_path))
            writer.SetInputData(result_polydata)
            writer.SetFileTypeToBinary()
            writer.Write()
            
            print(f"VTK boolean intersection operation completed")
            print(f"Result saved to: {self.cal_intersect_lipid_stl_path}")
        


        return
    
    @staticmethod
    def gmsh_healshape(step_path: Path):

        gmsh.initialize()
        gmsh.model.add("Stenosis Model")
        gmsh.option.setNumber("General.Terminal", 1)   

        shape = gmsh.model.occ.importShapes(str(step_path))[0]
        gmsh.model.occ.synchronize()

        gmsh.model.occ.healShapes([shape])
        gmsh.model.occ.synchronize()

        #utils_gmsh.gmsh_display(exit = True)

    def solid_gmshing(self, mesh_size: float = 0.05, nproc: int = 10,
                      order: int = 2, second_order_linear: bool = True,
                      high_order_optimize: int = 0, box_vout: float = 0.2,
                      optimize: bool = True, write: bool = True,
                      algorithm_3d: int = 10):
        '''
        Mesh the multi-material solid and return quality stats.

        Returns a dict: {mesh_size, n_tet, n_nodes, min_quality, mean_quality,
                         n_negative, seconds}.

        Quadratic note: with second_order_linear=True the mid-side nodes are placed
        on straight edges, so a valid linear tet stays valid as a 10-node tet (no
        negative Jacobian) while still giving quadratic displacement interpolation
        (the part that fixes the linear-tet stress speckle). curved (=False) follows
        the CAD surfaces more closely but mid-side nodes can invert on thin/curved
        regions like the fibrous cap -> negative quality; only use with
        high_order_optimize>=2.
        '''

        start_time = time.time()
        gmsh.initialize()
        gmsh.model.add("Stenosis Model")
        gmsh.option.setNumber("General.Terminal", 1)

        #STEP1: lipid n fc -> new fc
        lipid = gmsh.model.occ.importShapes(str(self.lipid_path))[0]   #[(3,1)]
        gmsh.model.occ.synchronize()

        fc    = gmsh.model.occ.importShapes(str(self.fc_path))[0]     #[(3,2)]
        gmsh.model.occ.synchronize()

        gmsh.model.occ.intersect([lipid], [fc], removeObject = False, removeTool = True) 
        gmsh.model.occ.synchronize()

        gmsh.model.occ.removeAllDuplicates() # Intersection of the lipid and fc varies the lipid core.
        gmsh.model.occ.synchronize()
        
        lipid = (3,3)

        #Check if the entities are in the volumes
        volumes = gmsh.model.getEntities(3)
        if lipid and fc in volumes:
            print("STEP1 complete, lipid and fc are in the volumes")
        else:
            raise Exception("STEP1 Failed.")


        '''
        fc - (3,2)
        lipid - (3,3)
        '''

        # #STEP3 fc - lumen -> new fc
        lumen = gmsh.model.occ.importShapes(str(self.lumen_path))[0] #(3,4)
        gmsh.model.occ.synchronize()

        gmsh.model.occ.cut([fc], [lumen], removeObject = True, removeTool = False)
        gmsh.model.occ.synchronize()

        gmsh.model.occ.removeAllDuplicates()
        gmsh.model.occ.synchronize()


        volumes = gmsh.model.getEntities(3)
        if fc and lumen in volumes:
            print("STEP3 complete, fc and solid are in the volumes")
        else:
            raise Exception("STEP3 Failed.")
        


        #STEP4 solid - (lipid+fc+lumen) -> new solid
        solid = gmsh.model.occ.importShapes(str(self.solid_path))[0]
        gmsh.model.occ.synchronize()
        

        gmsh.model.occ.cut([solid], [lipid, fc, lumen], removeObject = False, removeTool = False)
        gmsh.model.occ.synchronize()

        gmsh.model.occ.removeAllDuplicates()
        gmsh.model.occ.synchronize()

        #solid = (3,6)

        volumes = set(gmsh.model.getEntities(3))
        solid = volumes - set([lipid, fc, lumen])
        
        if len(solid) != 1:
            raise Exception("STEP4 Failed.")
        else:
            print("STEP4 complete, solid is in the volumes")
            solid = solid.pop() # solid - (3,6)
        
        # #now import the smooth_cal_step file.
        smooth_cal = gmsh.model.occ.importShapes(str(self.smooth_cal_step_path))[0]
        gmsh.model.occ.synchronize()
        
        lipid_center = gmsh.model.occ.getCenterOfMass(lipid[0], lipid[1])
        fc_center = gmsh.model.occ.getCenterOfMass(fc[0], fc[1])
        lumen_center = gmsh.model.occ.getCenterOfMass(lumen[0], lumen[1])
        solid_center = gmsh.model.occ.getCenterOfMass(solid[0], solid[1])
        smooth_cal_center = gmsh.model.occ.getCenterOfMass(smooth_cal[0], smooth_cal[1])


        gmsh.model.occ.healShapes([smooth_cal])
        gmsh.model.occ.synchronize()


        volumes = set(gmsh.model.getEntities(3))
        for vol in volumes:
            vol_center = gmsh.model.occ.getCenterOfMass(vol[0], vol[1])
            if np.allclose(vol_center, smooth_cal_center, atol=1e-6):
                smooth_cal = vol
                continue
            elif np.allclose(vol_center, solid_center, atol=1e-6):
                solid = vol
                continue
            elif np.allclose(vol_center, lipid_center, atol=1e-6):
                lipid = vol
                continue
            elif np.allclose(vol_center, fc_center, atol=1e-6):
                fc = vol
                continue
            elif np.allclose(vol_center, lumen_center, atol=1e-6):
                lumen = vol
                continue
            else:
                raise Exception("STEP5 Failed.")
        # # gmsh.model.occ.convertToNURBS([smooth_cal])
        # # gmsh.model.occ.synchronize()

        gmsh.model.occ.cut([lipid], [smooth_cal], removeObject = True, removeTool = False)
        gmsh.model.occ.synchronize()
    
        ############################################
        ############# PHYSICAL TAGGING #############
        ############################################
        gmsh.model.addPhysicalGroup(3, [solid[1]], tag=1, name="solid")
        gmsh.model.addPhysicalGroup(3, [lipid[1]], tag=2, name="lipid")
        gmsh.model.addPhysicalGroup(3, [fc[1]], tag=3, name="fc")
        gmsh.model.addPhysicalGroup(3, [smooth_cal[1]], tag=8, name="ca")

        # Extract volume tags from lumen
        lumen_volume_tags = [lumen[1]]

        # Get the boundary surfaces between solid and fc fc = (3,7)
        fc_surfaces    = [abs(s[1]) for s in gmsh.model.getBoundary([fc],    oriented=False)]
        solid_surfaces = [abs(s[1]) for s in gmsh.model.getBoundary([solid], oriented=False)]
        lipid_surfaces = [abs(s[1]) for s in gmsh.model.getBoundary([lipid], oriented=False)]

        wall_in_fc_tags    = utils_gmsh.find_wall_surfaces(fc_surfaces,    lumen_volume_tags)
        wall_in_solid_tags = utils_gmsh.find_wall_surfaces(solid_surfaces, lumen_volume_tags)  
        gmsh.model.occ.synchronize()

        lipid_in_fc_tags = list(set(lipid_surfaces) & set(fc_surfaces))
        gmsh.model.addPhysicalGroup(2, wall_in_fc_tags, tag=5, name="wall_in_fc")
        gmsh.model.addPhysicalGroup(2, wall_in_solid_tags, tag=4, name="wall_in_solid")
        gmsh.model.addPhysicalGroup(2, lipid_in_fc_tags, tag=7, name="lipid_in_fc")
        gmsh.model.occ.synchronize()
        
        all_solid_surfaces = gmsh.model.getBoundary([solid], oriented=False)

        surface_centers = []
        for surface in all_solid_surfaces:
            com = gmsh.model.occ.getCenterOfMass(surface[0], abs(surface[1]))
            surface_centers.append(com[2])

        # Convert to numpy array for easy math
        surface_centers = np.array(surface_centers)

        # Find indices closest to -20.0 and +80.0
        side1_index = np.argmin(np.abs(surface_centers - (-20.0)))
        side2_index = np.argmin(np.abs(surface_centers - 80.0))

        # Correct: pick from all_solid_surfaces, not from surface_centers
        side1_surface = all_solid_surfaces[side1_index]
        side2_surface = all_solid_surfaces[side2_index]

        # Add physical group for inlet and outlet
        gmsh.model.addPhysicalGroup(2, [abs(side1_surface[1]), abs(side2_surface[1])], tag=6, name="Two_sides")
        gmsh.model.occ.synchronize()

        #remove lumen
        gmsh.model.occ.remove([lumen])
        gmsh.model.occ.synchronize()

        #CHECK The physical groups are all in the list
        for (dim, tag) in gmsh.model.getPhysicalGroups(3):
            if tag not in [1,2,3,8]:
                raise Exception(f"(dim = 3)Physical group {tag} is not in the list")

        for (dim, tag) in gmsh.model.getPhysicalGroups(2):
            if tag not in [4,5,6,7]:
                raise Exception(f"(dim = 2)Physical group {tag} is not in the list")
        
        #utils_gmsh.gmsh_display(exit = True)

        ####################################
        ############# MESHING ##############
        ####################################
        '''

            B. Meshing Procedures
                1. Generate distance field (Criteria: wall_in_fc  or all_fc_faces)
                2. Generate threshold field from the distance field
                3. Set mesh options as below.
                4. Generate mesh
                5. save as msh file
                6. save as vtu file(0.1 scaled)
            
            #Mesh options need to be considered
                1. Fibrou cap mesh size - 0.05, other wise 0.1
                2. Number of CPU cores - 32 if available > 32, otherwise 10.
                3. Smoothing must be used in order to get a posiive Jacobi ratio.
                4. Quadratic tetra
            
            [Mesh Algorithm Types]
            1	Delaunay	        The fastest, handles complex mesh size fields well (default)
            4	Frontal	            Front-based, generates cleaner elements for boundary shapes
            5	Frontal Delaunay	Mix of Delaunay and Frontal for higher quality
            6	Frontal Hex	        Specify on hexa Generation 
            7	MMG3D	            External MMG library based adaptive mesh reconstruction
            9	R-tree	            Spatial partitioning tree based (large-scale parallel partitioning, etc.)

            2D
            1: MeshAdapt, 2: Automatic, 3: Initial mesh only, 
            5: Delaunay,  6: Frontal-Delaunay, 7: BAMG, 8: Frontal-Delaunay for Quads, 
            9: Packing of Parallelograms, 11: Quasi-structured Quad

            3D
            1: Delaunay, 3: Initial mesh only, 
            4: Frontal, 7: MMG3D, 9: R-tree, 10: HXT
        '''
        
        #CPU core setting
        print(f"Number of going to be used CPU cores on gmshing: {nproc}")
        utils_gmsh.slash_lines()    
        gmsh.option.setNumber("General.NumThreads", nproc)  # Leave some cores free for system
        gmsh.option.setNumber("Mesh.MaxNumThreads2D", nproc)  # Leave some cores free for system
        gmsh.option.setNumber("Mesh.MaxNumThreads3D", nproc)  # Use 12 threads for 3D meshing
        ''']
        ┌─────┬────────────────────────┬───────────┬──────────────────────────────────┬─────────────────────────────────────────────┐
        │ 값  │        알고리즘        │   속도    │            Robustness            │                    품질                     │
        ├─────┼────────────────────────┼───────────┼──────────────────────────────────┼─────────────────────────────────────────────┤
        │ 0   │ optimization 없음      │ 가장 빠름 │ 항상 "성공"                      │ Negative Jacobian 그대로 남음 → solver fail │
        ├─────┼────────────────────────┼───────────┼──────────────────────────────────┼─────────────────────────────────────────────┤
        │ 1   │ Standard optimization  │ 느림      │ 낮음 (수렴 못 하면 timeout/fail) │ 성공하면 최고                               │
        ├─────┼────────────────────────┼───────────┼──────────────────────────────────┼─────────────────────────────────────────────┤
        │ 2   │ Elastic + optimization │ 가장 느림 │ 중간                             │ 1과 비슷                                    │
        ├─────┼────────────────────────┼───────────┼──────────────────────────────────┼─────────────────────────────────────────────┤
        │ 3   │ Elastic only           │ 중간      │ 중간                             │ 중간                                        │
        ├─────┼────────────────────────┼───────────┼──────────────────────────────────┼─────────────────────────────────────────────┤
        │ 4   │ Fast curving           │ 빠름      │ 높음                             │ 약간 낮지만 충분                            │
        └─────┴────────────────────────┴───────────┴──────────────────────────────────┴─────────────────────────────────────────────┘
        '''
        gmsh.model.occ.synchronize()

        # Add box field around lipid core for refined meshing
        box_field = gmsh.model.mesh.field.add("Box")
        gmsh.model.mesh.field.setNumber(box_field, "Thickness", 3.0)  # Transition layer thickness
        gmsh.model.mesh.field.setNumber(box_field, "VIn", mesh_size)  # Fine mesh size inside box
        gmsh.model.mesh.field.setNumber(box_field, "VOut", box_vout)   # Regular mesh size outside box
        
        # Get bounding box of lipid volume (lipid2)
        z_lesion = abs(self.vessel_model.lesion_length) / 2 * 10 # cm -> mm.
        gmsh.model.mesh.field.setNumber(box_field, "XMin", -10.0)
        gmsh.model.mesh.field.setNumber(box_field, "XMax", 10.0)
        gmsh.model.mesh.field.setNumber(box_field, "YMin", -10.0)
        gmsh.model.mesh.field.setNumber(box_field, "YMax", 10.0)
        gmsh.model.mesh.field.setNumber(box_field, "ZMin", -z_lesion)  #revise the z in here.
        gmsh.model.mesh.field.setNumber(box_field, "ZMax", z_lesion)

        # Set box field as background field
        gmsh.model.mesh.field.setAsBackgroundMesh(box_field)

        gmsh.option.setNumber("Mesh.Algorithm"          , 5)   # 2D Delaunay 5
        gmsh.option.setNumber("Mesh.Algorithm3D"        , algorithm_3d)  # 1=Delaunay, 10=HXT
        gmsh.option.setNumber("Mesh.Smoothing"          , 5)   # Laplacian smoothing passes
        # Optimise the LINEAR base mesh first so the worst tets are cleaned up.
        # With second_order_linear this directly fixes the quadratic quality too,
        # because straight-edged 10-node tets inherit the linear tet Jacobian.
        gmsh.option.setNumber("Mesh.Optimize"           , 1 if optimize else 0)
        gmsh.option.setNumber("Mesh.OptimizeNetgen"     , 1 if optimize else 0)
        gmsh.model.occ.synchronize()

        # 1) generate the linear mesh, 2) optimise it, 3) raise to high order
        gmsh.option.setNumber("Mesh.ElementOrder", 1)
        gmsh.model.mesh.generate(3)
        if optimize:
            try:
                gmsh.model.mesh.optimize("Netgen")
            except Exception as e:
                print(f"[mesh] Netgen optimize skipped: {e}")

        if order >= 2:
            gmsh.option.setNumber("Mesh.SecondOrderLinear", 1 if second_order_linear else 0)
            gmsh.option.setNumber("Mesh.HighOrderOptimize", high_order_optimize)
            gmsh.option.setNumber("Mesh.HighOrderPassMax", 25)
            gmsh.model.mesh.setOrder(2)
            if high_order_optimize and not second_order_linear:
                try:
                    gmsh.model.mesh.optimize("HighOrder")
                except Exception as e:
                    print(f"[mesh] HighOrder optimize skipped: {e}")

        utils_gmsh.check_mesh_quality(0)

        # --- collect stats (tet quality = scaled jacobian / minSICN) ---
        QualityType = 0
        gmsh.option.setNumber("Mesh.QualityType", QualityType)
        etypes, etags, _ = gmsh.model.mesh.getElements(dim=3)
        n_tet = int(sum(len(t) for t in etags))
        quals = []
        for t in etags:
            if len(t):
                quals.extend(gmsh.model.mesh.getElementQualities(t))
        quals = np.asarray(quals) if quals else np.array([0.0])
        node_tags, _, _ = gmsh.model.mesh.getNodes()
        stats = {
            "mesh_size": mesh_size, "order": order,
            "second_order_linear": second_order_linear,
            "n_tet": n_tet, "n_nodes": int(len(node_tags)),
            "min_quality": float(quals.min()), "mean_quality": float(quals.mean()),
            "n_negative": int((quals <= 0).sum()),
            "seconds": time.time() - start_time,
        }
        print(f"[mesh] size={mesh_size} order={order} SOL={second_order_linear} "
              f"-> n_tet={n_tet:,} nodes={len(node_tags):,} "
              f"min_q={stats['min_quality']:.4f} mean_q={stats['mean_quality']:.4f} "
              f"neg={stats['n_negative']} ({stats['seconds']:.1f}s)")

        if write:
            gmsh.write(str(self.final_solid_msh_path))
            print("!!!Gmsh mesh completely saved on:", self.final_solid_msh_path)
        gmsh.finalize()
        if write:
            utils_gmsh.get_mesh_info(self.final_solid_msh_path) # .msh file info by meshio
        print(f"\nTime taken for gmshing: {time.time() - start_time} seconds\n")
        return stats

    def solid_gmshing_sweep(self, sizes=(0.05, 0.055, 0.06, 0.065), nproc: int = 10,
                            max_elements: int = 5_000_000, min_quality: float = 0.0,
                            second_order_linear: bool = True, box_vout: float = 0.2):
        '''
        Find a good quadratic mesh size automatically and write it.

        Strategy: walk `sizes` from fine -> coarse. For each, mesh and keep the
        FINEST size whose element count is < max_elements, has no inverted (negative)
        elements, and whose min tet quality > min_quality. The first size that
        satisfies all three is the best (finest allowed) and we stop there.

        Rationale (case 0 calibration): smaller size -> MORE elements, so to stay
        under the 5M budget we coarsen, not refine. SecondOrderLinear=True keeps the
        10-node tets valid (no negative Jacobian) at every size, so quality is no
        longer the binding constraint -- element count is. e.g. case 0:
            0.05  -> 5.64M tets (over budget)   -> reject
            0.055 -> 1.52M tets, min_q 0.166    -> ACCEPT
        '''
        results = []
        chosen = None
        for s in sizes:
            st = self.solid_gmshing(mesh_size=s, nproc=nproc, order=2,
                                    second_order_linear=second_order_linear,
                                    high_order_optimize=0, box_vout=box_vout,
                                    optimize=True, write=False)
            results.append(st)
            ok = (st["n_tet"] < max_elements
                  and st["n_negative"] == 0
                  and st["min_quality"] > min_quality)
            print(f"[sweep] size={s} n_tet={st['n_tet']:,} min_q={st['min_quality']:.4f} "
                  f"neg={st['n_negative']} -> {'ACCEPT' if ok else 'reject'}")
            if ok:
                chosen = st
                break

        if chosen is None:
            raise RuntimeError(
                f"No size in {sizes} met budget<{max_elements} & min_q>{min_quality}. "
                f"Tried: {[(r['mesh_size'], r['n_tet'], round(r['min_quality'],3)) for r in results]}")

        # Re-mesh the winner and WRITE it (the loop above probed with write=False).
        final = self.solid_gmshing(mesh_size=chosen["mesh_size"], nproc=nproc, order=2,
                                   second_order_linear=second_order_linear,
                                   high_order_optimize=0, box_vout=box_vout,
                                   optimize=True, write=True)
        print(f"[sweep] FINAL size={chosen['mesh_size']} "
              f"n_tet={final['n_tet']:,} min_q={final['min_quality']:.4f} written to "
              f"{self.final_solid_msh_path}")
        return final

    def solid_gmshing_quality_sweep(self, sizes=(0.05, 0.045, 0.04), min_quality: float = 0.0,
                                    nproc: int = 10, second_order_linear: bool = True,
                                    box_vout: float = 0.2):
        '''
        Quality-driven size sweep (user spec): start at the COARSEST size (0.05) and
        only go finer (0.045 -> 0.04) if the quality is bad. Accept the first size
        whose tets are valid (no negative Jacobian) and whose min quality > min_quality.
        No element-count cap -- element count is whatever that size produces.

        With second_order_linear=True the 10-node tets stay valid at every size, so
        0.05 normally passes immediately. Writes each attempted size (write=True) and
        stops at the first acceptable one, so the winning .msh is already on disk
        (no wasted re-mesh). If none pass, the finest size's mesh is kept.
        '''
        results = []
        for s in sizes:
            st = self.solid_gmshing(mesh_size=s, nproc=nproc, order=2,
                                    second_order_linear=second_order_linear,
                                    high_order_optimize=0, box_vout=box_vout,
                                    optimize=True, write=True)
            results.append(st)
            ok = (st["n_negative"] == 0) and (st["min_quality"] > min_quality)
            print(f"[qsweep] size={s} n_tet={st['n_tet']:,} min_q={st['min_quality']:.4f} "
                  f"neg={st['n_negative']} -> {'ACCEPT' if ok else 'bad quality, refine'}")
            if ok:
                st["accepted"] = True
                return st
        # none passed: keep the best (highest min_quality); its msh may have been
        # overwritten by a later size, so re-write the best one.
        best = max(results, key=lambda r: r["min_quality"])
        print(f"[qsweep] no size passed min_q>{min_quality}; re-writing best size={best['mesh_size']} "
              f"(min_q={best['min_quality']:.4f})")
        final = self.solid_gmshing(mesh_size=best["mesh_size"], nproc=nproc, order=2,
                                   second_order_linear=second_order_linear,
                                   high_order_optimize=0, box_vout=box_vout,
                                   optimize=True, write=True)
        final["accepted"] = False
        return final

    def solid_gmshing_production(self, nproc: int = 20, min_quality: float = 0.0,
                                 max_tet: int = 6_000_000, coarsen_size: float = 0.055,
                                 second_order_linear: bool = True, box_vout: float = 0.2):
        '''
        Production mesh policy (simulation_goal.md):
          * start size 0.05, HXT (Algorithm3D=10)
          * HXT meshing fails  -> retry the SAME size with Delaunay (Algorithm3D=1)
          * element count > max_tet (6M) -> re-mesh coarser at coarsen_size (0.055)
          * STEP1-5 (geometry boolean) fails -> re-raise so the case is SKIPPED
          * quality bad (min_q <= min_quality or any negative tet) -> refine 0.045 -> 0.04
          * nothing acceptable -> raise RuntimeError (skip the case)
        Writes the accepted mesh to self.final_solid_msh_path; returns its stats dict
        (with extra key 'algorithm_3d'). SecondOrderLinear=True keeps 10-node tets valid,
        so the quality-refine path rarely fires; the binding issues are HXT robustness,
        the 6M cap, and geometry (STEP) failures.
        '''
        last_err = None
        for size in (0.05, 0.045, 0.04):
            for algo in (10, 1):  # HXT first, Delaunay fallback
                algo_name = "HXT" if algo == 10 else "Delaunay"
                try:
                    st = self.solid_gmshing(mesh_size=size, nproc=nproc, order=2,
                                            second_order_linear=second_order_linear,
                                            high_order_optimize=0, box_vout=box_vout,
                                            optimize=True, write=True, algorithm_3d=algo)
                except Exception as e:
                    if gmsh.is_initialized():
                        gmsh.finalize()            # solid_gmshing aborts before finalize
                    # Only a MESHING failure (HXT/Delaunay 3D mesh) is fixable by changing
                    # size/algorithm. A geometry failure (STEP1-5 boolean / OCC entity /
                    # healShapes) fails identically regardless of size/algo -> skip the case
                    # immediately instead of wasting all 6 size/algo attempts.
                    msg = str(e)
                    is_mesh_err = any(k in msg for k in
                                      ("HXT", "PLC", "mesh failed", "recover",
                                       "constrained", "Self-intersect", "intersect"))
                    if not is_mesh_err:
                        raise
                    print(f"[prod] size={size} {algo_name} 3D mesh failed: {msg[:120]} -> next")
                    last_err = e
                    continue

                # >6M elements: coarsen once to coarsen_size (same algorithm)
                if st["n_tet"] > max_tet:
                    print(f"[prod] size={size} -> {st['n_tet']:,} > {max_tet:,}; coarsen to {coarsen_size}")
                    try:
                        st = self.solid_gmshing(mesh_size=coarsen_size, nproc=nproc, order=2,
                                                second_order_linear=second_order_linear,
                                                high_order_optimize=0, box_vout=box_vout,
                                                optimize=True, write=True, algorithm_3d=algo)
                    except Exception as e:
                        if gmsh.is_initialized():
                            gmsh.finalize()
                        msg = str(e)
                        if not any(k in msg for k in ("HXT", "PLC", "mesh failed", "recover",
                                                      "constrained", "Self-intersect", "intersect")):
                            raise
                        last_err = e
                        continue

                ok = (st["n_negative"] == 0) and (st["min_quality"] > min_quality)
                print(f"[prod] size={st['mesh_size']} {algo_name} n_tet={st['n_tet']:,} "
                      f"nodes={st['n_nodes']:,} min_q={st['min_quality']:.4f} "
                      f"neg={st['n_negative']} -> {'ACCEPT' if ok else 'bad quality, refine'}")
                if ok:
                    st["algorithm_3d"] = algo
                    return st
                break  # quality bad with a valid mesh -> go to next (finer) size

        raise RuntimeError(f"mesh skip: no size/algo gave min_q>{min_quality} (last_err={last_err})")

    def remove_redundant_files(self):
        # Keep ONLY the quadratic mesh (*quad.msh) and the cal-volume json; drop the
        # heavy intermediates (cal_subdomain.msh, *.vtu, smo_cal.*, true_lipid.stl,
        # linear .msh, etc). NOTE: must keep *quad.msh — ANSYS reads it next.
        for file in self.rst_dir.glob("*"):
            if file.is_file() and not (file.name.endswith("quad.msh") or file.suffix == ".json"):
                file.unlink()

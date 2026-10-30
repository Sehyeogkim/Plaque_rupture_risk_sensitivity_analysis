import os
import shutil 
import numpy as np
import pandas as pd
from utils.utils_geo_mesh.utils_CAD import *

class VesselCADModel:

    def __init__(self, 
                PI, alpha, 
                lipid_length_ratio, fc_av_th, 

                d_fc_ca, fraction, 
                ca_axial_skewness, ca_shoulder_skewness,

                ca_axial_strength, ca_shoulder_strength,
                
                #New Lumen Morphology parameters
                DOS, lesion_length, lumen_axial_skewness,

                r_max = 0.1,  thick_ratio = 0.2):
        
        '''
        @ Latest Update 0813 Jeff
        - Vornoli Tesslation method parameters(0731).
        - New Lumen Morphology parameters are added(0813).
        '''

        ############################################################
        #Lumen parameters.
        self.r_max = r_max
        self.r_min = (1 - DOS) * r_max # DOS : 1 - Dmin / Dmax (Diameter based)
        self.lesion_length = lesion_length
        self.lumen_skewness = lumen_axial_skewness  # [-1, 1]

        #Variables for the lumen skewness.
        self.z_start = -2.0 #cm 
        self.z_end   =  8.0 #cm
        self.z_lesion = self.lesion_length / 2
        self.z_peak  = self.lumen_skewness * self.z_lesion # min R, z coordinate (varies depend on the lumen skewness)


        self.T        = self.lesion_length #Period of the sinousal functions that frequently defined in the CAD codes.
        
        pass

    @staticmethod
    def CAD_instance_from_idx(case_idx: int, case_csv_path: str):
        '''
        Create a CAD instance on the sepcific case_idx
        return the CAD instance.
        '''
        para_dict = pd.read_csv(case_csv_path).iloc[case_idx].to_dict()

        CAD_instance = VesselCADModel(
            para_dict["PI"], 
            para_dict["alpha"], 
            para_dict["lipid_length_ratio"], 
            para_dict["fc_av_th"],  
            para_dict["d_fc_ca"], 
            para_dict["fraction"], 
            para_dict["ca_axial_skewness"], 
            para_dict["ca_shoulder_skewness"],
            para_dict["ca_strength_ratio"], 
            1.0,
            para_dict["DOS"],
            para_dict["lesion_length"],
            para_dict["lumen_axial_skewness"],
            r_max = 0.1
        )        

        return CAD_instance

    def tree_xyzts_generator(self, tree_path = "tree.dat", xyzt_path = "xyzts.dat", num_points = 30, unit = "cm"):

        '''
            Input: 
            - model: VesselModel instance
            - tree_path: path to save the tree.dat file
            - xyzt_path: path to save the xyzts.dat file
            - num_points: number of points to create the files
            - unit: "mm" or "cm"

            Output:
            - No return but save two files.


            ########### Procedures ###########
            1. Define the range of z coordinate to create the files.
            2. Along the z coordinate, save x,y,z,r as tree.dat format.
            3. Create xyzts.dat file
        '''

        if unit == "mm":
            factor = 10.0
        elif unit == "cm":
            factor = 1.0
        else:
            raise ValueError("Invalid unit. Please use 'mm' or 'cm'.")


        #1. Define the range of z coordinate.
        # z = z_left + z_right (middle is z_peak where min area is located)
        #self.z_end = self.z_end - 0.1
        length_left = self.z_peak - self.z_start
        total_length = self.z_end - self.z_start
        num_left = int(length_left / total_length * num_points)
        num_right = num_points - num_left

        # lumen [self.z_start -> self.z_leak -> self.z_end]
        z_left = np.linspace(self.z_start, self.z_peak, num_left, endpoint=False).tolist() #lesion left
        z_right = np.linspace(self.z_peak, self.z_end, num_right, endpoint=True).tolist() #lesion right
        z = z_left + z_right


        #2. Create tree.dat file (unit: mm)
        #num_points is given.
        seg_id = 0
        num_segments = 1

        if tree_path is not None:
            with open(tree_path, "w") as f:
                # Header: total points, total segments
                f.write(f"{num_points} {num_segments}\n")
                # Segment info: segment_id, number_of_points_in_segment
                f.write(f"{seg_id} {num_points}\n")
                
                # Data points: x y z r 0 (last column is always 0)
                for i in range(len(z)):
                    x = 0.0  # Centerline x-coordinate
                    y = float(y_center_lumen(self, z[i])) * factor # mm to cm
                    r = float(radius_lumen(self, z[i])) * factor # mm to cm
                    # Use consistent precision (6 decimal places)
                    f.write(f"{x:.6f} {y:.6f} {z[i]:.6f} {r:.6f} 0\n")

            print("tree.dat is saved to", tree_path)


        #3. create xyzts.dat file
        # e.g.) 188 19 5 5 10 1 100000 0.000001 
        num_segments = 1
        radial_points = 5
        circum_points = 5
        time_step = 5 # frequency
        time_start = 1
        time_stop = 100000
        tol = 0.00001

        if xyzt_path is not None:
            with open(xyzt_path, "w") as f:
                f.write(f"{num_points} {num_segments} {radial_points} {circum_points} {time_step} {time_start} {time_stop} {tol}\n")
                f.write(f"{seg_id} {num_points} -1\n") # -1 is the parent segment
                for i in range(len(z)):
                    x = 0.0  # Centerline x-coordinate
                    y = float(y_center_lumen(self, z[i])) * factor # mm to cm
                    r = float(radius_lumen(self, z[i])) * factor # mm to cm
                    # Use consistent precision (6 decimal places)
                    f.write(f"{x:.6f} {y:.6f} {z[i]:.6f} {r:.6f}\n")
            
            print("xyzts.dat is saved to", xyzt_path)
        
        return


#LumenMeshing for Meshing(Simmetrix)
class LumenMeshing:

    def __init__(self, parent):
        '''
        parent: lumen_main instance
        '''
        #Input path(lumen.stp)
        self.exe_dir = parent.meshing_dir
        self.mesh_complete_dir = parent.mesh_complete_dir
        self.mesh_surfaces_dir = os.path.join(self.mesh_complete_dir, "mesh-surfaces")
        
        #set the path.
        self.stp_path = parent.stp_path
        self.vtk_path = os.path.join(self.exe_dir, "lumen.vtk")
        self.vtp_path = os.path.join(self.exe_dir, "lumen.vtp")

    def validate(self):
        if not os.path.exists(self.mesh_complete_dir):
            raise FileNotFoundError(f"mesh-complete directory not found: {self.mesh_complete_dir}")
        if not os.path.exists(self.mesh_surfaces_dir):
            raise FileNotFoundError(f"mesh-surfaces directory not found: {self.mesh_surfaces_dir}")
    
    #Transform stp file to vtk file.
    @staticmethod
    def _stp_to_vtk(stp_path):

        ''' 
        Convert stp file to vtk file.
        Return the vtk file path.
        '''
        # stp_path (.stp) or (.step)
        if stp_path.endswith(".step"):
            vtk_path = stp_path.replace(".step", ".vtk")
        elif stp_path.endswith(".stp"):
            vtk_path = stp_path.replace(".stp", ".vtk")
        else:
            raise ValueError(f"Invalid stp file: {stp_path}")

        import gmsh
        import meshio

        gmsh.initialize()
        gmsh.model.add("stp_model")
        gmsh.model.occ.importShapes(stp_path)
        gmsh.model.occ.synchronize()

        surface_dims_tags = gmsh.model.getEntities(2)
        surface_centers = []
        for surface in surface_dims_tags:
            com = gmsh.model.occ.getCenterOfMass(surface[0], abs(surface[1]))
            surface_centers.append((surface, com[2]))
        surface_centers.sort(key=lambda x: x[1])

        inlet_dim_tag = surface_centers[0][0]
        wall_dim_tag = surface_centers[1][0]
        outlet_dim_tag = surface_centers[2][0]

        wall_tag = wall_dim_tag[1]
        inlet_tag = inlet_dim_tag[1]
        outlet_tag = outlet_dim_tag[1]

        gmsh.model.addPhysicalGroup(2, [wall_tag], tag=1, name="wall")
        gmsh.model.addPhysicalGroup(2, [inlet_tag], tag=2, name="inlet")
        gmsh.model.addPhysicalGroup(2, [outlet_tag], tag=3, name="outlet")

        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.1)
        gmsh.model.mesh.generate(2)
        gmsh.model.occ.synchronize()

        msh_path = stp_path.replace(".stp", ".msh")
        gmsh.write(msh_path)
        gmsh.finalize()

        mesh = meshio.read(msh_path)
        mesh.points *= 0.1

        tri_blocks = []
        tri_phys_tags = []
        for i, cell_block in enumerate(mesh.cells):
            if cell_block.type == "triangle":
                tri_blocks.append(cell_block.data)
                if "gmsh:physical" in mesh.cell_data:
                    tri_phys_tags.append(mesh.cell_data["gmsh:physical"][i])
                else:
                    raise RuntimeError("No gmsh:physical tags found in the .msh file.")

        if not tri_blocks:
            raise RuntimeError("No triangle cells found in the .msh file (did you generate 2D mesh?).")

        import numpy as np
        fluid_cells = np.vstack(tri_blocks)
        face_ids = np.concatenate(tri_phys_tags).astype(np.int32)

        new_mesh = meshio.Mesh(points=mesh.points, cells=[("triangle", fluid_cells)], cell_data={"ModelFaceID": [face_ids]})
        
        #vtk file path
        meshio.write(vtk_path, new_mesh, file_format="vtk42")

        return vtk_path

    #Transform vtk file to vtp file.
    @staticmethod
    def _vtk_to_vtp(vtk_path: str):

        '''
        Convert vtk file to vtp file.
        Return the vtp file path.
        '''

        import vtk
        # vtk_path (.vtk)
        if vtk_path.endswith(".vtk"):
            vtp_path = vtk_path.replace(".vtk", ".vtp")
        else:
            raise ValueError(f"Invalid vtk file: {vtk_path}")

        ds_reader = vtk.vtkDataSetReader()
        ds_reader.SetFileName(vtk_path)
        ds_reader.ReadAllScalarsOn()
        ds_reader.ReadAllVectorsOn()
        ds_reader.Update()
        data = ds_reader.GetOutput()

        if isinstance(data, vtk.vtkPolyData):
            poly = data
        else:
            surf = vtk.vtkDataSetSurfaceFilter()
            surf.SetInputData(data)
            surf.Update()
            poly = surf.GetOutput()

        # Debug: 확인용 – CellData 배열들 출력
        cdata = poly.GetCellData()
        arr_names = [cdata.GetArrayName(i) for i in range(cdata.GetNumberOfArrays())]
        print("[vtk_to_vtp] CellData arrays:", arr_names)

        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(vtp_path)
        writer.SetInputData(poly)
        writer.SetDataModeToBinary()
        ok = writer.Write()
        if ok == 0:
            raise IOError(f"Failed to write VTP to: {vtp_path}")
        
        return vtp_path
    
    #Assemble the mesh and tree.
    @staticmethod
    def _assemble_mesh_tree(exe_dir: str):
        '''
            Go inside the exec_dir and assemble the mesh and tree.
            and get out of the exec_dir.
        '''
        cwd = os.getcwd()
        try:
            os.chdir(exe_dir)

            # Expected outputs from mesher
            must_exist = [
                "surface_with_id_1.vtp",
                "surface_with_id_2.vtp",
                "surface_with_id_3.vtp",
                "mesh-complete.mesh.vtu",
                "remeshed_model.vtp",
            ]
            for p in must_exist:
                if not os.path.exists(p):
                    raise FileNotFoundError(f"Expected meshing artifact missing: {p}")

            os.rename("surface_with_id_1.vtp", "wall.vtp")
            os.rename("surface_with_id_2.vtp", "inlet.vtp")
            os.rename("surface_with_id_3.vtp", "outlet.vtp")
            os.rename("remeshed_model.vtp", "mesh-complete.exterior.vtp")

            shutil.copy("wall.vtp", "walls_combined.vtp")

            os.makedirs("mesh-complete", exist_ok=True)
            os.makedirs("mesh-surfaces", exist_ok=True)

            shutil.move("wall.vtp", "mesh-surfaces/wall.vtp")
            shutil.move("inlet.vtp", "mesh-surfaces/inlet.vtp")
            shutil.move("outlet.vtp", "mesh-surfaces/outlet.vtp")

            shutil.move("mesh-complete.mesh.vtu", "mesh-complete/mesh-complete.mesh.vtu")
            shutil.move("mesh-complete.exterior.vtp", "mesh-complete/mesh-complete.exterior.vtp")
            shutil.move("walls_combined.vtp", "mesh-complete/walls_combined.vtp")

            shutil.move("mesh-surfaces", "mesh-complete/mesh-surfaces")
        finally:
            os.chdir(cwd)



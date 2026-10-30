import gmsh
import meshio
import numpy as np
import os
import multiprocessing
import time
import utils.utils_geo_mesh.utils_gmsh as utils
import pandas as pd
import json

'''
updated contents by 1113 jeff
- Deleted the optimization after and while meshing for time saving.
- Changed the mesh size to 0.05.
- Changed the algorithm to 10. HXT try this.
- Linear interpolation for high order mesh.
- No Laplacian smoothing.
- There are two types exists
    - type 1: no cal just only lipid core.
    - type 2: cal inside the lipid core.
        - This is trikcy, so we propagte the mesh inside the lipid core and then, smooth the mesh.
          after that, 

Let us write down the code for the type 1 first.
'''

def gmshing_solid(lumen_stp_path, solid_stp_path, lipid_stp_path, fc_stp_path, save_folder_path, nproc = 5, 
                  Terminal_display = True, save_vtu = False, mesh_size = 0.04, lesion_length = 1.0,
                  algo2d = 5, algo3d = 10):

    start_time = time.time()

    gmsh.initialize()
    gmsh.model.add("Stenosis Model")

    #Terminal display setting
    if Terminal_display:   
        gmsh.option.setNumber("General.Terminal", 1)   
    else:
        gmsh.option.setNumber("General.Verbosity", 0) 


    #STEP1: lipid n fc -> new fc
    lipid = gmsh.model.occ.importShapes(str(lipid_stp_path))[0]   #[(3,1)]
    gmsh.model.occ.synchronize()

    fc    = gmsh.model.occ.importShapes(str(fc_stp_path))[0]     #[(3,2)]
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
    lumen = gmsh.model.occ.importShapes(str(lumen_stp_path))[0] #(3,4)
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
    solid = gmsh.model.occ.importShapes(str(solid_stp_path))[0]
    gmsh.model.occ.synchronize()

    gmsh.model.occ.cut([solid], [lipid, fc, lumen], removeObject = False, removeTool = False)
    gmsh.model.occ.synchronize()

    gmsh.model.occ.removeAllDuplicates()
    gmsh.model.occ.synchronize()

    solid = (3,6)

    volumes = set(gmsh.model.getEntities(3))


    solid = volumes - set([lipid, fc, lumen])
    
    if len(solid) != 1:
        raise Exception("STEP4 Failed.")
    else:
        print("STEP4 complete, solid is in the volumes")
        solid = solid.pop() # solid - (3,6)



    ############################################
    ############# PHYSICAL TAGGING #############
    ############################################
    gmsh.model.addPhysicalGroup(3, [solid[1]], tag=1, name="solid")
    gmsh.model.addPhysicalGroup(3, [lipid[1]], tag=2, name="lipid")
    gmsh.model.addPhysicalGroup(3, [fc[1]], tag=3, name="fc")

    # Extract volume tags from lumen
    lumen_volume_tags = [lumen[1]]

    # Get the boundary surfaces between solid and fc fc = (3,7)
    fc_surfaces    = [abs(s[1]) for s in gmsh.model.getBoundary([fc],    oriented=False)]
    solid_surfaces = [abs(s[1]) for s in gmsh.model.getBoundary([solid], oriented=False)]
    lipid_surfaces = [abs(s[1]) for s in gmsh.model.getBoundary([lipid], oriented=False)]

    wall_in_fc_tags    = utils.find_wall_surfaces(fc_surfaces,    lumen_volume_tags)
    wall_in_solid_tags = utils.find_wall_surfaces(solid_surfaces, lumen_volume_tags)  
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
    physical_groups = gmsh.model.getPhysicalGroups(3)
    for (dim, tag) in physical_groups:
        if tag not in [1,2,3]:
            raise Exception(f"(dim = 3)Physical group {tag} is not in the list")

    for (dim, tag) in gmsh.model.getPhysicalGroups(2):
        if tag not in [4,5,6,7]:
            raise Exception(f"(dim = 2)Physical group {tag} is not in the list")
   

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
    utils.slash_lines()
    gmsh.option.setNumber("General.NumThreads", nproc)  # Leave some cores free for system
    gmsh.option.setNumber("Mesh.MaxNumThreads2D", nproc)  # Leave some cores free for system
    gmsh.option.setNumber("Mesh.MaxNumThreads3D", nproc)  # Use 12 threads for 3D meshing
    #gmsh.option.setNumber("Mesh.HighOrderOptimize", 1)  # No optimization while meshing.
    gmsh.model.occ.synchronize()

    # ---- Distance-field sizing from the fibrous-cap (fc) surfaces ----
    # fc_surfaces was computed above from getBoundary([fc]); refine near fc,
    # coarsen away from it. Replaces the old lesion-box field.
    z_lesion = abs(lesion_length) / 2 * 10  # cm -> mm (kept for reference)

    dist_field = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(dist_field, "SurfacesList", fc_surfaces)
    gmsh.model.mesh.field.setNumber(dist_field, "Sampling", 1000)

    thresh_field = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(thresh_field, "InField", dist_field)
    gmsh.model.mesh.field.setNumber(thresh_field, "SizeMin", mesh_size)   # fine near fc
    gmsh.model.mesh.field.setNumber(thresh_field, "SizeMax", 0.2)         # coarse far from fc
    gmsh.model.mesh.field.setNumber(thresh_field, "DistMin", 0.0)
    gmsh.model.mesh.field.setNumber(thresh_field, "DistMax", 3.0)
    gmsh.model.mesh.field.setNumber(thresh_field, "Sigmoid", 0)
    gmsh.model.mesh.field.setAsBackgroundMesh(thresh_field)

    gmsh.option.setNumber("Mesh.MeshSizeMin", mesh_size)
    gmsh.option.setNumber("Mesh.MeshSizeMax", 0.2)

    gmsh.option.setNumber("Mesh.ElementOrder"       , 2)   # quadratic - 2 : Linear - 1
    gmsh.option.setNumber("Mesh.SecondOrderLinear"  , 1)   # Linear interpolation for high order mesh.
    gmsh.option.setNumber("Mesh.Algorithm"          , algo2d)   # 2D Delaunay 5, Frontal-Delaunay 6
    gmsh.option.setNumber("Mesh.Algorithm3D"        , algo3d)   # 3D Delaunay 1, HXT 10
    gmsh.option.setNumber("Mesh.Smoothing"          , 10)  # Laplacian smoothing (positive Jacobian)
    gmsh.model.occ.synchronize()

    #Generate mesh
    gmsh.model.mesh.generate(3)

    #Post meshing optimization (high-order node untangling).
    gmsh.model.mesh.optimize('HighOrder')
    #gmsh.model.mesh.optimize("Netgen")
    utils.check_mesh_quality(0)

    # per-element minSICN quality + save quality.json
    try:
        tet_tags = gmsh.model.mesh.getElementsByType(11)[0]
        q_arr = np.array(gmsh.model.mesh.getElementQualities(tet_tags, qualityName="minSICN"))
        quality_stats = {
            "num_elements": int(len(q_arr)),
            "min_quality": float(np.min(q_arr)),
            "avg_quality": float(np.mean(q_arr)),
            "neg_jacobian_count": int(np.sum(q_arr < 0)),
            "poor_count": int(np.sum(q_arr < 0.3)),
        }
        print(f"Mesh quality: min={quality_stats['min_quality']:.4f}, avg={quality_stats['avg_quality']:.4f}, "
              f"neg_jacobian={quality_stats['neg_jacobian_count']}, poor(<0.3)={quality_stats['poor_count']}, "
              f"num_elements={quality_stats['num_elements']}")
        with open(os.path.join(save_folder_path, "quality.json"), "w") as f:
            json.dump(quality_stats, f, indent=2)
    except Exception as e:
        print(f"Warning: Could not extract mesh quality: {e}")

    #save as mesh
    gmsh_save_path = os.path.join(save_folder_path, f"solid_type_1.msh") #unscaled solid file.
    gmsh.write(gmsh_save_path)
    gmsh.finalize()

    utils.get_mesh_info(gmsh_save_path) # .msh file info by meshio
    print("!!!Gmsh mesh completely saved on:", gmsh_save_path)
    print(f"\nTime taken for gmshing: {time.time() - start_time} seconds\n")

    return gmsh_save_path



if __name__ == "__main__":

    non_meshed_list = []
    from pathlib import Path

    # Create error log file if it doesn't exist
    current_dir = Path(__file__).parent
    error_log_path = f"Gmsh_errors_type1.txt"
    if not os.path.exists(error_log_path):
        open(error_log_path, 'w').close()

    #read csv file. updated by 1009 jeff
    csv_path = current_dir / "pre_data" / "parameter.csv"
    df = pd.read_csv(str(csv_path))

    for i in [0,4]:
        try:
            solid_data_dir = current_dir / "solid_data" / f"case_{i}"
            lumen_path = solid_data_dir / "lumen.stp"
            solid_path = solid_data_dir / "solid.stp"
            lipid_path = solid_data_dir / "lipid.stp"
            fc_path = solid_data_dir / "fc.stp"

            # # if msh already exists, skip the meshing
            # msh_path = solid_data_dir / f"solid_Linear_{i}.msh"
            # if msh_path.exists():
            #     print(f"msh already exists for case {i}, skipping the meshing")
            #     #continue

            print(f"\n\n************Processing case {i}************\n")
            # Start a timer to check if meshing takes too long
            start = time.time()

            # Run meshing with timeout and check return value
            timeout_seconds = 300 # 5mins
            mesh_size = 0.055 # 50million, 
            save_vtu = True
            nproc = 30
            lesion_length = df.loc[i, 'lesion_length'] # read lesion length from the csv file.

            mesh_process = multiprocessing.Process(
            target=gmshing_solid,
            args=(lumen_path, solid_path, lipid_path, fc_path, solid_data_dir, 
                  nproc, True, save_vtu, mesh_size, lesion_length)
            )
            
            mesh_process.start()
            mesh_process.join(timeout=timeout_seconds)
            if mesh_process.is_alive():
                error_msg = f"Meshing took longer than {timeout_seconds} seconds timeout"
                with open(error_log_path, "a") as f:
                    f.write(f"Case {i}: Timeout Error - {error_msg}\n\n")
                raise TimeoutError(error_msg)
            print(f"Meshing took {time.time() - start} seconds")
                
        except Exception as e:
            print(f"Error in case {i}: {e}")
            print(f"Skipping case {i} and continuing with next case...\n")
            gmsh.finalize()
            non_meshed_list.append(i)

            with open(error_log_path, "a") as f:
                f.write(f"Case {i}: {type(e).__name__} - {str(e)}\n\n")
            continue
    
    print(f"Non-meshed cases: {non_meshed_list}")
    print(f"Total non-meshed cases: {len(non_meshed_list)}")



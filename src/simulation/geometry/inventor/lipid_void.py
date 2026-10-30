import math
import numpy as np
import os
from .utils_CAD import *
'''
so when the lipid CAD operaion filaed -> error: void
then use this function instead of the previous lipid one.
'''


def lipid_void_step(model, save_folder_path):
    '''
       CAD lipid core operatino for void lipid.
    '''

    #Autodesk inventor - New part document
    lipid_doc = model.inv.Documents.Add(12290, model.inv.FileManager.GetTemplateFile(12290, 8962))
    lipid_doc.DisplayName = "Lipid"
    lipid_doc.UnitsOfMeasure.LengthUnits = 11268 #11269 -> mm, 11268 -> cm  
    model.part_def = lipid_doc.ComponentDefinition

    model.sketches3d_col = model.part_def.Sketches3D  #set of sketches.
    model.tg = model.inv.TransientGeometry #set of points, lines, circles, etc.

    sections = model.inv.TransientObjects.CreateObjectCollection() #object collection for lofting.
    centerline_sketch = model.sketches3d_col.Add() #sketch3D to contain 2 points.
    centerline_sketch.Name = "points"

    #calculate z_lipid_L and z_lipid_R
    model.z_lipid_L, model.z_lipid_R = find_ab(model) # Function in the utils.CAD


    ###################################################################
    ############  Profiles and 2 points back and forth. ###############
    ###################################################################

    #Define point1 (z = z_lipid_L)
    y1 = y_center_lumen(model, model.z_lipid_L) + radius_lumen(model, model.z_lipid_L)
    point1 = model.tg.CreatePoint(0, y1, model.z_lipid_L)
    point1_sk = centerline_sketch.SketchPoints3D.Add(point1)
    sections.Add(point1_sk)
    
    #Profiles for the left and right side.
    z_profile_L = np.linspace(model.z_lipid_L, model.z_peak, 5, endpoint=False)[1:]
    z_profile_R = np.linspace(model.z_peak, model.z_lipid_R, 6, endpoint=False)
    z_profile = np.concatenate([z_profile_L, z_profile_R])

    xy_plane = model.part_def.WorkPlanes.Item("XY Plane")
    

    for z in z_profile:

        #Define a working plane.
        pt = model.tg.CreatePoint(0, 0, z)
        pt_sk = centerline_sketch.SketchPoints3D.Add(pt)
        wp = model.part_def.WorkPlanes.AddByPlaneAndPoint(xy_plane, pt_sk, False)
        wp.Visible = True

        #Draw a profile on the 2D sketch.
        sketch2d = model.part_def.Sketches.Add(wp)
        wp.Visible = False
        sketch2d.Visible = True

        #Parameters for the external arc.
        r_lipid = lipid_ex_y_coordinate(model, z) - y_center_lesion(model,z) # lipid_area radius
        θ = alpha_theta(model, z)
        β = beta(model, θ, z)

        #Let us Draw total 3 arcs and 2 lines.
        x_β, y_β = r_lipid * math.sin(β), r_lipid * math.cos(β) + y_center_lesion(model,z)
        x_θ, y_θ = r_lipid * math.sin(θ), r_lipid * math.cos(θ) + y_center_lesion(model,z)


        #Points for 3 arcs and 2 lines.
        center_lesion = model.tg.CreatePoint2d(0, y_center_lesion(model,z))
        start = model.tg.CreatePoint2d(x_β, y_β)  
        intermediate1 = model.tg.CreatePoint2d(x_θ, y_θ)
        intermediate2 = model.tg.CreatePoint2d(-x_θ,y_θ)
        end = model.tg.CreatePoint2d(-x_β, y_β)
        center_lumen = model.tg.CreatePoint2d(0, y_center_lumen(model,z))

        
        #DRAW POINTS ON THE SKETCH.
        center_lesion_sk = sketch2d.SketchPoints.Add(center_lesion)
        start_sk = sketch2d.SketchPoints.Add(start)
        intermediate1_sk = sketch2d.SketchPoints.Add(intermediate1)
        intermediate2_sk = sketch2d.SketchPoints.Add(intermediate2)
        end_sk = sketch2d.SketchPoints.Add(end)
        center_lumen_sk = sketch2d.SketchPoints.Add(center_lumen)


        #Draw An External Arc and 2 lines.
        line1 = sketch2d.SketchLines.AddByTwoPoints(center_lumen_sk, start_sk)
        arc1 = sketch2d.SketchArcs.AddByCenterStartEndPoint(center_lesion_sk, start_sk, intermediate1_sk)
        arc2 = sketch2d.SketchArcs.AddByCenterStartEndPoint(center_lesion_sk, intermediate1_sk, intermediate2_sk)
        arc3 = sketch2d.SketchArcs.AddByCenterStartEndPoint(center_lesion_sk, intermediate2_sk, end_sk)
        line2 = sketch2d.SketchLines.AddByTwoPoints(center_lumen_sk, end_sk)


        #Safe fillet radius
        safe_ratio = 0.5
        fillet_radius = calculate_safe_fillet_radius(model, z, safe_ratio)


        try:
            sketch2d.SketchArcs.AddByFillet(
                arc1, line1, fillet_radius,
                arc1.EndSketchPoint.Geometry,
                line1.StartSketchPoint.Geometry
            )

        except Exception as e:
            import traceback
            print("Fillet generation failed for the lipid profile:", e)
            traceback.print_exc()


        try:
            sketch2d.SketchArcs.AddByFillet(
                arc3, line2, fillet_radius,
                arc3.StartSketchPoint.Geometry,
                line2.StartSketchPoint.Geometry
            )

        except Exception as e:
            import traceback
            print("Fillet generation failed for the lipid profile:", e)
            traceback.print_exc()
    

        profile = sketch2d.Profiles.AddForSolid()
        sections.Add(profile)



    #Define Last point2 (z = z_lipid_R)
    y2 = y_center_lumen(model, model.z_lipid_R) + radius_lumen(model, model.z_lipid_R)
    point2 = model.tg.CreatePoint(0, y2, model.z_lipid_R)
    point2_sk = centerline_sketch.SketchPoints3D.Add(point2)
    sections.Add(point2_sk)


    ##############################################
    ################# Lofting ####################
    ##############################################

    #define lofting OBJECT.
    loftFeatures = model.part_def.Features.LoftFeatures
    loftDef = loftFeatures.CreateLoftDefinition(sections, 20485) #20485-> body, 20484-> surface

    #define planes for the tangential lofting.
    # tangent_plane1 = model.part_def.WorkPlanes.AddByPlaneAndPoint(xy_plane, point1_sk, False)
    # tanget_plane3 = model.part_def.WorkPlanes.AddByPlaneAndPoint(xy_plane, point2_sk, False)
    # tangent_plane1.Visible = True
    # tanget_plane3.Visible = True

    #First and last point conditions.(use 34309 -> sharp point, 34306 -> tangent point, 34310 -> tangent plane)
    loftDef.FirstSectionCondition = 34306
    loftDef.LastSectionCondition = 34306
    loftDef.FirstSectionImpact = 2.0
    loftDef.LastSectionImpact = 2.0
    # loftDef.FirstSectionTangentPlane = tangent_plane1
    # loftDef.LastSectionTangentPlane = tanget_plane3
    
    #Merge tangent faces option.
    loftDef.MergeTangentFaces = True

    #lofting execute.
    try:
        loft_object = loftFeatures.Add(loftDef) #Lofting execute
        loft_object.Name = "lipid"
    except Exception as e:
        print("Lofting failed with Tangental points:", e)
        try:
            loftDef.FirstSectionCondition = 34309
            loftDef.LastSectionCondition = 34309
            loft_object = loftFeatures.Add(loftDef) #Lofting execute
            loft_object.Name = "lipid"

        except Exception as e:
            import traceback
            print("Lofting failed:", e)
            traceback.print_exc()
            return None



    ##############################################
    ################# Save files #################
    ##############################################
    step_path = os.path.join(save_folder_path, "lipid.stp")
    ipt_path = os.path.join(save_folder_path, "lipid.ipt")

    lipid_doc.SaveAs(ipt_path, True)
    save_as_step(model, lipid_doc, step_path) #save as step from utils.py
    print(f"✅ Completely save file: {step_path}")
    lipid_doc.Close(model.close_CAD)

    return step_path
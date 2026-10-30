import math
import numpy as np
import os
from .utils_CAD import *


def lumen_step(model, save_folder_path):
    """
    Required parameters

    - r_max : average radius of the vessel.
    - r_min : minimum radius of the vessel(stenosis part).
    - lesion_length : length of the lesion.

    @updated 0720 jeff
    Change the lofting method from Centerline to Guideline.
    """

    #Autodesk inventor - New part document
    fluid_doc = model.inv.Documents.Add(12290, model.inv.FileManager.GetTemplateFile(12290, 8962))
    fluid_doc.DisplayName = "Lumen"
    units = fluid_doc.UnitsOfMeasure
    units.LengthUnits = 11268 #11269 -> mm, 11268 -> cm  
    model.part_def = fluid_doc.ComponentDefinition

    model.sketches3d_col = model.part_def.Sketches3D  #set of sketches.
    model.tg = model.inv.TransientGeometry

    object_collection = model.inv.TransientObjects.CreateObjectCollection() #will collect points. for centerline.
    centerline_sketch = model.sketches3d_col.Add() #add a new sketch includes in the sketch3d collection.
    centerline_sketch.Name = "centerline"

    sections = model.inv.TransientObjects.CreateObjectCollection() # Object collection for profiles for lofting.
    xy_plane = model.part_def.WorkPlanes.Item("XY Plane") # define global xy plane for the centerline.


    #set of z coordinates for Lofting.
    z_front = np.linspace(model.z_start, - model.z_lesion, 4, endpoint=False)
    z_lesion_left = np.linspace(-model.z_lesion, model.z_peak, 2, endpoint=False) #lesion left
    z_lesion_right = np.linspace(model.z_peak, model.z_lesion, 2, endpoint=False) #lesion right
    z_back = np.linspace(model.z_lesion, model.z_end, 8, endpoint = True)


    z_centerline = np.concatenate([z_front, z_lesion_left, z_lesion_right, z_back])

    for z in z_centerline:
        
        y_lumen = y_center_lumen(model, z)
        r_lumen = radius_lumen(model, z)

        point = model.tg.CreatePoint(0,y_lumen,z) #define point
        sketch_point = centerline_sketch.SketchPoints3D.Add(point)
        object_collection.Add(sketch_point)

        wp = model.part_def.WorkPlanes.AddByPlaneAndPoint(xy_plane, sketch_point, False)
        wp.Visible = True

        #Draw circle on the work plane
        sketch2d = model.part_def.Sketches.Add(wp) #add sketch on the work plane.
        center_point = model.tg.CreatePoint2d(0, 0)
        sketch2d.SketchCircles.AddByCenterRadius(center_point, r_lumen)


        #add profile doesn't matter whether it is normal or abnormal.
        profile = sketch2d.Profiles.AddForSolid()
        sections.Add(profile)
        wp.Visible = False
        sketch2d.Visible = True


    #Draw centerline
    centerline_sketch.SketchSplines3D.Add(object_collection, 26370)

    ##############################################
    ################# Lofting ####################
    ##############################################
    try:
        loftFeatures = model.part_def.Features.LoftFeatures
        loftDef = loftFeatures.CreateLoftDefinition(sections, 20485) #20485-> body, 20484-> surface
        profile_centerline = centerline_sketch.Profiles3D.AddOpen()
        loftDef.Centerline = profile_centerline
        loft_object = loftFeatures.Add(loftDef) #Lofting execute
        loft_object.Name = "lumen"
        #print(f"✅ {loft_object.Name} Lofting completed")
    except Exception as e:
        print(f"Error: {e}")
        fluid_doc.Close(True)
        return None



    ##############################################
    ################# Save files #################
    ##############################################

    #save file on the save_folder_path(input)
    step_path = os.path.join(save_folder_path, "lumen.stp")
    ipt_path = os.path.join(save_folder_path, "lumen.ipt")

    fluid_doc.SaveAs(ipt_path, True)
    #print(f"✅ completely save ipt file as: {ipt_path}")
    save_as_step(model, fluid_doc, step_path) #save as step from utils.py
    print(f"✅ Completely save file: {step_path}")

    #close the document, True -> save the file.
    fluid_doc.Close(model.close_CAD)

    return step_path

    

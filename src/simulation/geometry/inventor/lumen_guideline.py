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
    object_collection_guideline1 = model.inv.TransientObjects.CreateObjectCollection() #will collect points. for centerline.
    object_collection_guideline2 = model.inv.TransientObjects.CreateObjectCollection() #will collect points. for centerline.

    #define 3D sketch for the two guidelines and a centerline.
    guideline1_sketch = model.sketches3d_col.Add() #add a new sketch includes in the sketch3d collection.
    guideline1_sketch.Name = "Guideline1"

    guideline2_sketch = model.sketches3d_col.Add() #add a new sketch includes in the sketch3d collection.
    guideline2_sketch.Name = "Guideline2"

    centerline_sketch = model.sketches3d_col.Add() #add a new sketch includes in the sketch3d collection.
    centerline_sketch.Name = "centerline"

    sections = model.inv.TransientObjects.CreateObjectCollection() # Object collection for profiles for lofting.
    xy_plane = model.part_def.WorkPlanes.Item("XY Plane") # define global xy plane for the centerline.


    #set of z coordinates for Lofting.
    z_front = create_dense_points(model.z_start, - model.z_lesion, 5, dense_end=True)
    z_lesion_left = np.linspace(-model.z_lesion, model.z_peak, 10, endpoint=False).tolist()[1:] #lesion left
    z_lesion_right = np.linspace(model.z_peak, model.z_lesion, 11, endpoint=False).tolist() #lesion right
    z_back = create_dense_points(model.z_lesion, model.z_end, 10, dense_end=False)


    z_guideline = z_front + z_lesion_left + z_lesion_right + z_back # zcoordinate set for the guideline.
    z_profile = [model.z_start, - model.z_lesion,  model.z_peak,  model.z_lesion,  model.z_end] # z coordinate set for the profile.


    for z in z_guideline:
        
        y = y_center_lumen(model, z)
        r_lumen = radius_lumen(model, z)

        point1 = model.tg.CreatePoint(0,y+r_lumen,z) #define point
        point2 = model.tg.CreatePoint(0,y-r_lumen,z) #define point

        sketch_point1 = guideline1_sketch.SketchPoints3D.Add(point1) #add point to the defined sketch
        sketch_point2 = guideline2_sketch.SketchPoints3D.Add(point2) #add point to the defined sketch

        object_collection_guideline1.Add(sketch_point1) #add point to the collection of objects.
        object_collection_guideline2.Add(sketch_point2) #add point to the collection of objects.

        if z in z_profile:

            #parallel to the xy plane and pass the point.
            center_point = model.tg.CreatePoint(0,y,z)
            sketch_point = centerline_sketch.SketchPoints3D.Add(center_point)
            object_collection.Add(sketch_point)

            
            wp = model.part_def.WorkPlanes.AddByPlaneAndPoint(xy_plane, sketch_point, False)
            wp.Visible = True

            #Draw circle on the work plane
            sketch2d = model.part_def.Sketches.Add(wp) #add sketch on the work plane.
            center_point = model.tg.CreatePoint2d(0, 0)

            sketch2d.SketchCircles.AddByCenterRadius(center_point, r_lumen)

            #add profile for the lofting.
            profile = sketch2d.Profiles.AddForSolid()
            sections.Add(profile)

            wp.Visible = False
            sketch2d.Visible = True



    # Create a 3D spline -> Guideline1 for the lofting.
    guideline1_sketch.SketchSplines3D.Add(object_collection_guideline1, 26370) #26371 Autocad, 26370 Minimum E method, 26369 the Centripetal Method

    guideline2_sketch.SketchSplines3D.Add(object_collection_guideline2, 26370) #26371 Autocad, 26370 Minimum E method, 26369 the Centripetal Method



    ##############################################
    ################# Lofting ####################
    ##############################################
    loftFeatures = model.part_def.Features.LoftFeatures
    loftDef = loftFeatures.CreateLoftDefinition(sections, 20485) #20485-> body, 20484-> surface
 

    profile_guideline1 = guideline1_sketch.Profiles3D.AddOpen()
    profile_guideline2 = guideline2_sketch.Profiles3D.AddOpen()

    loftDef.LoftRails.Add(profile_guideline1)
    loftDef.LoftRails.Add(profile_guideline2)


    # profile_centerline = centerline_sketch.Profiles3D.AddOpen()
    # loftDef.Centerline = profile_centerline

    loft_object = loftFeatures.Add(loftDef) #Lofting execute
    loft_object.Name = "lumen"
    #print(f"✅ {loft_object.Name} Lofting completed")




    ##############################################
    ################# Save files #################
    ##############################################

    #save file on the save_folder_path(input)
    step_path = os.path.join(save_folder_path, "lumen.stp")
    ipt_path = os.path.join(save_folder_path, "lumen.ipt")

    fluid_doc.SaveAs(ipt_path, True)
    #print(f"✅ completely save ipt file as: {ipt_path}")
    save_as_step(model, fluid_doc, step_path) #save as step from utils.py
    print(f"Saved file: {step_path}")
  
    #close the document, True -> save the file.
    fluid_doc.Close(model.close_CAD)
   
    return ipt_path

    

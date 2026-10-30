import os
from .utils_CAD import save_as_step
'''
New way just, thickness the lumen and save the file.
'''

#Procedure1: Offset the fc.stp on AutodeskInventor.
def AUTODESK_Offset(model, tool_ipt_path, offset, save_folder_path, save_obj_name):

    '''
    Offset from the fc.ipt
    return the offset_ipt_path
    '''

    #Offset1 open the fc file.
    offset_doc = model.inv.Documents.Open(tool_ipt_path)  
    part_def = offset_doc.ComponentDefinition
    surface_bodies = part_def.SurfaceBodies.Item(1)
    offset_doc.UnitsOfMeasure.LengthUnits = 11268 #11269 -> mm, 11268 -> cm 
    #print("surface_bodies.Faces.Count:", surface_bodies.Faces.Count)

    for face in surface_bodies.Faces: 
        #print("face.SurfaceType:", face.SurfaceType)
        # 5890 - plane surface(two caps), 5897 Bspline surface(wall that we wanna thicken)
        if face.SurfaceType != 5890:
            face_object = face
    
    # Prepare the Feature Definition
    face_collection = model.inv.TransientObjects.CreateObjectCollection()
    face_collection.Add(face_object) #collection of faces to offset.

    # Get the ThickenFeatures collection object
    thicken_features = part_def.Features.ThickenFeatures

    try: 
        thicken_features.Add(
            face_collection,
            offset,#distance(unit:cm)
            20993, # Extend direction
            20481,  # Join operation
            AutomaticBlending = True
        )
    except Exception as e:
        print(f"Error offset: {e}")
        offset_doc.Close(True)
        raise RuntimeError(f"Error offset: {e}")

    #save only as ipt file.
    step_path = os.path.join(save_folder_path, f"{save_obj_name}.stp")
    ipt_path = os.path.join(save_folder_path, f"{save_obj_name}.ipt")
    offset_doc.SaveAs(ipt_path, True)
    save_as_step(model, offset_doc, step_path) #save as step from utils.py
    print(f"Saved file: {step_path}")
    offset_doc.Close(model.close_CAD)

    return step_path, ipt_path

def fc_step(model, save_folder_path, lumen_ipt_path):
    '''
        Main Function: Create the subdomain of the calcification.
        1. Offset (d_fc_ca) from the fc.ipt
        2. Read the lipid.ipt
        3. Boolean operation between 1,2
        4. Save the file as ipt and stp

        return the ca_stp_path
    '''
    offset = model.fc_av_th
    fc_step_path, _ = AUTODESK_Offset(model, lumen_ipt_path, offset, save_folder_path, "fc")
    return fc_step_path

def fc_offset_step(model, save_folder_path, lumen_ipt_path):
    '''
    model: VesselModel instance
    save_folder_path: save folder path
    lumen_ipt_path: lumen ipt path
    '''

    offset = model.d_fc_ca + model.fc_av_th
    fc_offset_step_path, _ = AUTODESK_Offset(model, lumen_ipt_path, offset, save_folder_path, "fc_offset")
    return fc_offset_step_path

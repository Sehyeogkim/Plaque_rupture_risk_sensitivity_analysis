'''
read the ipt and save to stp file.
'''
import os

def ipt_to_stp(model, ipt_path, stp_path, close_ipt: bool = True):
    
    #open the ipt file.
    offset_doc = model.inv.Documents.Open(ipt_path)  

    #save the stp file.
    # Inventor COM signature: SaveAs(FileName, SaveCopyAs)
    offset_doc.SaveAs(stp_path, True)
    offset_doc.Close(close_ipt)
    return stp_path

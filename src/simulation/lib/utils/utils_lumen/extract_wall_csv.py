import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pyvista as pv

def _find_pressure_key(point_data_keys):
    """check the name of the pressure array and data type"""
    
    candidates = [
        "Pressure",
        "pressure",
        "P",
        "p",
    ]

    for k in point_data_keys:
        if k in candidates:
            return k

    raise KeyError("Could not find a pressure point-data array (expected one of: Pressure, pressure, P, p).")


def _find_inplane_traction_key(point_data):
    """check the name of the inplane traction vector and data type"""
    
    candidates = [
        "r_inplane_traction",
        "rinplane_traction",
        "inplane_traction",
        "traction_inplane",
    ]

    for name in candidates:
        if name in point_data and point_data[name].ndim == 2 and point_data[name].shape[1] == 3:
            return name
    
    raise KeyError("Could not find a 3-component in-plane traction point-data array.")


def _filter_wall_by_face_id(mesh: pv.PolyData, exclude_face_ids = [2,3]):
    
    """
    Remove inlet/outlet caps by ModelFaceID. If exclude_face_ids is None and keep_largest_if_missing=True,
    keep only face IDs that collectively cover the largest areas (auto cap removal).
    """

    if 'ModelFaceID' not in mesh.cell_data.keys():
        raise KeyError("The input VTP must have cell_data['ModelFaceID'].")

    face_ids = mesh.cell_data["ModelFaceID"].astype(int)
    exclude_face_ids = set(int(x) for x in exclude_face_ids)
    mask = ~np.isin(face_ids, list(exclude_face_ids))
    
    # Work around PyVista versions where extract_cells may fail due to
    # missing private metadata on the input dataset by deep-copying first.
    mesh_copy = mesh.copy(deep=True)
    kept_cell_ids = np.nonzero(mask)[0].astype(np.int64)
    submesh = mesh_copy.extract_cells(kept_cell_ids)
    return submesh.clean()


def _prepare_normals(mesh: pv.PolyData) -> pv.PolyData:
    """
    Compute per-cell and per-point normals.
    Save cell normals as cell_data['normal'] and point normals as point_data['normal'].
    """
    
    # compute_normals returns a new mesh unless inplace=True (we want a returned copy anyway)
    out = mesh.compute_normals(
        cell_normals=True,
        point_normals=True,
        auto_orient_normals=True,
        consistent_normals=True,
        inplace=False
    )

    
    return out



#Main function to extract the wall data from the vtp file.
def extract_wall_data(vtp_file: str,
                      output_csv: str = None,
                      exclude_face_ids=[2,3]) -> str:
    """
    Process a single wall VTP at one timestep:

    Steps:
      1) Remove inlet/outlet caps via ModelFaceID (id -> 2,3).
      2) Compute per-cell normals -> save as cell_data['normal'].
      3) Ensure point_data has: pressure (scalar), r_inplane traction (3-vector), normal (3-vector).
      4) Compute Total_traction = pressure * normal + r_inplane_traction.
      5) Save CSV with columns: x, y, z, Tx, Ty, Tz.

    Args:
      vtp_file: path to input .vtp (includes caps).
      output_csv: path to output CSV; defaults to <vtp_basename>_total_traction.csv
      exclude_face_ids: list/iterable of ModelFaceID values to exclude (inlets/outlets).
      auto_detect_caps: if True and exclude_face_ids is None, auto-remove likely caps by area.
      pressure_key: explicit pressure array name in point_data (default: auto-detect).
      inplane_traction_key: explicit vector array name for r_inplane traction (default: auto-detect).

    Returns:
      The CSV path written.
    """
    if not os.path.isfile(vtp_file):
        raise FileNotFoundError(f"VTP not found: {vtp_file}")

    mesh0 = pv.read(vtp_file)

    if not isinstance(mesh0, pv.PolyData):
        mesh0 = mesh0.extract_surface().clean()

    # 1) Remove inlet/outlet caps
    wall = _filter_wall_by_face_id(mesh0, exclude_face_ids)
    
    # Ensure wall is PolyData for compute_normals
    if not isinstance(wall, pv.PolyData):
        wall = wall.extract_surface().clean()

    # 2) Compute normals (cell + point), save cell as 'Normals'
    wall = _prepare_normals(wall)

    # 3) check whether vtp has P and r_inplane_traction.
    pressure_key = _find_pressure_key(list(wall.point_data.keys()))
    inplane_traction_key = _find_inplane_traction_key(wall.point_data)

    pressure = np.asarray(wall.point_data[pressure_key]).reshape(-1)
    normal = np.asarray(wall.point_data["Normals"])  # from compute_normals(point_normals=True)
    r_inplane = np.asarray(wall.point_data[inplane_traction_key])

    # 5) Total_traction = pressure * normal + r_inplane_traction
    total_trac = pressure[:, None] * normal + r_inplane
    wall.point_data["Total_traction"] = total_trac

    # 4) Save the vtp file
    wall.save("wall_total_traction.vtp")

    # 6) Save CSV
    if output_csv is None:
        base, _ = os.path.splitext(vtp_file)
        output_csv = f"{base}_total_traction.csv"
    xyz = np.asarray(wall.points)
    df = pd.DataFrame({
        "x": xyz[:, 0],
        "y": xyz[:, 1],
        "z": xyz[:, 2],
        "Tx": total_trac[:, 0],
        "Ty": total_trac[:, 1],
        "Tz": total_trac[:, 2],
    })

    df.to_csv(output_csv, index=False)

    return output_csv

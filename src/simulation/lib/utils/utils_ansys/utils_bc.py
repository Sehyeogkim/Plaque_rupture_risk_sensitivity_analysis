'''
Subfunctions for the Boundary conditions:

-Pressure and traction on the wall_inner surface.

'''

import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
import time

def Interpolate_wall_data(wall_csv_path, node_ids, node_coords):
    '''
    Interpolate the wall data (Only Total traction) according to the node_ids and node_coords.

    INPUT:
    - wall_csv_path: the path of the wall data.
    - node_ids: the node IDS of the wall (SURF154 nodes)
    - node_coords: the coordinates of the wall.

    OUTPUT:
    - interpolated_data: only node_id and traction data.(x,y,z)
    '''
    start_time = time.time()
    df = pd.read_csv(wall_csv_path)

    wall_points = df[['x', 'y', 'z']].values
    traction = df[['Tx', 'Ty', 'Tz']].values

    tree = cKDTree(wall_points)
    distances, nearest_indices = tree.query(node_coords, k=3)

    power = 2
    with np.errstate(divide='ignore'):
        weights = 1.0 / (distances ** power)
    weights[np.isinf(weights)] = 1e12
    norm_weights = weights / np.sum(weights, axis=1, keepdims=True)

    new_tractions = np.sum(norm_weights[..., None] * traction[nearest_indices], axis=1)

    interpolated_data = pd.DataFrame({
        "node_ID": node_ids,
        "Tx": new_tractions[:, 0],
        "Ty": new_tractions[:, 1],
        "Tz": new_tractions[:, 2],
    })

    end_time = time.time()
    print(f"Interpolated wall data created in memory: {end_time - start_time:.2f} seconds  ")
    return interpolated_data

def Apply_Traction(mapdl, interpolated_df, element_array, txt_path="apply_traction.txt"):
    
    df = interpolated_df
    scale_factor = 1.0 # check the direction plz.

    elems, conn = zip(*[(arr[-10], [arr[-8], arr[-7], arr[-6]]) for arr in element_array])
    elems = np.array(elems, dtype=np.int32)
    conn  = np.array(conn, dtype=np.int32)

    node_ids = df["node_ID"].to_numpy(dtype=np.int32)
    tx = (df["Tx"].to_numpy() * scale_factor).astype(float)
    ty = (df["Ty"].to_numpy() * scale_factor).astype(float)
    tz = (df["Tz"].to_numpy() * scale_factor).astype(float)

    node_id_to_idx = {nid: idx for idx, nid in enumerate(node_ids)}
    conn_idx = np.array([[node_id_to_idx[nid] for nid in elem] for elem in conn])
    Tx_e = tx[conn_idx]
    Ty_e = ty[conn_idx]
    Tz_e = tz[conn_idx]

    #averaging the traction data
    Tx_e = np.round(Tx_e.mean(axis=1), decimals=2).astype(float)
    Ty_e = np.round(Ty_e.mean(axis=1), decimals=2).astype(float)
    Tz_e = np.round(Tz_e.mean(axis=1), decimals=2).astype(float)

    with open(txt_path, 'w') as f:
        for e, x, y, z in zip(elems, Tx_e, Ty_e, Tz_e):
            f.write(f"sfe, {e}, 1, PRES, 0, {x}\n")
            f.write(f"sfe, {e}, 2, PRES, 0, {y}\n")
            f.write(f"sfe, {e}, 3, PRES, 0, {z}\n")
    mapdl.input(txt_path)    
    
    return

import sys
sys.path.insert(0, '/home/jeff/repo/master_python/src')

import pandas as pd
import math
from utils.utils_geo_mesh.main_CAD import VesselModel, CAD_instance_from_idx
from utils.utils_geo_mesh.utils_CAD import radius_lesion

csv_path = '/home/jeff/project/55c_final_defense_0512/Figure_final/figure_result1_VI/plaqueburden/input.csv'
df = pd.read_csv(csv_path)

results = []
for i in range(len(df)):
    model = CAD_instance_from_idx(i, csv_path)

    z_MLA = model.z_peak  # axial location of minimum lumen area
    r_lumen = model.r_min  # lumen radius at MLA = (1 - DOS) * r_max
    R_vessel = radius_lesion(model, z_MLA)  # vessel wall radius at MLA

    A_lumen = math.pi * r_lumen**2
    A_vessel = math.pi * R_vessel**2

    PB = (A_vessel - A_lumen) / A_vessel * 100.0

    results.append({'case_id': i, 'PB(%)': round(PB, 4)})

out_df = pd.DataFrame(results)
out_df.to_csv('/home/jeff/project/55c_final_defense_0512/Figure_final/figure_result1_VI/plaqueburden/PB.csv', index=False)
print(out_df.to_string(index=False))
print(f"\nTotal cases: {len(out_df)}")
print(f"PB range: {out_df['PB(%)'].min():.2f}% ~ {out_df['PB(%)'].max():.2f}%")

'''
    Read the result.json file and save the Q(t), P(t) at the inlet and outlet.

    following data will be predefined for the simulation:
    time_step_size = 0.01
    # of time steps = 500

    t_cycle = 1.0
    so the time step per cycle = time_step_size / t_cycle = 100
    num_cycle = # of time steps / time step per cycle = 5

    Since we need to analyze the last cycle, so we just need to analyze the last 5th cycle.

    last_cycle = 5
    start_t_step = (last_cycle - 1) * time_step_per_cycle + 1
    end_t_step = last_cycle * time_step_per_cycle

    REad
    - result_start_t_step
    - result_end_t_step

    and then post process the data
    at the start_pt_id = 1 -> P_inlet(t)
    and the end_pt_id = 49 -> Q_outlet(t)

    there would be (time step per cycle) number of data on each file.

    Read P_inlet(t) and find the max and min time step + save mean(P(t))

    Read Q_outlet(t) and find Q at the above max and min time step + save mean(Q(t))

    Finally, we get

    max time step -> P Q at that time
    min time step -> P Q at that time
    mean P and Q.

'''

import json
import os
import numpy as np


# function to define the start and end time step to analyze cuz we will analyze the last cycle
def start_end_t_step(total_time_step = 500, time_step_size = 0.01, t_cycle = 1.0):
    '''
    Input:
    - total_time_step: total number of time steps
    - time_step_size: time step size
    - t_cycle: cardiac cycle time

    Output:
    - start_t_step: start time step to analyze
    - end_t_step: end time step to analyze
    '''
    # time step per a cycle
    time_step_per_cycle = int(t_cycle / time_step_size)
    # number of cycles for whole simulation
    num_cycle = int(total_time_step / time_step_per_cycle)

    # We will analyze the last cycle
    last_cycle = num_cycle
    start_t_step = (last_cycle - 1) * time_step_per_cycle + 1
    end_t_step = last_cycle * time_step_per_cycle

    return start_t_step, end_t_step

#extraqct data at the specific point
def data_at_sepcific_pt(start_timestep = 1, end_timestep = 500, index = 49, result_folder_path = './result'):
    
    '''
    Input:
    - total_points: total number of points
    - start_timestep: start timestep
    - end_timestep: end timestep
    - index: point index

    Output:
    - Q(t), P(t) data file at the " given specified point index "
    '''

    # Initialize lists to store time series data 
    flow_data = []
    pressure_data = []

    print(f"\nProcessing point index {index}...")
    print(f"Processing from timestep {start_timestep} to {end_timestep}...")

    # Read all result files
    for timestep in range(start_timestep, end_timestep + 1):
        filename = os.path.join(result_folder_path, f"result_{timestep}.json") # result file name - result.

        print(f"Reading {filename}...")
        if not os.path.exists(filename):
            print(f"Warning: {filename} not found, skipping...")
            continue
        
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            
            # Extract data from segments (assuming single segment for now)
            if 'segments' in data and len(data['segments']) > 0:
                segment = data['segments'][0]
                double_point_data = segment['double_point_data']
                
                # Get flow and pressure at the specified point index
                flow_value = double_point_data['flow'][index]
                pressure_value = double_point_data['pressure'][index]
                
                flow_data.append(flow_value)
                pressure_data.append(pressure_value)
                
            else:
                print(f"Warning: No segments found in {filename}")
                
        except Exception as e:
            print(f"Error reading {filename}: {e}")
            continue

    # Save data to .dat files
    if flow_data:
        # Save flow data
        Q_save_path = os.path.join(result_folder_path, f"Q_pt_{index}.dat")
        with open(Q_save_path, 'w') as f:
            for i, flow in enumerate(flow_data):
                f.write(f"{flow}\n")
        
        # Save pressure data
        P_save_path = os.path.join(result_folder_path, f"P_pt_{index}.dat")
        with open(P_save_path, 'w') as f:
            for i, pressure in enumerate(pressure_data):
                f.write(f"{pressure}\n")
        
        print(f"Data saved:")
        print(f"  - Q_pt_{index}.dat: {len(flow_data)} time steps")
        print(f"  - P_pt_{index}.dat: {len(pressure_data)} time steps")
        
        # Print some statistics
        print(f"\nFlow statistics for pt {index}:")
        print(f"  Min: {min(flow_data):.6f}")
        print(f"  Max: {max(flow_data):.6f}")
        print(f"  Mean: {np.mean(flow_data):.6f}")
        
        print(f"\nPressure statistics for pt {index}:")
        print(f"  Min: {min(pressure_data):.6f}")
        print(f"  Max: {max(pressure_data):.6f}")
        print(f"  Mean: {np.mean(pressure_data):.6f}")
        
    else:
        print("No data found to save.")

    return flow_data, pressure_data


def find_pt_id_from_xyzts(xyzts_path, z_given):
    '''
    Find point id from the xyzts.dat file. given z coordinate.
    
    - read xyzts.dat and find the point id (largest z < z) and point id (smallest z > z)
    - return the point id

    #procedure
    1. read the xyzts.dat file and get the all z coordinates of the points.
    2. find the point id close to the given z coordinate.

    '''
    # Read xyzts.dat: data rows start at the third line with 4 columns: x y z s
    # Point ids are implicit, starting at 0 for the first data row.
    z_values = []
    try:
        with open(xyzts_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                # Data rows have exactly 4 numeric columns
                if len(parts) != 4:
                    continue
                try:
                    # x, y, z, s (we only need z)
                    _x = float(parts[0])
                    _y = float(parts[1])
                    z = float(parts[2])
                    _s = float(parts[3])
                except ValueError:
                    continue
                z_values.append(z)
    except FileNotFoundError:
        raise FileNotFoundError(f"xyzts file not found: {xyzts_path}")

    if not z_values:
        raise ValueError("No valid xyzts data rows (with 4 columns) were found.")

    # Find the point id close to the given z coordinate.
    for idx, z in enumerate(z_values):
        if z >= z_given:
            return idx # return the point id
    return len(z_values) - 1

#post processing for the 1D simulation
def post_processing_1d(total_time_step = 500, time_step_size = 0.01, result_dir = './result', json_path = None,
                     t_cycle = 1.0, start_pt_id = 0, end_pt_id = 49):

    '''
        Input:
            - total_time_step: total number of time steps
            - time_step_size: time step size
            - result_folder_path: result folder path where .json files are saved
            - t_cycle: cardiac cycle time
            - start_pt_id: start tree point id
            - end_pt_id: end tree point id
    '''

    # get the start and end time step to analyze
    start_t_step, end_t_step = start_end_t_step(total_time_step, time_step_size, t_cycle)
    print(f"Start time step: {start_t_step}")
    print(f"End time step: {end_t_step}")
    
    Q_in_data, P_in_data = data_at_sepcific_pt(start_t_step, end_t_step, start_pt_id, result_dir)
    Q_out_data, P_out_data = data_at_sepcific_pt(start_t_step, end_t_step, end_pt_id, result_dir)

    # get the systolic and diastolic time step from the P_in data.
    sys_step_num = np.argmax(P_in_data) 
    dia_step_num = np.argmin(P_in_data)
    
    print(f"Systolic time step: {sys_step_num}")
    print(f"Diastolic time step: {dia_step_num}")

    #Systolic data
    P_sys_in = float(f"{P_in_data[sys_step_num]:.6f}")
    P_sys_out = float(f"{P_out_data[sys_step_num]:.6f}")
    Q_sys = float(f"{Q_out_data[sys_step_num]:.6f}")
    R_sys = float(P_sys_out / Q_sys)

    #Diastolic data
    P_dia_in = float(f"{P_in_data[dia_step_num]:.6f}")
    P_dia_out = float(f"{P_out_data[dia_step_num]:.6f}")
    Q_dia = float(f"{Q_out_data[dia_step_num]:.6f}")
    R_dia = float(P_dia_out / Q_dia)

    #Mean data
    P_mean_in = float(f"{np.mean(P_in_data):.6f}")
    P_mean_out = float(f"{np.mean(P_out_data):.6f}")
    Q_mean = float(f"{np.mean(Q_out_data):.6f}")
    R_mean = float(P_mean_out / Q_mean)

    #FFR
    

    # from tabulate import tabulate
    table = [
        ["P_inlet (systolic)", P_sys_in],
        ["P_outlet (systolic)", P_sys_out],
        ["Q_outlet (systolic)", Q_sys],
        ["P_inlet (diastolic)", P_dia_in],
        ["P_outlet (diastolic)", P_dia_out],
        ["Q_outlet (diastolic)", Q_dia], 
        ["P_inlet (average)", P_mean_in],
        ["P_outlet (average)", P_mean_out],
        ["Q_outlet (average)", Q_mean]
    ]
    from tabulate import tabulate
    print(tabulate(table, headers=["Parameter", "Value"], tablefmt="grid"))


    # Save as JSON file: Q, P_in, P_out in sequence, sys, dia, mean in a dict
    if json_path is not None:
        import json
        data_dict = {
            "systolic":  {"Q": Q_sys,  "Pin": P_sys_in,  "Pout": P_sys_out, "R": R_sys},
            "diastolic":  {"Q": Q_dia,  "Pin": P_dia_in,  "Pout": P_dia_out, "R": R_dia},
            "mean": {"Q": Q_mean, "Pin": P_mean_in, "Pout": P_mean_out, "R": R_mean}
        }
        # The 'indent' parameter in json.dump specifies the number of spaces to use for indentation in the output file, making the JSON more readable.
        json_path = os.path.join(json_path)
        with open(json_path, "w") as f:
            json.dump(data_dict, f, indent=4)

    return

def post_processing_FFR(total_time_step = 500, time_step_size = 0.01, result_dir = './result', json_path = None,
                     t_cycle = 1.0, pt_id_proximal = 0, pt_id_distal = 49):
    '''
    Input:
        - total_time_step: total number of time steps
        - time_step_size: time step size
        - result_dir: result directory
        - t_cycle: cardiac cycle time
        - pt_id_proximal: proximal point id
        - pt_id_distal: distal point id

    FFR : Fractional Flow Reserve
    FFR = mean(P_distal) / mean(P_proximal)

    P_proximal = P_inlet
    P_distal = 2 ~ 3 cm distal to the stenosis.
    '''

    # get the start and end time step to analyze
    start_t_step, end_t_step = start_end_t_step(total_time_step, time_step_size, t_cycle)
        
    _, P_proximal = data_at_sepcific_pt(start_t_step, end_t_step, pt_id_proximal, result_dir)
    _, P_distal = data_at_sepcific_pt(start_t_step, end_t_step, pt_id_distal, result_dir)
    
    # get the systolic and diastolic time step from the P_in data.
    P_proximal_mean = np.mean(P_proximal)
    P_distal_mean = np.mean(P_distal)
    
    FFR = float(P_distal_mean / P_proximal_mean)

    #open json file and in the dict add FFR
    with open(json_path, 'r') as f:
        data = json.load(f)
    data['FFR'] = FFR
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4)
    return FFR
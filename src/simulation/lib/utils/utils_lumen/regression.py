import numpy as np
from scipy.stats import linregress



##Toools
def read_file(file_path):
    '''
    slab.txt -> [[P on pt1 at t1 P on pt2 at t1 P on pt3 at t1 /n ], [P on pt1 at t2 P on pt2 at t2 P on pt3 at t2, /n ], ...]
    '''
    with open(file_path, 'r') as file:
        lines = file.readlines()
    return lines

def write_file(file_path, lines):
    with open(file_path, 'w') as file:
        for line in lines:
            file.write(line)

def write_transposed_file(file_path, lines):
    """
    입력된 lines(문자열 리스트)를 각 줄을 공백으로 분리하여 2차원 리스트로 만들고,
    전치(transpose)한 뒤, 각 행을 공백으로 join하여 파일에 씁니다.
    """
    # 각 줄의 앞뒤 공백 제거 후 split (여러 개의 공백, 탭 모두 구분자로 취급)
    data = [line.strip().split() for line in lines]
    # 데이터가 비어있거나 각 줄의 길이가 다르면 오류가 날 수 있음에 주의
    # 전치: 각 원소는 tuple 형태
    transposed_data = list(zip(*data))
    # 각 tuple을 공백으로 join하고 마지막에 newline 추가
    new_lines = [" ".join(item) + "\n" for item in transposed_data]
    write_file(file_path, new_lines)

'''
(Made by WJ choi and modified a bit by JEFF)

Procedure of this code.
1. Read slab.txt and FlowHist.dat
2. Get the pressure data at the low, high, transient time step
3. Get the Q data at the low, high, transient time step
4. Calculate the slope of the trend line for each column using linear regression
5. Save the result to slopes.dat
'''
def read_data_from_inp(inp_path):
    """
    Extract key-value pairs from solver_ramp.inp file.
    Returns dictionary with parameter names as keys and their values as items.
    """
    data_dict = {}
    
    with open(inp_path, 'r') as file:
        for line in file:
            line = line.strip()
            
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue
                
            # Look for lines with colon (key: value format)
            if ':' in line:
                # Split on colon, taking only the first occurrence
                parts = line.split(':', 1)
                if len(parts) == 2:
                    key = parts[0].strip()
                    value = parts[1].strip()
                    
                    # Remove inline comments (everything after #)
                    if '#' in value:
                        value = value.split('#')[0].strip()
                    
                    # Try to convert to appropriate data type
                    if value.lower() in ['true', 'false']:
                        data_dict[key] = value.lower() == 'true'
                    elif value.replace('.', '').replace('-', '').isdigit():
                        # Check if it's a valid number (including decimals and negatives)
                        if '.' in value:
                            data_dict[key] = float(value)
                        else:
                            data_dict[key] = int(value)
                    else:
                        try:
                            # Try float first for scientific notation, etc.
                            data_dict[key] = float(value)
                        except ValueError:
                            # If all else fails, keep as string
                            data_dict[key] = value
    
    return data_dict




# Main function
import os
def regression( slab_path = 'slab.txt', flow_hist_path = 'FlowHist.dat', ramp_inp_path = 'solver_ramp.inp', save_dir = "./result"):
    '''
        !!!! 
        FlowHist data saved all the time step data, however, slab.txt data saved only at the restart time step.
        !!! save_dir is the directory to save the result. default is ./result


        Required Parameters:
        -  slab.txt
        -  FlowHist.dat
        -  .inp file( we need to know following data)
            - num_timestep
            - time_step_size
            - restart_save_step
            - Q_st_step
            - Q_scale_factor
            - Q_max_factor

        return 
        - slab_p_path
        - variable_path
        - slopes_path
    '''


    ###################################################################
    inp_data = read_data_from_inp(ramp_inp_path)
    
    #num_timestep = inp_data['Number of Timesteps']
    time_step_size = inp_data['Time Step Size'] # 0.0002
    Q_st_step = inp_data['Start Timestep for Flow Control'] # 250
    Q_scale_factor = inp_data['Flow Control Scale Factor'] # 0.001
    Q_max_factor = inp_data['Maximum Flow Control Scale Factor'] # 1.5
    slab_save_step = 5 # set from the xyzts.dat file.

    #Q ramped setting
    # Q_stop_step - time step when Q stop increasing
    # d = Q_stop_step - Q_st_step (increasing time interval)
    # Since, Q_max = Q in + (d * Q_scale_factor) * Q in
    d = (Q_max_factor - 1) / Q_scale_factor
    Q_stop_step = Q_st_step + d #Now we can get the Q_stop_step.

    #Denote, followings are time step at low high and transient pressure step on the slab.txt
    # Remeber slab.txt time step is scaled down from the FlowHist.dat time step(which saved all of them.)
    P_low_step = int(Q_st_step / slab_save_step)
    P_high_step = int(Q_stop_step / slab_save_step)
    P_transient_step = int((Q_st_step + Q_stop_step) / (slab_save_step * 2)) # at the middle.
    
    # At the increasing time interval, we should pick two data point to calculate the slope.
    # we will pick quarter and 3/4 at the increasing time interval.


    ###################################################################
    # 1. slab.txt and FlowHist.dat
    slab_lines = read_file(slab_path)
    flow_hist_lines = read_file(flow_hist_path)
    # 3. slab.txt에서 low, high, transient 해당 번째 줄 저장
    pressure_result_lines = [slab_lines[i-1] for i in [P_low_step, P_high_step, P_transient_step]]
    # Transpose slab_pressure.dat
    slab_p_path = os.path.join(save_dir, 'slab_pressure.dat')
    write_transposed_file(slab_p_path, pressure_result_lines)

    '''
        Now we get slab_pressure.dat as follows(p1: low, p2: transient, p3: high):
        p1 p3 p2 at point1
        p1 p3 p2 at point 2
        p1 p3 p2 at point 3
        ...
        p1 p3 p2 at point n = 50
    '''

    # 5. FlowHist.dat에서 P_low*steps, P_high*steps, P_transient*steps번째 줄 저장
    variable_lines = [flow_hist_lines[i-1] for i in [int(Q_st_step), int(Q_stop_step), int((Q_st_step + Q_stop_step) / 2)]]
    variable_path = os.path.join(save_dir, 'variables.dat')
    write_transposed_file(variable_path, variable_lines)
    
    '''
    variables.dat -> (Q1: low, Q2: transient, Q3: high)
        Q1 Q3 Q2 at the outlet
    '''

    # 6. FlowHist.dat slope calculation
    start_line = int(Q_st_step + d * (0.25)) # Q_st_step -> Q_end_Step = Q_st_step + d (increaing interval)
    end_line = int(Q_st_step + d * (0.75))
    trend_lines = flow_hist_lines[start_line-1:end_line]

    # split and make them to float, and then save as list on each line.
    data_columns = np.array([list(map(float, line.split())) for line in trend_lines])

    # Create x-axis values (time)
    time = np.arange(start_line, end_line + 1) * time_step_size

    # Calculate the slope of the trend line for each column using linear regression
    num_columns = data_columns.shape[1]  # number of columns
    slopes = []

    #using the linear regressin, calcaulte the slope of the trend line
    for col in range(num_columns):
        slope, _, _, _, _ = linregress(time, data_columns[:, col])
        slopes.append(slope)

    # Save the result to slopes.dat
    slopes_path = os.path.join(save_dir, 'slopes.dat')
    with open(slopes_path, 'w') as f:
        for slope in slopes:
            f.write(f"{slope}\n")

    return slab_p_path, variable_path, slopes_path



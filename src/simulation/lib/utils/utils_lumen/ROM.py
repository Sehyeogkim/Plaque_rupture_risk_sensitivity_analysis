import numpy as np

def parameter_calculator(element_number, slab_p_path, variables_rearranged_path):
    # Initialize parameters 
    p1 = []
    p2 = []
    p3 = []
    new_p1 = []
    new_p2 = []
    new_p3 = []
    data = []
    rows_to_remove = []

    # Reading slab pressure results of 3D case
    with open(slab_p_path, 'r') as datfile1:
        for line in datfile1:
            row = line.split()
            p1.append(float(row[0]))
            p2.append(float(row[1]))
            p3.append(float(row[2]))        

    # Reading segment number, dqdt, q1(low), q2(high), q3(transient)
    with open(variables_rearranged_path, 'r') as datfile2:
        for line in datfile2:
            columns = line.split()
            data.append([int(columns[0]), float(columns[1]), float(columns[2]), float(columns[3]), float(columns[4])])

    # Make a list of information that has segment number and pressure data
    index_for_loop = 0
    for segment, num_elements in enumerate(element_number):
        for i in range(index_for_loop, index_for_loop + num_elements):
            new_p1.append([segment, p1[i]])
            new_p2.append([segment, p2[i]])
            new_p3.append([segment, p3[i]])
        index_for_loop += num_elements

    # Initialize a, b, c parameters
    number_of_abc = np.shape(new_p1)[0] 
    a = np.zeros((number_of_abc, 1))
    b = np.zeros((number_of_abc, 1))
    c = np.zeros((number_of_abc, 1))

    # First value of parameters are zero
    a[0] = 0
    b[0] = 0
    c[0] = 0

    # Calculate a, b, c parameters
    for n in range(1, number_of_abc):
        qL = data[new_p1[n-1][0]][2]
        qH = data[new_p1[n-1][0]][3]
        qC = data[new_p1[n-1][0]][4]
        dqdt = data[new_p1[n-1][0]][1]

        b[n] = ((new_p1[n-1][1] - new_p1[n][1])/qL - (new_p2[n-1][1] - new_p2[n][1])/qH)/(qL - qH)
        a[n] = (new_p1[n-1][1] - new_p1[n][1])/qL - b[n] * qL
        c[n] = (new_p3[n-1][1] - new_p3[n][1] - a[n] * qC - b[n] * qC * qC)/dqdt

    # Removing first row of segment number
    for m in range(1, number_of_abc):
        if new_p1[m][0] != new_p1[m-1][0]:
            rows_to_remove.append(m)

    a = np.delete(a, rows_to_remove)
    b = np.delete(b, rows_to_remove)
    c = np.delete(c, rows_to_remove)

    return a, b, c

def create_treedata_final(a, b, c, number_of_segment, input_filename, output_filename):
 # 1. output_treedata.dat의 내용을 읽습니다.
    with open(input_filename, "r") as infile:
        lines = infile.readlines()
    
    # 2. treedata_final.dat 파일을 작성합니다.
    with open(output_filename, "w") as outfile:
        # (1) output_treedata.dat의 모든 내용을 그대로 기록합니다.
        outfile.writelines(lines)
        
        # (2) "point double flow x 1" 작성 (x는 number_of_segment)
        outfile.write(f"point double flow {number_of_segment} 1\n")
        # (3) number_of_segment 줄에 "0.001"을 기록합니다.
        for _ in range(number_of_segment):
            outfile.write("0.001\n")
        
        # (4) "point double flow_acceleration x 1" 작성
        outfile.write(f"point double flow_acceleration {number_of_segment} 1\n")
        # (5) number_of_segment 줄에 "0"을 기록합니다.
        for _ in range(number_of_segment):
            outfile.write("0\n")
        
        # (6) "point double pressure x 1" 작성
        outfile.write(f"point double pressure {number_of_segment} 1\n")
        # (7) number_of_segment 줄에 "0"을 기록합니다.
        for _ in range(number_of_segment):
            outfile.write("0\n")
        
        # (8) "point double segment_resistance x 3" 작성
        outfile.write(f"point double segment_resistance {number_of_segment} 3\n")
        # (9) a, b, c의 데이터를 zip하여 각 줄에 "a b c" 형식으로 기록합니다.
        for ai, bi, ci in zip(a, b, c):
            outfile.write(f"{ai} {bi} {ci}\n")

def get_number_of_segment_and_elements(filename):
    """
    output_treedata.dat 파일을 읽어, 
      - number_of_segment: 파일의 첫 번째 줄의 두번째 숫자 
        (예: 첫 줄이 "421 26"이면 number_of_segment는 26)
      - element_numbers: process_treedata와 유사하게 처리하면서,
                         각 처리할 줄에서 두번째 숫자를 추출하여 리스트로 만듭니다.
                         이때, 두번째 줄의 두번째 숫자도 첫 번째 요소로 추가합니다.
    반환: (number_of_segment, element_numbers)
    """
    with open(filename, "r") as f:
        # 빈 줄은 제외하고 읽습니다.
        lines = [line.strip() for line in f if line.strip()]
    
    if not lines:
        print("오류: 파일에 데이터가 없습니다.")
        return None, []
    
    # 첫 번째 줄의 두번째 숫자가 number_of_segment
    first_line_parts = lines[0].split()
    if len(first_line_parts) < 2:
        print("오류: 첫 번째 줄에는 최소 두 개의 숫자가 있어야 합니다.")
        return None, []
    try:
        number_of_segment = int(first_line_parts[1])
        global_numbers = int(first_line_parts[0])
    except ValueError:
        print("오류: 첫 번째 줄의 두번째 숫자가 정수가 아닙니다.")
        return None, []
    
    element_numbers = []
    i = 1  # process_treedata와 동일하게 첫 줄(인덱스 0)은 건너뜁니다.
    
    # 두번째 줄 (인덱스 1): 두 개의 숫자가 있어야 하며, 두번째 숫자가 건너뛸 줄의 수를 의미합니다.
    if i < len(lines):
        parts = lines[i].split()
        if len(parts) < 2:
            print(f"오류: {i+1}번째 줄에 숫자가 두 개 이상 있어야 합니다.")
            return number_of_segment, element_numbers
        try:
            skip_count = int(parts[1])
        except ValueError:
            print(f"오류: {i+1}번째 줄의 두번째 숫자가 정수가 아닙니다.")
            return number_of_segment, element_numbers
        # 수정된 부분: 두번째 줄의 두번째 숫자를 element_numbers의 첫 요소로 추가
        element_numbers.append(skip_count)
        # 두번째 줄 이후 skip_count개의 줄을 건너뛰고 다음 줄부터 처리합니다.
        i = i + skip_count + 1
    else:
        return number_of_segment, element_numbers
    
    # 이후 3개 이상의 숫자가 있는 줄마다 두번째 숫자를 추출하여 element_numbers에 추가합니다.
    while i < len(lines):
        parts = lines[i].split()
        if len(parts) < 3:
            break  # 3개 미만이면 처리를 중단
        
        try:
            element = int(parts[1])
        except ValueError:
            print(f"오류: {i+1}번째 줄의 두번째 숫자가 정수가 아닙니다.")
            break
        element_numbers.append(element)
        
        # 현재 줄의 두번째 숫자를 건너뛸 줄 수로 사용합니다.
        try:
            skip_count = int(parts[1])
        except ValueError:
            print(f"오류: {i+1}번째 줄의 두번째 숫자가 정수가 아닙니다.")
            break
        
        i = i + skip_count + 1

    return number_of_segment, element_numbers, global_numbers

def process_treedata(filename):
    """
    output_treedata.dat 파일을 읽어, 각 처리할 줄에서 
    첫번째와 세번째 숫자를 추출하여 (segment, parent) 튜플 리스트를 반환합니다.
    
    파일의 첫 줄은 건너뛰고, 두번째 줄의 두번째 숫자(skip_count)를 이용하여 
    이후 줄들을 건너뛰는 방식입니다.
    """
    tree_data = []
    with open(filename, "r") as f:
        lines = [line.strip() for line in f if line.strip()]
    
    # 첫 줄은 건너뜁니다.
    i = 1

    # 두번째 줄(인덱스 1)에서 건너뛸 줄 수를 읽습니다.
    if i < len(lines):
        parts = lines[i].split()
        print(parts)
        if len(parts) < 2:
            print(f"오류: {i+1}번째 줄에 숫자가 두 개 이상 있어야 합니다.")
            return tree_data
        try:
            skip_count = int(parts[1])
        except ValueError:
            print(f"오류: {i+1}번째 줄의 두번째 숫자가 정수가 아닙니다.")
            return tree_data
        i = i + skip_count + 1
    else:
        return tree_data

    # 이후 각 줄에서 3개의 숫자 중 첫번째와 세번째 값을 읽어 튜플로 저장합니다.
    while i < len(lines):
        parts = lines[i].split()

        if len(parts) < 3:
            break
        try:
            seg = int(parts[0])
            parent = int(parts[2])
        except ValueError:
            print(f"오류: {i+1}번째 줄의 숫자 변환에 실패하였습니다.")
            break

        tree_data.append((seg, parent))
        
        try:
            skip_count = int(parts[1])
        except ValueError:
            print(f"오류: {i+1}번째 줄의 두번째 숫자가 정수가 아닙니다.")
            break

        i = i + skip_count + 1

    return tree_data

def read_outlet_information(filename):
    """
    outlet_information.dat 파일을 읽어 각 줄의 세번째 숫자(해당 outlet의 segment_number)를 문자열 리스트로 반환합니다.
    """
    outlet_segments = []
    with open(filename, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            outlet_segments.append(parts[2])
    return outlet_segments

def read_single_column_file(filename):
    """
    slopes.dat 같이 각 줄에 하나의 값이 있는 파일을 읽어, 
    각 줄의 첫번째 값을 문자열 리스트로 반환합니다.
    """
    values = []
    with open(filename, "r") as f:
        for line in f:
            if line.strip():
                values.append(line.strip().split()[0])
    return values

def read_multi_column_file(filename):
    """
    variables.dat 같이 각 줄에 여러 값이 있는 파일을 읽어,
    각 줄을 공백 기준으로 split한 2차원 리스트(각 행에 3개 값이 있다고 가정)를 반환합니다.
    """
    data = []
    with open(filename, "r") as f:
        for line in f:
            if line.strip():
                data.append(line.strip().split())
    return data

def build_intermediate_data(outlet_info_path, slopes_path, variables_path):
    """
    outlet_information.dat, slopes.dat, variables.dat 파일을 읽어
    각 outlet에 해당하는 intermediate 정보를 구성합니다.
    
    각 outlet의 정보는 outlet_information.dat의 3열 (segment_number), 
    slopes.dat의 값, 그리고 variables.dat의 3개 값으로 구성되며,
    결과는 dict { segment_number (int) : (slope, var1, var2, var3) } (모두 float) 로 반환됩니다.
    """
    outlet_segs = read_outlet_information(outlet_info_path)
    slopes_all = read_single_column_file(slopes_path)
    variables_all = read_multi_column_file(variables_path)

    # 단일 세그먼트 또는 파일 불일치 대비: outlet 행 개수에 맞춰 정렬
    # 관례: variables.dat, slopes.dat 의 첫 행이 inlet, 이후가 outlet 들이라면 inlet(첫 행)을 제거
    def align_rows(all_rows, expected_count):
        if len(all_rows) == expected_count:
            return all_rows[-expected_count:]
        if len(all_rows) == expected_count + 1:
            return all_rows[1:]
        # 가능한 경우 마지막 expected_count 개만 사용
        if len(all_rows) > expected_count:
            return all_rows[-expected_count:]
        # 부족한 경우: 가능한 만큼만 사용하고 경고
        print("경고: 데이터 행 수가 outlet 수보다 적습니다. 가능한 데이터만 사용합니다.")
        return all_rows

    variables = align_rows(variables_all, len(outlet_segs))
    slopes = align_rows(slopes_all, len(outlet_segs))

    if len(variables) != len(outlet_segs) or len(slopes) != len(outlet_segs):
        print("경고: outlet_information.dat 과 변수/기울기 행 수가 일치하지 않습니다.")

    intermediate = {}
    for i, seg_str in enumerate(outlet_segs):
        try:
            seg = int(seg_str)
            # 가용 범위 내에서만 매핑
            if i < len(variables):
                v1, v2, v3 = map(float, variables[i])
            else:
                v1 = v2 = v3 = 0.0
            s = float(slopes[i]) if i < len(slopes) else 0.0
            intermediate[seg] = (s, v1, v2, v3)
        except ValueError:
            print(f"오류: intermediate data의 {i+1}번째 행에서 숫자 변환에 실패하였습니다.")
            continue
    return intermediate

def build_children_dict(tree):
    """
    tree (튜플 리스트: (segment, parent))를 받아, 
    parent -> [child, child, ...] 형태의 dictionary를 생성합니다.
    """
    children = {}
    for seg, parent in tree:
        children.setdefault(parent, []).append(seg)
    return children

def compute_aggregated(segment, children_dict, intermediate_data, memo):
    """
    재귀적으로 segment의 aggregate 값을 계산합니다.
    
    - 만약 해당 segment가 intermediate_data에 존재하면 그 값을 그대로 사용합니다.
    - 존재하지 않으면 자식들의 aggregate 값을 element‑wise(열별) 합산합니다.
    - 자식이 없는 경우엔 모두 0을 반환합니다.
    
    memo: 이미 계산된 값을 저장하기 위한 dict.
    """
    if segment in memo:
        return memo[segment]
    if segment in intermediate_data:
        memo[segment] = intermediate_data[segment]
        return intermediate_data[segment]
    total = (0.0, 0.0, 0.0, 0.0)
    if segment in children_dict:
        for child in children_dict[segment]:
            child_val = compute_aggregated(child, children_dict, intermediate_data, memo)
            total = tuple(t + c for t, c in zip(total, child_val))
    memo[segment] = total
    return total

def write_variable_rearranged(outfile, tree, intermediate_data):
    """
    tree: process_treedata()로 얻은 (segment, parent) 튜플 리스트  
    intermediate_data: outlet 및 variable 정보를 모은 dict { segment : (slope, var1, var2, var3) }
    
    최종적으로 variable_rearranged.dat 파일을 출력합니다.
    - 1열: 모든 segment 번호 (tree에서 나타난 child와 parent 모두; root인 0 포함)
    - 2열~5열: 해당 segment의 slope와 variable 값들 (만약 intermediate 정보에 없으면 자식들의 aggregate 합산 값)
    """
    children_dict = build_children_dict(tree)

    # tree에 나타난 child와 parent, 그리고 intermediate_data에 존재하는 segment, root(0)를 모두 포함
    all_segments = set()
    for seg, parent in tree:
        all_segments.add(seg)
        all_segments.add(parent)
    all_segments.update(intermediate_data.keys())
    all_segments.add(0)

    memo = {}
    sorted_segments = sorted(all_segments)
    with open(outfile, "w") as f:
        for seg in sorted_segments:
            agg = compute_aggregated(seg, children_dict, intermediate_data, memo)
            f.write(f"{seg} {agg[0]} {agg[1]} {agg[2]} {agg[3]}\n")




import os
def ROM_main(tree_path, outlet_info_path, slab_p_path, variable_path, slopes_path, save_dir):
    
    
    tree_data = process_treedata(tree_path)
    intermediate_data = build_intermediate_data(outlet_info_path, slopes_path, variable_path)
    
    var_rearr_path = os.path.join(save_dir, "variable_rearranged.dat")
    write_variable_rearranged(var_rearr_path, tree_data, intermediate_data)
    
    number_of_segment, element_numbers, global_numbers = get_number_of_segment_and_elements(tree_path)
    a, b, c = parameter_calculator(element_numbers, slab_p_path, var_rearr_path)
    
    abc_tree_path = os.path.join(save_dir, "treedata_abc.dat")
    create_treedata_final(a, b, c, global_numbers, tree_path, abc_tree_path)

    return abc_tree_path
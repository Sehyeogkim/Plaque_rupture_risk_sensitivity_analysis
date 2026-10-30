import math
import numpy as np
from scipy import optimize
'''
@updated 0720 jeff
- radius_lumen function is updated to include skewness.
'''

#Function to completely save the Autodesk Inventor file to .step file.
def save_as_step(model, doc, output_path):
    # 3) Select STEP Translator Add-In (by using VBA GUID)
    STEP_GUID   = "{90AF7F40-0C01-11D5-8E83-0010B541CD80}"
    translator  = model.inv.ApplicationAddIns.ItemById(STEP_GUID)

    # 4) TranslationContext, options, DataMedium (For translation)
    tobj = model.inv.TransientObjects
    context = tobj.CreateTranslationContext()
    context.Type = 13059 #file to file.
    options   = tobj.CreateNameValueMap() # options for step saving.
    TargetData  = tobj.CreateDataMedium()
    options.SetValue("ApplicationProtocol", 3)
    options.SetValue("SplineFitTolerance", 0.00001) #cm @updated 0729 jeff
    TargetData.FileName = output_path

    translator.SaveCopyAs(doc, context, options, TargetData)
    return


#Lumen Centerline and radius
def y_center_lumen(model, z):

    '''
    @updated 0720 jeff
    Lumen parameter included version.
    including model.skewness range [-1,1]
    '''
    
    if abs(z) <= model.z_lesion:
        a = -model.z_lesion
        b = model.z_lesion
        z_peak = model.lumen_skewness * (model.z_lesion)

        '''
        h(z) + r(z) = r_max
        let us define h(z) along the z axis.
        '''
        h = 0.0
        h_peak = model.r_max - model.r_min
        #h_peak = - y_center_lumen_old(model, 0) # Maximum height of the peak.
        
        # Normalize z left domain to [0, 1]
        if z <= z_peak:
            denominator = z_peak - a
            if denominator == 0: # Avoid division by zero
                norm_z = 1.0
            else:
                norm_z = (z - a) / denominator
            
            # Smooth up curve (0 -> h_peak)
            h = (h_peak/2.0) * (1 - np.cos(norm_z * np.pi))
            
        # Normalize z right domain to [0, 1]    
        else:
            denominator = b - z_peak
            if denominator == 0: # Avoid division by zero
                norm_z = 1.0
            else:
                norm_z = (z - z_peak) / denominator
                
            # Smooth down curve (h_peak -> 0)
            h = (h_peak/2.0) * (1 + np.cos(norm_z * np.pi))
            
        # Final radius = maximum radius - radius decrease amount
        return - h
    
    else:
        return 0.0
        
def radius_lumen(model, z):
    r = model.r_max + y_center_lumen(model, z)
    return r


#Soild Vessel Centerline and radius
def y_center_lesion(model, z):

    '''
    @updated 0813 jeff
    Lumen parameter included version.
    including model.skewness range [-1,1]
    '''
    
    if abs(z) <= model.z_lesion:
        a = -model.z_lesion
        b = model.z_lesion
        z_peak = model.lumen_skewness * (model.z_lesion)

        '''
        h(z) + r(z) = r_max
        let us define h(z) along the z axis.
        '''
        h = 0.0
        h_peak = model.r_pos - model.r_ex
        
        # Normalize z left domain to [0, 1]
        if z <= z_peak:
            denominator = z_peak - a
            if denominator == 0: # Avoid division by zero
                norm_z = 1.0
            else:
                norm_z = (z - a) / denominator
            
            # Smooth up curve (0 -> h_peak)
            h = (h_peak/2.0) * (1 - np.cos(norm_z * np.pi))
            
        # Normalize z right domain to [0, 1]    
        else:
            denominator = b - z_peak
            if denominator == 0: # Avoid division by zero
                norm_z = 1.0
            else:
                norm_z = (z - z_peak) / denominator
                
            # Smooth down curve (h_peak -> 0)
            h = (h_peak/2.0) * (1 + np.cos(norm_z * np.pi))
            
        # Final radius = maximum radius - radius decrease amount
        return h
    
    else:
        return 0.0

def radius_lesion(model, z):
    r = y_center_lesion(model, z) + model.r_ex
    return r


#Lipid arc angle
def alpha_theta(model, z):
    '''
    return, the θ(z) (******half value, radian)

    New Boundary condition:
    - θ(z = z_peak) = model.alpha
    - θ(|z| = z_lipid ) = 0
    - dramatic drop near |z| = z_lipid

    By experiment, sqrt(cosx) would be the appropriate function.
    '''

    # degree to radian the MAximum lipid arc angle at z = z_peak
    alpha_rad = math.radians(model.alpha) 
    
    #For simplicity, we set alpha as h
    #We use Sqrt(cos(x)) function to make the dramatic drop near |z| = z_lipid

    a = model.z_lipid_L  
    b = model.z_lipid_R
    z_peak = model.z_peak

    h = 0.0
    h_peak = alpha_rad

    # Normalize z left domain to [0, 1]
    if z <= z_peak:
        denominator = z_peak - a
        if denominator == 0: # Avoid division by zero
            norm_z = 1.0
        else:
            norm_z = (z - a) / denominator
        
        # norm_z = 0 -> alpha = 0, norm_z = 1 -> alpha = alpha_rad
        h = h_peak * (math.sqrt(math.sin(norm_z * np.pi / 2))) 
        
    # Normalize z right domain to [0, 1]    
    else:
        denominator = b - z_peak
        if denominator == 0: # Avoid division by zero
            norm_z = 1.0
        else:
            norm_z = (z - z_peak) / denominator
            
        # norm_z = 0 (z = z_peak) -> alpha = alpha_rad, norm_z = 1 (z = z_lipid) -> alpha = 0
        h = h_peak * math.sqrt(math.cos(norm_z * np.pi / 2))
    

    alpha = h / 2 # half value (radian)
    return alpha


# Lipid external boundary y coordinate
def lipid_ex_y_coordinate(model,z):

    A_max = y_center_lesion(model, model.z_peak) + radius_lesion(model, model.z_peak) - model.lipid_wall_thickness
    A_min_left = y_center_lumen(model, model.z_lipid_L) + radius_lumen(model, model.z_lipid_L)
    A_min_right = y_center_lumen(model, model.z_lipid_R) + radius_lumen(model, model.z_lipid_R)
    A_min = 0.0
    if z < 0:
        A_min = A_min_left
    else:
        A_min = A_min_right


    a = model.z_lipid_L  
    b = model.z_lipid_R
    z_peak = model.z_peak

    h = 0.0
    h_peak = A_max - A_min

    # Normalize z left domain to [0, 1]
    if z <= z_peak:
        denominator = z_peak - a
        if denominator == 0: # Avoid division by zero
            norm_z = 1.0
        else:
            norm_z = (z - a) / denominator
        
        # norm_z = 0 -> alpha = 0, norm_z = 1 -> alpha = alpha_rad
        h = h_peak * math.sqrt(math.sin(norm_z * np.pi / 2))
        
    # Normalize z right domain to [0, 1]    
    else:
        denominator = b - z_peak
        if denominator == 0: # Avoid division by zero
            norm_z = 1.0
        else:
            norm_z = (z - z_peak) / denominator
            
        # norm_z = 0 (z = z_peak) -> alpha = alpha_rad, norm_z = 1 (z = z_lipid) -> alpha = 0
        h = h_peak * math.sqrt(math.cos(norm_z * np.pi / 2))
    
    lipid_ex_y = h + A_min
    
    return lipid_ex_y

#Lipid beta angle
def beta(model, θ, z):
    '''
    calculateions are done w/ the half theat -> half beta
    '''
    if θ == 0.0:
        return 0.0

    y_lumen  = y_center_lumen(model, z)
    y_lesion = y_center_lesion(model, z)
    r_lipid  = radius_lesion(model, z) - model.lipid_wall_thickness

    def f(β):
        pt_B = (0.0, y_lesion)
        pt_A = (0.0, y_lumen)
        pt_C = (r_lipid * math.sin(β),
                r_lipid * math.cos(β) + y_center_lesion(model, z))

        vect_CB = (pt_B[0] - pt_C[0], pt_B[1] - pt_C[1])
        vect_CA = (pt_A[0] - pt_C[0], pt_A[1] - pt_C[1])

        dot_product = vect_CB[0] * vect_CA[0] + vect_CB[1] * vect_CA[1]
        magnitude_CB = math.sqrt(vect_CB[0]**2 + vect_CB[1]**2)
        magnitude_CA = math.sqrt(vect_CA[0]**2 + vect_CA[1]**2)

        cos_angle = dot_product / (magnitude_CB * magnitude_CA)
        γ = math.acos(cos_angle)

        return 	β - γ - θ
    
    # θ + γ(β) = β

    sol = optimize.root_scalar(f,
                            bracket=[θ, math.pi],    # 이분법 기반 메서드(brentq 등)용 구간
                            method='brentq',         # 혹은 'bisect', 'secant', 'ridder'
                            xtol=1e-6)

    return sol.root
 
def calculate_safe_fillet_radius(model, z, safe_ratio = 0.3):
    """
    Calculate safe fillet radius based on geometry
    """

    theat = alpha_theta(model, z)
    a = y_center_lesion(model, z) - y_center_lumen(model, z) # distance between two centers.
    
    r_max_safe = a * math.sin(theat) * safe_ratio
    r_max_safe = round(r_max_safe, 4)

    return r_max_safe


#fc (local minimum thickness case, not used on the current average thickness project.)
def radius_fc_abnormal(model, θ_prime, θ_range, θ, z):

    '''
    return: r(θ)
    from θ_prime and r_large, r_small
    '''
    r_l = radius_lumen(model, z) + model.avg_fc_th #large radius
    r_s = radius_lumen(model, z) + model.min_fc_th #small radius

    A = (r_l - r_s) / 2 #amplitude
    T = 2 * θ_range # period

    # - cos function form.
    r = - A * math.cos(2 * math.pi * (θ - θ_prime) / T) + (r_l + r_s) / 2

    return r

#0720 jeff
def create_dense_points(start, end, n_points, dense_end=True):
    """
    This function is used for the guideline lofting on Lumen and Solid.
    Dense points between start and end.
    using exponential function.
    
    Args:
        start, end: start and end
        n_points: total number of points
        dense_end: True means dense at the end, False means dense at the start
    """
    if dense_end:
        # dense at the end
        t = np.linspace(0, 1, n_points)
        # exponential function (slowly at 0, quickly at 1)
        t_transformed = 1 - (1 - t)**2  # 수정!
        points = start + (end - start) * t_transformed
    else:
        # dense at the start
        t = np.linspace(0, 1, n_points)
        # exponential function (quickly at 0, slowly at 1)
        t_transformed = t**2  # 수정!
        points = start + (end - start) * t_transformed
    
    return points.tolist()




#0826 jeff
def find_ab(model, n_samples: int = 4001, tol: float = 1e-10):

    """
    derived from the equation of the lipid arc angle.
    for (a, b) such that:
      1) b - a = model.lipid_length
      2) f(a) = f(b), with the function: f(z) = y_center_lumen(model, z) + radius_lumen(model, z)
      3) z_lesion > b > z_peak > a > - z_lesion

    Returns
    -------
    a, b : floats

    Raises
    ------
    ValueError if no root is found in the feasible interval.
    """

    z_peak = model.z_peak
    z_lesion = model.z_lesion
    lipid_length = model.lipid_length

    L = float(lipid_length)
    if L <= 0:
        raise ValueError("lipid_length must be positive.")

    # Feasible interval for a:
    #   a > -z_lesion
    #   a <  z_peak
    #   a > z_peak - L     (since b=a+L > z_peak)
    #   a < z_lesion - L   (since b=a+L < z_lesion)
    a_lo = max(-z_lesion, z_peak - L)
    a_hi = min(z_peak,   z_lesion - L)

    # A tiny padding to avoid hitting the bounds numerically
    pad = 1e-12 * max(1.0, abs(a_lo), abs(a_hi))
    a_lo += pad
    a_hi -= pad

    if not (a_lo < a_hi):
        raise ValueError("Feasible interval for 'a' is empty. Check z_lesion, z_peak, and lipid_length.")

    def f(z: float) -> float:
        return float(y_center_lumen(model, z) + radius_lumen(model, z))

    def g(a: float) -> float:
        # g(a) = f(a) - f(a+L) ; root wanted
        return f(a) - f(a + L)

    # Sample g on the feasible interval to bracket roots
    xs = np.linspace(a_lo, a_hi, n_samples)
    vals = np.array([g(x) for x in xs], dtype=float)

    # Helper: bisection (robust; no SciPy needed)
    def bisect_root(fun, left, right, tol=tol, maxiter=100):
        fl = fun(left)
        fr = fun(right)
        if not np.isfinite(fl) or not np.isfinite(fr):
            raise RuntimeError("Non-finite function values in bracket.")
        if fl == 0.0:
            return left
        if fr == 0.0:
            return right
        if fl * fr > 0:
            raise RuntimeError("Root not bracketed for bisection.")
        a_, b_ = left, right
        fa, fb = fl, fr
        for _ in range(maxiter):
            m = 0.5 * (a_ + b_)
            fm = fun(m)
            if fm == 0.0 or abs(b_ - a_) < tol:
                return m
            # choose sub-interval
            if fa * fm <= 0:
                b_, fb = m, fm
            else:
                a_, fa = m, fm
        return 0.5 * (a_ + b_)

    # 1) exact hits
    exact_idx = np.where(np.isclose(vals, 0.0, atol=tol, rtol=0.0))[0].tolist()

    # 2) sign-change brackets
    sign_changes = []
    for i in range(len(xs) - 1):
        v1, v2 = vals[i], vals[i + 1]
        if not (np.isfinite(v1) and np.isfinite(v2)):
            continue
        if v1 == 0.0:
            exact_idx.append(i)
        elif v2 == 0.0:
            exact_idx.append(i + 1)
        elif v1 * v2 < 0:
            sign_changes.append((xs[i], xs[i + 1]))

    candidates = []

    # add exact solutions
    for idx in exact_idx:
        a = xs[idx]
        b = a + L
        if (-z_lesion < a < z_peak) and (z_peak < b < z_lesion):
            candidates.append(a)

    # refine sign-change brackets
    for left, right in sign_changes:
        try:
            a = bisect_root(g, left, right, tol=tol)
            b = a + L
            if (-z_lesion < a < z_peak) and (z_peak < b < z_lesion):
                candidates.append(a)
        except RuntimeError:
            continue

    if not candidates:
        raise ValueError("No (a, b) pair satisfying both equations was found in the feasible interval.")

    # Choose the root whose midpoint is closest to z_peak (natural tie-breaker)
    def midpoint_distance(a):
        return abs((a + (a + L)) * 0.5 - z_peak)

    a_star = min(candidates, key=midpoint_distance)
    b_star = a_star + L
    return float(a_star), float(b_star)

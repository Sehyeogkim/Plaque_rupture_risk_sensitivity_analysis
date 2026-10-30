import vtk
from pathlib import Path

def _smoothing_vtu(VTU_PATH: Path,
                   STL_PATH: Path,
                   RESOLUTION = [60, 60, 60],
                   ISO_VALUE: float = 0.5,
                   SMOOTH_MODE: str = "taubin",
                   SMOOTH_STRENGTH: float = 0.01,
                   NUM_ITERATIONS: int = 500):

    '''
    Smoothing pipeline: VTU -> Voxelize -> Marching Cubes -> Smooth -> STL

    SMOOTH_MODE options:
        "taubin"    - vtkWindowedSincPolyDataFilter (volume-preserving, recommended)
        "laplacian" - vtkSmoothPolyDataFilter (stronger smoothing, may shrink volume)
        "windowed_sinc" - legacy mode, same as taubin but with different default params

    # [!] (Very Important) Isosurface Value
    # Determines which value to use as the boundary surface in voxelized data.
    # Example 1: To extract boundary surface separating material IDs (0 and 1) -> 0.5
    # Example 2: To extract based on distance field value 10.0 -> 10.0
    # Example 3: To extract based on pressure 50 Pa -> 50.0
    '''
    print(f"\n--- Smoothing VTU to STL ---")
    print(f"  Mode: {SMOOTH_MODE}, Resolution: {RESOLUTION}, Iterations: {NUM_ITERATIONS}, Strength/Passband: {SMOOTH_STRENGTH}")

    # --- 2. Load VTU ---
    try:
        reader = vtk.vtkXMLUnstructuredGridReader()
        reader.SetFileName(str(VTU_PATH))
        reader.Update()
        vtu_data = reader.GetOutput()

        if vtu_data.GetNumberOfPoints() == 0:
            raise Exception("VTU file has no point data.")
        if vtu_data.GetPointData().GetScalars() is None:
            raise Exception("VTU file has no scalar data (Scalars) to resample.")

        print(f"'{VTU_PATH}' loaded successfully.")
        print(f" - Original scalar range: {vtu_data.GetPointData().GetScalars().GetRange()}")

    except Exception as e:
        print(f"Error occurred while loading file: {e}")
        exit()

    # --- 3. VTU -> vtkImageData (Resample to Image) ---
    print(f"Starting Resample to Image... (Resolution: {RESOLUTION})")
    resampler = vtk.vtkResampleToImage()
    resampler.SetInputConnection(reader.GetOutputPort())

    # Set resampling bounds to match the original VTU's bounding box
    bounds = vtu_data.GetBounds()
    resampler.SetSamplingBounds(bounds)

    # Set resampling resolution (number of voxels)
    resampler.SetSamplingDimensions(RESOLUTION)

    # Configure to interpolate based on VTU's point data (not cell data)
    resampler.SetUseInputBounds(False)
    resampler.Update()

    image_data = resampler.GetOutput()
    print(f"Resample to Image completed. Generated voxel resolution: {image_data.GetDimensions()}")

    # --- 4. vtkImageData -> vtkPolyData (Marching Cubes) ---
    print(f"Running Marching Cubes (Flying Edges)... (ISO value: {ISO_VALUE})")

    # Use vtkFlyingEdges3D, the latest/fast version of Marching Cubes
    cutter = vtk.vtkFlyingEdges3D()
    cutter.SetInputData(image_data)
    cutter.ComputeNormalsOn()  # Compute normals for smoothing and rendering
    cutter.SetValue(0, ISO_VALUE) # Set boundary surface value
    cutter.Update()

    print("Marching Cubes completed. Surface extracted.")

    # --- 5. Smoothing ---
    if SMOOTH_MODE == "laplacian":
        print(f"Starting Laplacian smoothing (iterations={NUM_ITERATIONS}, relaxation={SMOOTH_STRENGTH})...")
        smoother = vtk.vtkSmoothPolyDataFilter()
        smoother.SetInputData(cutter.GetOutput())
        smoother.SetNumberOfIterations(NUM_ITERATIONS)
        smoother.SetRelaxationFactor(SMOOTH_STRENGTH)
        smoother.FeatureEdgeSmoothingOff()
        smoother.BoundarySmoothingOn()
        smoother.Update()

    elif SMOOTH_MODE in ("taubin", "windowed_sinc"):
        print(f"Starting Taubin/WindowedSinc smoothing (iterations={NUM_ITERATIONS}, passband={SMOOTH_STRENGTH})...")
        smoother = vtk.vtkWindowedSincPolyDataFilter()
        smoother.SetInputData(cutter.GetOutput())
        smoother.SetNumberOfIterations(NUM_ITERATIONS)
        smoother.SetPassBand(SMOOTH_STRENGTH)
        smoother.NormalizeCoordinatesOn()
        smoother.Update()

    else:
        raise ValueError(f"Unknown SMOOTH_MODE: '{SMOOTH_MODE}'. Use 'taubin', 'laplacian', or 'windowed_sinc'.")

    final_surface = smoother.GetOutput()
    print(f"Smoothing completed. Points: {final_surface.GetNumberOfPoints()}, Cells: {final_surface.GetNumberOfCells()}")


    # --- 6. Save as STL file ---
    print(f"Saving final result to '{STL_PATH}' file...")
    stl_writer = vtk.vtkSTLWriter()
    stl_writer.SetFileName(str(STL_PATH))
    stl_writer.SetInputData(final_surface)
    stl_writer.SetFileTypeToBinary() # Binary format is more efficient for STL
    stl_writer.Write()

    print("\n--- All tasks completed ---")

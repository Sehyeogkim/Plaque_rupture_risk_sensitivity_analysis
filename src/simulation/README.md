# Forward-model code

The forward model maps rows of `data/inputs/lhs_design_1000.csv` to hemodynamic and fibrous-cap stress outputs. The analysis uses the supplied CSV files in `data/simulation/`; running the forward model requires the licensed software and machine-specific inputs below.

| Stage | Code | Software |
|---|---|---|
| CAD geometry | `geometry/inventor/` | Autodesk Inventor 2026, pywin32 |
| Calcification geometry | `geometry/calcification/` | Gmsh, FreeCAD |
| Solid and fluid meshing | `meshing/` | Gmsh, Simmetrix MeshSim |
| Hemodynamics | `hemodynamics/`, `lib/lumen_codes/` | svSolver and an in-house 1D solver |
| Structural stress | `structural/` | ANSYS Mechanical APDL 2025 R1 |
| Stress and plaque-burden extraction | `postprocessing/` | PyVista, NumPy |
| Mesh studies | `mesh_independence/` | Solver-specific scripts |

The in-house 1D solver, raw meshes, and VTU stress fields are not included. Some scripts retain absolute paths from the machines used for the simulation campaign. The CP calcification mesher is configured to read the bundled design CSV by row index.

Of 1,000 sampled cases, 936 LAP and 879 CP cases have usable systolic and diastolic stress outputs. `data/simulation/failed_cases_{LAP,CP}.csv` records the other cases.

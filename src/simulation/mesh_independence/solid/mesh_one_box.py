#!/usr/bin/env python
"""Single-case LAP solid mesher, BOX size-field recipe (mesh independence test).

Uses meshing/solid/lap_meshing_box.py (Box field, VIn=mesh_size, VOut=0.2, Thickness=3mm,
order 2 / SecondOrderLinear / Algo2D=5 / Algo3D=10 HXT, + HighOrder optimize + quality.json)
-- the same Box recipe as the March production cohort, with the HighOrder untangling pass
that the dist recipe also uses.

Reads solid_data/case_<c>/{lumen,solid,lipid,fc}.stp and lesion_length from
pre_data/parameter_0728.csv (row index == case_id).

Usage: mesh_one_box.py <case> <nproc> <mesh_size> <outdir> [algo2d] [algo3d]
Writes <outdir>/solid_LAP_<case>_box_<size>[_a<algo2d>_<algo3d>].msh + matching quality json.

algo2d/algo3d override what the module hardcodes (2D=5 Delaunay, 3D=10 HXT). Needed
because at size 0.05 Algorithm 5 fails on surface 8 ("N elements remain invalid"),
after which gmsh over-refines the surface and the element count blows up ~4x in the
WRONG direction. 2D: 1 MeshAdapt, 5 Delaunay, 6 Frontal-Delaunay.
3D: 1 Delaunay, 4 Frontal, 10 HXT.
Prints one final `STATUS {json}` line (incl. node count read back from the .msh header).
"""
import sys, os, io, json, time, shutil, contextlib
from pathlib import Path
import pandas as pd

Q = Path("/home/jeff/project/54_analysis")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "meshing" / "solid"))
CSV = Q / "pre_data" / "parameter_0728.csv"


def node_count(msh):
    with open(msh, "r") as f:
        for line in f:
            if line.startswith("$Nodes"):
                return int(f.readline().split()[1])
    return None


def main():
    case = int(sys.argv[1])
    nproc = int(sys.argv[2])
    size = float(sys.argv[3])
    outdir = Path(sys.argv[4])
    algo2d = int(sys.argv[5]) if len(sys.argv) > 5 else None
    algo3d = int(sys.argv[6]) if len(sys.argv) > 6 else None
    outdir.mkdir(parents=True, exist_ok=True)

    tag = f"{size:g}"
    if algo2d is not None or algo3d is not None:
        tag += f"_a{algo2d or 5}_{algo3d or 10}"
    out = outdir / f"solid_LAP_{case}_box_{tag}.msh"
    if out.exists():
        print("STATUS " + json.dumps({"case": case, "size": size, "status": "skipped",
                                      "nodes": node_count(out)}), flush=True)
        return

    src = Q / "solid_data" / f"case_{case}"
    work = outdir / f"work_box_{tag}"          # gmsh writes solid_type_1.msh here
    work.mkdir(parents=True, exist_ok=True)

    lesion = float(pd.read_csv(str(CSV)).loc[case, "lesion_length"])

    import lap_meshing_box as M
    if algo2d is not None or algo3d is not None:
        # The module sets Mesh.Algorithm / Mesh.Algorithm3D just before generate(3),
        # so override them inside a generate() wrapper -- the last write wins.
        import gmsh
        _gen = gmsh.model.mesh.generate

        def _gen_override(dim=3):
            if algo2d is not None:
                gmsh.option.setNumber("Mesh.Algorithm", algo2d)
            if algo3d is not None:
                gmsh.option.setNumber("Mesh.Algorithm3D", algo3d)
            return _gen(dim)

        gmsh.model.mesh.generate = _gen_override

    t0 = time.time()
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            msh = M.gmshing_solid(
                src / "lumen.stp", src / "solid.stp", src / "lipid.stp", src / "fc.stp",
                work, nproc, True, False, size, lesion,
            )
    except Exception as e:
        sys.stdout.write(buf.getvalue()[-4000:])
        print("STATUS " + json.dumps({"case": case, "size": size, "status": "failed",
                                      "error": f"{type(e).__name__}: {e}",
                                      "seconds": round(time.time() - t0, 1)}), flush=True)
        return

    sys.stdout.write(buf.getvalue()[-4000:])

    st = {"case": case, "size": size, "status": "done", "recipe": "box",
          "algo2d": algo2d or 5, "algo3d": algo3d or 10,
          "lesion_length": lesion, "seconds": round(time.time() - t0, 1)}
    qj = work / "quality.json"
    if qj.exists():
        st.update(json.loads(qj.read_text()))
        shutil.copy(str(qj), str(outdir / f"quality_box_{tag}.json"))

    shutil.move(str(msh), str(out))
    st["nodes"] = node_count(out)
    st["msh"] = str(out)
    print("STATUS " + json.dumps(st), flush=True)


if __name__ == "__main__":
    main()

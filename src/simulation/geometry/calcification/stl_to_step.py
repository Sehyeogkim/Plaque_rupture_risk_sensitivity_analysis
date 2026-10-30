#!/usr/bin/env python3
"""
STL 파일을 STEP 파일로 변환하는 스크립트
FreeCAD를 사용하여 변환합니다.

사용법:
    python stl_to_step.py <stl_file> [output_step_file]
    
또는 FreeCAD 명령줄 도구 사용:
    freecadcmd -c "import sys; sys.path.append('.'); exec(open('stl_to_step.py').read())"
"""

import sys
import os

def stl_to_step(stl_path, step_path=None):
    """
    STL 파일을 STEP 파일로 변환
    
    Args:
        stl_path: 입력 STL 파일 경로
        step_path: 출력 STEP 파일 경로 (None이면 자동 생성)
    """
    try:
        import FreeCAD
        import Part
        import Mesh
    except ImportError:
        print("FreeCAD가 설치되어 있지 않습니다.")
        print("설치 방법:")
        print("  conda install -c conda-forge freecad")
        print("  또는")
        print("  sudo apt-get install freecad freecad-python3")
        sys.exit(1)
    
    # FreeCAD 초기화 (headless 모드)
    # 로그 레벨 설정 (선택사항)
    try:
        FreeCAD.Console.SetStatus("Log", 0)
        FreeCAD.Console.SetStatus("Msg", 0)
        FreeCAD.Console.SetStatus("Wrn", 1)
        FreeCAD.Console.SetStatus("Err", 1)
    except:
        pass  # SetStatus가 없는 버전의 FreeCAD일 수 있음
    
    # 출력 파일 경로가 지정되지 않으면 자동 생성
    if step_path is None:
        base_name = os.path.splitext(stl_path)[0]
        step_path = base_name + ".step"
    
    print(f"STL 파일 로딩: {stl_path}")
    
    # STL 파일 로드
    mesh = Mesh.Mesh()
    mesh.read(stl_path)

    print(f"Mesh 정보: {len(mesh.Facets)} 개의 면")
    
    # Mesh를 Solid로 변환
    print("Mesh를 Solid로 변환 중...")
    
    # FreeCAD 문서 생성
    doc = FreeCAD.newDocument("TempDoc")
    
    try:
        # Mesh 객체 생성
        mesh_obj = doc.addObject("Mesh::Feature", "Mesh")
        mesh_obj.Mesh = mesh
        # Mesh를 Part Shape로 변환
        print("Mesh를 Part Shape로 변환 중...")
        shape = Part.Shape()
        shape.makeShapeFromMesh(mesh.Topology, 0.5)  # tolerance
        
        # RefineShape로 작은 엣지/면 제거하여 fragment 감소
        # FreeCAD Python API에서는 removeSplitter()를 사용
        print("RefineShape로 형상 최적화 중...")
        try:
            # 방법 1: removeSplitter() 사용 (refineShape와 유사한 효과)
            shape = shape.removeSplitter()
        except:
            # 방법 2: Part 객체를 통해 refineShape 시도
            try:
                part_obj = doc.addObject("Part::Feature", "Part")
                part_obj.Shape = shape
                # GUI 명령어를 Python으로 실행
                import FreeCADGui
                FreeCADGui.doCommand("App.ActiveDocument.Part.Shape = App.ActiveDocument.Part.Shape.removeSplitter()")
                doc.recompute()
                shape = part_obj.Shape
            except:
                # refineShape가 불가능하면 그대로 진행
                print("RefineShape를 적용할 수 없습니다. 원본 형상 사용.")
                pass
        
        # Shape를 Solid로 변환
        print("Shape를 Solid로 변환 중...")
        solid = Part.makeSolid(Part.Solid(shape))
        
        # STEP 파일로 저장
        print(f"STEP 파일로 저장: {step_path}")
        solid.exportStep(step_path)
        
    except Exception as e:
        print(f"오류 발생: {e}")
        print("대체 방법 시도 중...")
        import traceback
        traceback.print_exc()
        
        # 방법 2: Part 객체를 문서에 추가하여 변환
        try:
            part_obj = doc.addObject("Part::Feature", "Part")
            shape = Part.Shape()
            shape.makeShapeFromMesh(mesh.Topology, 0.1)
            part_obj.Shape = shape
            
            # RefineShape로 작은 엣지/면 제거하여 fragment 감소
            print("RefineShape로 형상 최적화 중...")
            try:
                # removeSplitter() 사용 (refineShape와 유사한 효과)
                shape = part_obj.Shape.removeSplitter()
                part_obj.Shape = shape
            except:
                # refineShape가 불가능하면 그대로 진행
                print("RefineShape를 적용할 수 없습니다. 원본 형상 사용.")
                shape = part_obj.Shape
            
            # Part 객체를 Solid로 변환
            solid = Part.makeSolid(Part.Solid(shape))
            solid.exportStep(step_path)
        except Exception as e2:
            print(f"대체 방법도 실패: {e2}")
            raise
    
    finally:
        # 문서 닫기
        FreeCAD.closeDocument("TempDoc")
    
    print(f"변환 완료: {step_path}")
    return step_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용법: python stl_to_step.py <stl_file> [output_step_file]")
        print("\n예시:")
        print("  python stl_to_step.py reconstructed_smooth_surface.stl")
        print("  python stl_to_step.py reconstructed_smooth_surface.stl output.step")
        sys.exit(1)
    
    stl_file = sys.argv[1]
    step_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not os.path.exists(stl_file):
        print(f"오류: 파일을 찾을 수 없습니다: {stl_file}")
        sys.exit(1)
    
    stl_to_step(stl_file, step_file)


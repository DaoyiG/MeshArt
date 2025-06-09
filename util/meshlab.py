import pymeshlab


def meshlab_proc(meshpath, output_path):
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(str(meshpath))
    ms.meshing_merge_close_vertices(threshold=pymeshlab.PercentageValue(1))
    ms.meshing_remove_duplicate_faces()
    ms.meshing_remove_null_faces()
    ms.meshing_remove_duplicate_vertices()
    ms.meshing_remove_unreferenced_vertices()
    ms.save_current_mesh(str(output_path), save_vertex_color=False, save_vertex_coord=False, save_face_color=False, save_wedge_texcoord=False)
    ms.clear()

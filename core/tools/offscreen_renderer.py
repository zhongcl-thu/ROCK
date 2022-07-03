import numpy as np
import trimesh
import pyrender
import ipdb


class Renderer:
    def __init__(self, model_paths, cam_K, H, W, scale=None):
        if not isinstance(model_paths, list):
            print("model_paths have to be list")
            raise RuntimeError
        self.scene = pyrender.Scene(
            ambient_light=[1., 1., 1.], bg_color=[0, 0, 0])
        self.camera = pyrender.IntrinsicsCamera(
            fx=cam_K[0, 0], fy=cam_K[1, 1], cx=cam_K[0, 2], cy=cam_K[1, 2], znear=0.1, zfar=2.0)
        self.cam_node = self.scene.add(self.camera, pose=np.eye(4))
        self.mesh_nodes = []

        for model_path in model_paths:
            print('model_path', model_path)
            obj_mesh = trimesh.load(model_path)
            if abs((scale-1)) > 1e-4:
                obj_mesh.vertices = obj_mesh.vertices*scale
                obj_mesh.faces = obj_mesh.faces*scale
            obj_mesh.vertices = obj_mesh.vertices
            obj_mesh.faces = obj_mesh.faces
            #colorVisual = obj_mesh.visual.to_color()
            mesh = pyrender.Mesh.from_trimesh(obj_mesh)
            mesh_node = self.scene.add(mesh, pose=np.eye(
                4), parent_node=self.cam_node)  # Object pose parent is cam
            self.mesh_nodes.append(mesh_node)

        self.H = H
        self.W = W

        self.r = pyrender.OffscreenRenderer(self.W, self.H)
        self.glcam_in_cvcam = np.array([[1, 0, 0, 0],
                                        [0, -1, 0, 0],
                                        [0, 0, -1, 0],
                                        [0, 0, 0, 1]])
        self.cvcam_in_glcam = np.linalg.inv(self.glcam_in_cvcam)

    def render(self, ob_in_cvcams):
        assert isinstance(ob_in_cvcams, list)
        for i, ob_in_cvcam in enumerate(ob_in_cvcams):
            ob_in_glcam = self.cvcam_in_glcam.dot(ob_in_cvcam)
            self.scene.set_pose(self.mesh_nodes[i], ob_in_glcam)
        color, depth = self.r.render(self.scene)  # depth: float
        return color, depth

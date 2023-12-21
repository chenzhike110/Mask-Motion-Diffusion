import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import json
from libs.body_model import BodyModel
import math

from pytorch3d.renderer import PerspectiveCameras, MeshRasterizer, RasterizationSettings
from pytorch3d.structures import Meshes
from pytorch3d.renderer import look_at_view_transform
from pytorch3d.transforms import euler_angles_to_matrix


class SHHQPreprocessor(nn.Module):

    def __init__(self, gen_height, gen_width, **kwargs):
        super().__init__()

        self.height = gen_height
        self.width = gen_width
        self.device = getattr(kwargs, "device", "cpu") 

        # self.x = nn.Linear(1, 1)
        self.mode = kwargs.get("coordinate_mode", "fix_body")

        self.register_buffer("vertex_approximation", torch.zeros([6890], dtype=torch.long))
        self.register_buffer("smpl_faces", torch.zeros([13776, 3], dtype=torch.long))
        self.register_buffer("smpl_faces_to_labels", torch.zeros([13776], dtype=torch.long))

        raster_settings = RasterizationSettings(
            image_size=(gen_height, gen_width),
            blur_radius=0.0, faces_per_pixel=1)

        self.rasterizer = MeshRasterizer(raster_settings=raster_settings)
        # same as denseposeCOCO
        self.register_buffer("CoarseParts", torch.tensor(
            [0, 1, 1, 2, 2, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8]
        ))


    @torch.no_grad()
    def init_smpl(self, smpl_faces, smpl_faces_to_labels):

        self.smpl_faces.copy_(smpl_faces)
        self.smpl_faces_to_labels.copy_(smpl_faces_to_labels)
        # self.smpl_faces_to_labels = self.CoarseParts[self.smpl_faces_to_labels]


    @torch.no_grad()
    def forward(self, data, rotate=False, elev=0, azim=-85, dist=1.4, **kwargs):

        batch_size = data["vertices"].shape[0]

        elevs = torch.randn(batch_size) * (kwargs["e_stddev"] if rotate else 0) + elev
        azims = torch.randn(batch_size) * (kwargs["a_stddev"] if rotate else 0) + azim
        dists = torch.randn(batch_size) * (kwargs["d_stddev"] if rotate else 0) + dist

        return self.forward_with_rotation(data, elevs, azims, dists, **kwargs)


    @torch.no_grad()
    def forward_with_rotation(self, data, elevs, azims, dists, **kwargs):

        if self.mode == "fix_body":
            data, R_raster = self._forward_fix_body(data, elevs, azims, dists, **kwargs)
        else:
            return NotImplementedError

        data = self._forward_rasterize(data, R_raster, **kwargs)

        return data


    @torch.no_grad()
    def _forward_fix_body(self, data, elevs, azims, dists, **kwargs):

        # start = time.time()

        batch_size = data["vertices"].shape[0]
        R, T = look_at_view_transform(dist=dists, elev=elevs, azim=azims, at=((0, -0.2, 0),), up=((0, 0, 1),), device=self.device)
        data["T"] = T
        return data, R

    @torch.no_grad()
    def _forward_rasterize(self, data, R_raster, **kwargs):

        batch_size = data["vertices"].shape[0]

        faces = self.smpl_faces.unsqueeze(0).repeat(batch_size, 1, 1)
        meshes = Meshes(verts=data["vertices"], faces=faces).to(self.device)

        T_raster = data["T"]

        cameras = PerspectiveCameras( R=R_raster, T=T_raster, 
                                     device=self.device)
        framents = self.rasterizer(meshes, cameras=cameras)
        # rasterize semantics
        # pix_to_face, zbuf, bary_coords, dists

        pix_to_face = framents.pix_to_face
        bg_mask = (pix_to_face < 0)
        pix_to_face = pix_to_face % len(self.smpl_faces)
    
        # rasterize segments
        # for batch_idx in range(batch_size):
        rasterized_body_seg = (self.smpl_faces_to_labels[pix_to_face] + 1) * 255. / self.CoarseParts.shape[0]
        rasterized_body_seg[bg_mask] = 0
        rasterized_body_seg = rasterized_body_seg.expand(-1, -1, -1, 3).cpu().numpy().astype(np.uint8)
        # rasterized_body_seg = cv2.applyColorMap(rasterized_body_seg, cv2.COLORMAP_JET)
        
        # rasterized_body_segs.append(rasterized_body_seg)

        data["rasterized_segments"] = rasterized_body_seg

        return data


@torch.no_grad()
def get_preprocessor(meta):

    preprocessor = SHHQPreprocessor(**meta)

    smpl = BodyModel("./deps/body_models/smplh/neutral/model.npz")
    smpl_faces = smpl.f

    densepose_data_path = "./datasets/densepose_data.json"
    densepose_data = json.load(open(densepose_data_path))

    smpl_faces_to_labels = list(range(len(smpl_faces)))
    smpl_faces_to_labels = torch.tensor(densepose_data["smpl_faces_to_densepose_faces"], dtype=torch.long)[smpl_faces_to_labels]
    smpl_faces_to_labels = torch.tensor(densepose_data["densepose_faces_to_labels"], dtype=torch.long)[smpl_faces_to_labels]

    preprocessor.init_smpl(smpl_faces, smpl_faces_to_labels)

    return preprocessor
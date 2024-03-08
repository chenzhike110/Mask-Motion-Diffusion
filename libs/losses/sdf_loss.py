import torch
from sdf import SDF
import torch.nn as nn

class Object_sdf(nn.Module):
    """
    signed distance field for mesh
    """
    def __init__(self, mesh):
        super(Object_sdf, self).__init__()
        self.sdf = SDF()

        self.register_buffer('faces', torch.from_numpy(mesh.faces)) 
        self.register_buffer('verts', torch.from_numpy(mesh.vertices).float())

        vmin = self.verts.min(dim=0)[0]
        vmax = self.verts.max(dim=0)[0]

        self.register_buffer('center', (vmax + vmin) / 2.0)
        self.register_buffer('scale', torch.max(vmax - vmin) / 2.0)

        self.verts = (self.verts - self.center) / self.scale

    def forward(self, queries):
        # normalize
        queries = (queries - self.center) / self.scale

        with torch.no_grad():
            phi = self.sdf(self.faces, self.verts, queries)
        
        closest = phi[:, :-1]
        distance = torch.norm(queries - closest, dim=-1)
        inside = (phi[:, -1] < 1).nonzero()
        distance[inside] *= -1
        return distance


import torch
from torch import Tensor
from typing import Tuple, Optional, List

Point = Tuple[float, float, float]
Face = Tuple[Point, Point, Point]


class Data():
    def __init__(self, vertex_features: Optional[Tensor] = None, vertex_positions: Optional[Tensor] = None,
                 vertex_adjacency: Optional[Tensor] = None, vertices_per_sample: List[int] = None,
                 faces: Optional[Tensor] = None, faces_per_sample: List[int] = None, y: Optional[Tensor] = None):
        self.vertex_features = vertex_features
        self.vertex_positions = vertex_positions
        self.vertex_adjacency = vertex_adjacency
        self.vertices_per_sample = vertices_per_sample
        self.faces = faces
        self.faces_per_sample = faces_per_sample
        self.y = y



# losses of of mesh prediction network

def voxel_loss():
    # minimize binary cross entropy between predicted voxel occupancy probabilities and true voxel occupancies.
    pass


# mesh losses

def mesh_sampling():
    # given a mesh sample a point cloud to be used in the loss functions
    pass


def chamfer_distance_between_point_clouds():
    pass


def normal_distance_between_point_clouds():
    pass


def edge_loss():
    # L(V,E) =1/|E| * ∑(v,v′)∈E ‖v−v′‖^2
    pass


# losses of the backbone network


def mask_loss():
    pass


def box_loss():
    pass

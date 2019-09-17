from torch.utils.cpp_extension import load
import torch
import datetime

if __name__ == "__main__":
    cubify_cpp = load(name="cubify_cpp", sources=["cubify.cpp", "cubify.cu"], verbose=True)
    import cubify_cpp

    C, H, W, threshold = 32, 32, 32, torch.zeros(3)
    voxel, result1, result2 = torch.zeros(C, H, W), torch.zeros(C * H * W, 24, 3), torch.zeros(C * H * W, 12, 9)
    iter_start = datetime.datetime.now()
    for i in range(32):
        cubify_cpp.get_fv(voxel, C, H, W, threshold, result1, result2)
    iter_end = datetime.datetime.now()
    print(iter_end - iter_start)

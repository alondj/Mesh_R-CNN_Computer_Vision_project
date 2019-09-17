#include <torch/extension.h>


void get_faces_vertices_cuda(torch::Tensor voxel, int  C, int H
            ,int W, torch::Tensor threshold,torch::Tensor result1, torch::Tensor result2);

void wraper(torch::Tensor voxel, int  C, int H
            ,int W, torch::Tensor threshold,torch::Tensor result1, torch::Tensor result2){
    get_faces_vertices_cuda(voxel,C,H,W,threshold,result1,result2);
 }


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("get_fv", &wraper, "get vertices");
}

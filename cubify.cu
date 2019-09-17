#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#define NUM_THREADS 500


template <typename scalar_t>
__global__ void get_faces_vertices_kernel
   (torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> array,
    int N,
    int C,
    int H,
    int W,
    torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> threshold,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> result1,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> result2) {

    auto t = blockDim.x * blockIdx.x + threadIdx.x;
    const auto per_thread = (C * H * W * N) / NUM_THREADS;
    t *= per_thread;

    if (t < C * H * W * N) {
        for (int k = t; k < t + per_thread; k++) {
            int b = k % N;
            int channel = k / N % C;
            int i = k / N / C / H % W;
            int j = k / N / C / H / W;

            int v_cell = 0;
            int f_cell = 0;

            scalar_t f1 = (scalar_t)channel;
            scalar_t f2 = (scalar_t)i;
            scalar_t f3 = (scalar_t)j;

            if (array[b][channel][i][j] > threshold[0]) {
                if (channel == 0 || array[b][channel - 1][i][j] <= threshold[0]) {
                     result1[k][v_cell+0][0]=(f1-0.5);
                     result1[k][v_cell+0][1]=(f2-0.5);
                     result1[k][v_cell+0][2]=(f3-0.5);


                     result1[k][v_cell+1][0]=(f1-0.5);
                     result1[k][v_cell+1][1]=(f2-0.5);
                     result1[k][v_cell+1][2]=(f3+0.5);

                     result1[k][v_cell+2][0]=(f1-0.5);
                     result1[k][v_cell+2][1]=(f2+0.5);
                     result1[k][v_cell+2][2]=(f3-0.5);

                     result1[k][v_cell+3][0]=(f1-0.5);
                     result1[k][v_cell+3][1]=(f2+0.5);
                     result1[k][v_cell+3][2]=(f3+0.5);

                     result2[k][f_cell+0][0]=result1[k][v_cell+0][0];
                     result2[k][f_cell+0][1]=result1[k][v_cell+0][1];
                     result2[k][f_cell+0][2]=result1[k][v_cell+0][2];
                     result2[k][f_cell+0][3]=result1[k][v_cell+1][0];
                     result2[k][f_cell+0][4]=result1[k][v_cell+1][1];
                     result2[k][f_cell+0][5]=result1[k][v_cell+1][2];
                     result2[k][f_cell+0][6]=result1[k][v_cell+2][0];
                     result2[k][f_cell+0][7]=result1[k][v_cell+2][1];
                     result2[k][f_cell+0][8]=result1[k][v_cell+2][2];

                     result2[k][f_cell+1][0]=result1[k][v_cell+1][0];
                     result2[k][f_cell+1][1]=result1[k][v_cell+1][1];
                     result2[k][f_cell+1][2]=result1[k][v_cell+1][2];
                     result2[k][f_cell+1][3]=result1[k][v_cell+2][0];
                     result2[k][f_cell+1][4]=result1[k][v_cell+2][1];
                     result2[k][f_cell+1][5]=result1[k][v_cell+2][2];
                     result2[k][f_cell+1][6]=result1[k][v_cell+3][0];
                     result2[k][f_cell+1][7]=result1[k][v_cell+3][1];
                     result2[k][f_cell+1][8]=result1[k][v_cell+3][2];

                     v_cell = v_cell + 4;
                     f_cell = f_cell + 2;
                }
                if (channel == C - 1 || array[b][channel + 1][i][j] <= threshold[0]) {
                     result1[k][v_cell+0][0]=(f1+0.5);
                     result1[k][v_cell+0][1]=(f2-0.5);
                     result1[k][v_cell+0][2]=(f3-0.5);
                     result1[k][v_cell+1][0]=(f1+0.5);
                     result1[k][v_cell+1][1]=(f2-0.5);
                     result1[k][v_cell+1][2]=(f3+0.5);
                     result1[k][v_cell+2][0]=(f1+0.5);
                     result1[k][v_cell+2][1]=(f2+0.5);
                     result1[k][v_cell+2][2]=(f3-0.5);
                     result1[k][v_cell+3][0]=(f1+0.5);
                     result1[k][v_cell+3][1]=(f2+0.5);
                     result1[k][v_cell+3][2]=(f3+0.5);

                     result2[k][f_cell+0][0]=result1[k][v_cell+0][0];
                     result2[k][f_cell+0][1]=result1[k][v_cell+0][1];
                     result2[k][f_cell+0][2]=result1[k][v_cell+0][2];
                     result2[k][f_cell+0][3]=result1[k][v_cell+1][0];
                     result2[k][f_cell+0][4]=result1[k][v_cell+1][1];
                     result2[k][f_cell+0][5]=result1[k][v_cell+1][2];
                     result2[k][f_cell+0][6]=result1[k][v_cell+2][0];
                     result2[k][f_cell+0][7]=result1[k][v_cell+2][1];
                     result2[k][f_cell+0][8]=result1[k][v_cell+2][2];

                     result2[k][f_cell+1][0]=result1[k][v_cell+1][0];
                     result2[k][f_cell+1][1]=result1[k][v_cell+1][1];
                     result2[k][f_cell+1][2]=result1[k][v_cell+1][2];
                     result2[k][f_cell+1][3]=result1[k][v_cell+2][0];
                     result2[k][f_cell+1][4]=result1[k][v_cell+2][1];
                     result2[k][f_cell+1][5]=result1[k][v_cell+2][2];
                     result2[k][f_cell+1][6]=result1[k][v_cell+3][0];
                     result2[k][f_cell+1][7]=result1[k][v_cell+3][1];
                     result2[k][f_cell+1][8]=result1[k][v_cell+3][2];

                     v_cell = v_cell + 4;
                     f_cell = f_cell + 2;
                }
                if (i == 0 || array[b][channel][i - 1][j] <= threshold[0]) {
                     result1[k][v_cell+0][0]=(f1+0.5);
                     result1[k][v_cell+0][1]=(f2-0.5);
                     result1[k][v_cell+0][2]=(f3-0.5);
                     result1[k][v_cell+1][0]=(f1+0.5);
                     result1[k][v_cell+1][1]=(f2-0.5);
                     result1[k][v_cell+1][2]=(f3+0.5);
                     result1[k][v_cell+2][0]=(f1-0.5);
                     result1[k][v_cell+2][1]=(f2-0.5);
                     result1[k][v_cell+2][2]=(f3-0.5);
                     result1[k][v_cell+3][0]=(f1-0.5);
                     result1[k][v_cell+3][1]=(f2-0.5);
                     result1[k][v_cell+3][2]=(f3+0.5);

                     result2[k][f_cell+0][0]=result1[k][v_cell+0][0];
                     result2[k][f_cell+0][1]=result1[k][v_cell+0][1];
                     result2[k][f_cell+0][2]=result1[k][v_cell+0][2];
                     result2[k][f_cell+0][3]=result1[k][v_cell+1][0];
                     result2[k][f_cell+0][4]=result1[k][v_cell+1][1];
                     result2[k][f_cell+0][5]=result1[k][v_cell+1][2];
                     result2[k][f_cell+0][6]=result1[k][v_cell+2][0];
                     result2[k][f_cell+0][7]=result1[k][v_cell+2][1];
                     result2[k][f_cell+0][8]=result1[k][v_cell+2][2];

                     result2[k][f_cell+1][0]=result1[k][v_cell+1][0];
                     result2[k][f_cell+1][1]=result1[k][v_cell+1][1];
                     result2[k][f_cell+1][2]=result1[k][v_cell+1][2];
                     result2[k][f_cell+1][3]=result1[k][v_cell+2][0];
                     result2[k][f_cell+1][4]=result1[k][v_cell+2][1];
                     result2[k][f_cell+1][5]=result1[k][v_cell+2][2];
                     result2[k][f_cell+1][6]=result1[k][v_cell+3][0];
                     result2[k][f_cell+1][7]=result1[k][v_cell+3][1];
                     result2[k][f_cell+1][8]=result1[k][v_cell+3][2];

                     v_cell = v_cell + 4;
                     f_cell = f_cell + 2;
                }
                if (i == H - 1 || array[b][channel][i + 1][j] <= threshold[0]) {
                     result1[k][v_cell+0][0]=(f1-0.5);
                     result1[k][v_cell+0][1]=(f2+0.5);
                     result1[k][v_cell+0][2]=(f3-0.5);
                     result1[k][v_cell+1][0]=(f1-0.5);
                     result1[k][v_cell+1][1]=(f2+0.5);
                     result1[k][v_cell+1][2]=(f3+0.5);
                     result1[k][v_cell+2][0]=(f1+0.5);
                     result1[k][v_cell+2][1]=(f2+0.5);
                     result1[k][v_cell+2][2]=(f3-0.5);
                     result1[k][v_cell+3][0]=(f1+0.5);
                     result1[k][v_cell+3][1]=(f2+0.5);
                     result1[k][v_cell+3][0]=(f3+0.5);

                     result2[k][f_cell+0][0]=result1[k][v_cell+0][0];
                     result2[k][f_cell+0][1]=result1[k][v_cell+0][1];
                     result2[k][f_cell+0][2]=result1[k][v_cell+0][2];
                     result2[k][f_cell+0][3]=result1[k][v_cell+1][0];
                     result2[k][f_cell+0][4]=result1[k][v_cell+1][1];
                     result2[k][f_cell+0][5]=result1[k][v_cell+1][2];
                     result2[k][f_cell+0][6]=result1[k][v_cell+2][0];
                     result2[k][f_cell+0][7]=result1[k][v_cell+2][1];
                     result2[k][f_cell+0][8]=result1[k][v_cell+2][2];

                     result2[k][f_cell+1][0]=result1[k][v_cell+1][0];
                     result2[k][f_cell+1][1]=result1[k][v_cell+1][1];
                     result2[k][f_cell+1][2]=result1[k][v_cell+1][2];
                     result2[k][f_cell+1][3]=result1[k][v_cell+2][0];
                     result2[k][f_cell+1][4]=result1[k][v_cell+2][1];
                     result2[k][f_cell+1][5]=result1[k][v_cell+2][2];
                     result2[k][f_cell+1][6]=result1[k][v_cell+3][0];
                     result2[k][f_cell+1][7]=result1[k][v_cell+3][1];
                     result2[k][f_cell+1][8]=result1[k][v_cell+3][2];

                     v_cell = v_cell + 4;
                     f_cell = f_cell + 2;
                }
                if (j == 0 || array[b][channel][i][j - 1] <= threshold[0]) {
                     result1[k][v_cell+0][0]=(f1+0.5);
                     result1[k][v_cell+0][1]=(f2-0.5);
                     result1[k][v_cell+0][2]=(f3-0.5);
                     result1[k][v_cell+1][0]=(f1-0.5);
                     result1[k][v_cell+1][1]=(f2-0.5);
                     result1[k][v_cell+1][2]=(f3-0.5);
                     result1[k][v_cell+2][0]=(f1+0.5);
                     result1[k][v_cell+2][1]=(f2+0.5);
                     result1[k][v_cell+2][2]=(f3-0.5);
                     result1[k][v_cell+3][0]=(f1-0.5);
                     result1[k][v_cell+3][1]=(f2+0.5);
                     result1[k][v_cell+3][2]=(f3-0.5);

                     result2[k][f_cell+0][0]=result1[k][v_cell+0][0];
                     result2[k][f_cell+0][1]=result1[k][v_cell+0][1];
                     result2[k][f_cell+0][2]=result1[k][v_cell+0][2];
                     result2[k][f_cell+0][3]=result1[k][v_cell+1][0];
                     result2[k][f_cell+0][4]=result1[k][v_cell+1][1];
                     result2[k][f_cell+0][5]=result1[k][v_cell+1][2];
                     result2[k][f_cell+0][6]=result1[k][v_cell+2][0];
                     result2[k][f_cell+0][7]=result1[k][v_cell+2][1];
                     result2[k][f_cell+0][8]=result1[k][v_cell+2][2];

                     result2[k][f_cell+1][0]=result1[k][v_cell+1][0];
                     result2[k][f_cell+1][1]=result1[k][v_cell+1][1];
                     result2[k][f_cell+1][2]=result1[k][v_cell+1][2];
                     result2[k][f_cell+1][3]=result1[k][v_cell+2][0];
                     result2[k][f_cell+1][4]=result1[k][v_cell+2][1];
                     result2[k][f_cell+1][5]=result1[k][v_cell+2][2];
                     result2[k][f_cell+1][6]=result1[k][v_cell+3][0];
                     result2[k][f_cell+1][7]=result1[k][v_cell+3][1];
                     result2[k][f_cell+1][8]=result1[k][v_cell+3][2];

                     v_cell = v_cell + 4;
                     f_cell = f_cell + 2;
                }
                if (j == W - 1 || array[b][channel][i][j + 1] <= threshold[0]) {
                     result1[k][v_cell+0][0]=(f1-0.5);
                     result1[k][v_cell+0][1]=(f2-0.5);
                     result1[k][v_cell+0][2]=(f3+0.5);
                     result1[k][v_cell+1][0]=(f1+0.5);
                     result1[k][v_cell+1][1]=(f2-0.5);
                     result1[k][v_cell+1][2]=(f3+0.5);
                     result1[k][v_cell+2][0]=(f1-0.5);
                     result1[k][v_cell+2][1]=(f2+0.5);
                     result1[k][v_cell+2][2]=(f3+0.5);
                     result1[k][v_cell+3][0]=(f1+0.5);
                     result1[k][v_cell+3][1]=(f2+0.5);
                     result1[k][v_cell+3][2]=(f3+0.5);

                     result2[k][f_cell+0][0]=result1[k][v_cell+0][0];
                     result2[k][f_cell+0][1]=result1[k][v_cell+0][1];
                     result2[k][f_cell+0][2]=result1[k][v_cell+0][2];
                     result2[k][f_cell+0][3]=result1[k][v_cell+1][0];
                     result2[k][f_cell+0][4]=result1[k][v_cell+1][1];
                     result2[k][f_cell+0][5]=result1[k][v_cell+1][2];
                     result2[k][f_cell+0][6]=result1[k][v_cell+2][0];
                     result2[k][f_cell+0][7]=result1[k][v_cell+2][1];
                     result2[k][f_cell+0][8]=result1[k][v_cell+2][2];

                     result2[k][f_cell+1][0]=result1[k][v_cell+1][0];
                     result2[k][f_cell+1][1]=result1[k][v_cell+1][1];
                     result2[k][f_cell+1][2]=result1[k][v_cell+1][2];
                     result2[k][f_cell+1][3]=result1[k][v_cell+2][0];
                     result2[k][f_cell+1][4]=result1[k][v_cell+2][1];
                     result2[k][f_cell+1][5]=result1[k][v_cell+2][2];
                     result2[k][f_cell+1][6]=result1[k][v_cell+3][0];
                     result2[k][f_cell+1][7]=result1[k][v_cell+3][1];
                     result2[k][f_cell+1][8]=result1[k][v_cell+3][2];

                     v_cell = v_cell + 4;
                     f_cell = f_cell + 2;
                }

            }
        }

    }

}



void get_faces_vertices_cuda(torch::Tensor voxel,int N, int  C, int H
            ,int W,torch::Tensor threshold ,torch::Tensor result1, torch::Tensor result2){

  const int threads = NUM_THREADS;
  const int blocks = (threads/1024)+1;

  AT_DISPATCH_FLOATING_TYPES(voxel.type(), "get_faces_vertices_cuda", ([&] {
    get_faces_vertices_kernel<scalar_t><<<blocks, threads>>>(
        voxel.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
        N,
        C,
        H,
        W,
        threshold.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>(),
        result1.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
        result2.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>());
  }));

}
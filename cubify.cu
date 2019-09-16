#include <iostream>
#include <set>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define NUM_THREADS 1024

struct vertice {
    double x;
    double y;
    double z;
};

struct face {
    vertice v1;
    vertice v2;
    vertice v3;
};

vertice init_vertice(double x, double y, double z) {
    vertice c;
    c.x = x;
    c.y = y;
    c.z = z;
    return c;
}

face init_face(vertice v1, vertice v2, vertice v3) {
    face f;
    f.v1 = v1;
    f.v2 = v2;
    f.v3 = v3;
    return f;
}

__global__
void get_faces_vertices(int ***array, int C, int H, int W, int threshold, std::set<vertice> *result1,
                        std::set<face> *result2) {
    int t = blockDim.x * blockIdx.x + threadIdx.x;
    int per_thread = (C * H * W) / NUM_THREADS;
    t *= per_thread;
    if (t < C * H * W) {
        for (int k = t; k < t + per_thread; k++) {
            int channel = k % C;
            int i = k / C % H;
            int j = k / C / H % W;
            result1[k] = std::set<vertice>();
            result2[k] = std::set<face>();
            if (array[channel][i][j] > threshold) {
                if (channel == 0 || array[channel - 1][i][j] <= threshold) {
                    vertice v0 = init_vertice(channel - 0.5, i - 0.5, j - 0.5);
                    vertice v1 = init_vertice(channel - 0.5, i - 0.5, j + 0.5);
                    vertice v2 = init_vertice(channel - 0.5, i + 0.5, j - 0.5);
                    vertice v3 = init_vertice(channel - 0.5, i + 0.5, j + 0.5);
                    result1[k].insert(v0);
                    result1[k].insert(v1);
                    result1[k].insert(v2);
                    result1[k].insert(v3);
                    face f1 = init_face(v0, v1, v2);
                    face f2 = init_face(v1, v2, v3);
                    result2[k].insert(f1);
                    result2[k].insert(f2);
                }
                if (channel == C - 1 || array[channel + 1][i][j] <= threshold) {
                    vertice v0 = init_vertice(channel + 0.5, i - 0.5, j - 0.5);
                    vertice v1 = init_vertice(channel + 0.5, i - 0.5, j + 0.5);
                    vertice v2 = init_vertice(channel + 0.5, i + 0.5, j - 0.5);
                    vertice v3 = init_vertice(channel + 0.5, i + 0.5, j + 0.5);
                    result1[k].insert(v0);
                    result1[k].insert(v1);
                    result1[k].insert(v2);
                    result1[k].insert(v3);
                    face f1 = init_face(v0, v1, v2);
                    face f2 = init_face(v1, v2, v3);
                    result2[k].insert(f1);
                    result2[k].insert(f2);
                }
                if (i == 0 || array[channel][i - 1][j] <= threshold) {
                    vertice v0 = init_vertice(channel + 0.5, i - 0.5, j - 0.5);
                    vertice v1 = init_vertice(channel + 0.5, i - 0.5, j + 0.5);
                    vertice v2 = init_vertice(channel - 0.5, i - 0.5, j - 0.5);
                    vertice v3 = init_vertice(channel - 0.5, i - 0.5, j + 0.5);
                    result1[k].insert(v0);
                    result1[k].insert(v1);
                    result1[k].insert(v2);
                    result1[k].insert(v3);
                    face f1 = init_face(v0, v1, v2);
                    face f2 = init_face(v1, v2, v3);
                    result2[k].insert(f1);
                    result2[k].insert(f2);
                }
                if (i == H - 1 || array[channel][i + 1][j] <= threshold) {
                    vertice v0 = init_vertice(channel - 0.5, i + 0.5, j - 0.5);
                    vertice v1 = init_vertice(channel - 0.5, i + 0.5, j + 0.5);
                    vertice v2 = init_vertice(channel + 0.5, i + 0.5, j - 0.5);
                    vertice v3 = init_vertice(channel + 0.5, i + 0.5, j + 0.5);
                    result1[k].insert(v0);
                    result1[k].insert(v1);
                    result1[k].insert(v2);
                    result1[k].insert(v3);
                    face f1 = init_face(v0, v1, v2);
                    face f2 = init_face(v1, v2, v3);
                    result2[k].insert(f1);
                    result2[k].insert(f2);
                }
                if (j == 0 || array[channel][i][j - 1] <= threshold) {
                    vertice v0 = init_vertice(channel + 0.5, i - 0.5, j - 0.5);
                    vertice v1 = init_vertice(channel - 0.5, i - 0.5, j - 0.5);
                    vertice v2 = init_vertice(channel + 0.5, i + 0.5, j - 0.5);
                    vertice v3 = init_vertice(channel - 0.5, i + 0.5, j - 0.5);
                    result1[k].insert(v0);
                    result1[k].insert(v1);
                    result1[k].insert(v2);
                    result1[k].insert(v3);
                    face f1 = init_face(v0, v1, v2);
                    face f2 = init_face(v1, v2, v3);
                    result2[k].insert(f1);
                    result2[k].insert(f2);
                }
                if (j == W - 1 || array[channel][i][j + 1] <= threshold) {
                    vertice v0 = init_vertice(channel - 0.5, i - 0.5, j + 0.5);
                    vertice v1 = init_vertice(channel + 0.5, i - 0.5, j + 0.5);
                    vertice v2 = init_vertice(channel - 0.5, i + 0.5, j + 0.5);
                    vertice v3 = init_vertice(channel + 0.5, i + 0.5, j + 0.5);
                    result1[k].insert(v0);
                    result1[k].insert(v1);
                    result1[k].insert(v2);
                    result1[k].insert(v3);
                    face f1 = init_face(v0, v1, v2);
                    face f2 = init_face(v1, v2, v3);
                    result2[k].insert(f1);
                    result2[k].insert(f2);
                }

            }
        }

    }

}



#include<iostream>
#include<cuda_runtime.h>
#include<vector>


__global__
void rgb_to_gray_scale_conversion_kernel(float *Pin, float *Pout, int height, int width){
    int CHANNELS = 3;
    
    int row = blockIdx.x*blockDim.x + threadIdx.x;
    int col = blockIdx.y*blockDim.y + threadIdx.y;

    if (row < height && col < width) {
        int gray_offset = row*width + col;
        int rgb_offset = CHANNELS*gray_offset;

        float r = Pin[rgb_offset];
        float g = Pin[rgb_offset+1];
        float b = Pin[rgb_offset+2];

        Pout[gray_offset] = 0.21f*r + 0.71f*g + 0.08f*b;
    }

}

int main(){
    int ROW = 16;
    int COL = 16;
    int CHANNELS = 3;

    // create vectors : rgb image - Pin_h, Pout_h

    float *Pin_h, *Pout_h;

    Pin_h = new float[ROW*COL*3];
    Pout_h = new float[ROW*COL];

    for(int i=0; i<ROW; i++){
        for(int j=0; j<COL; j++){
            for(int channel=0; channel<3; channel++){
                Pin_h[i*COL*CHANNELS + j*CHANNELS + channel] = 3.0; // 3 can be replaced by a random number between 0 to 255
            }
        } 
    }


    // create pointer for output image - Pout_d and input image Pin_d on device side
    float *Pin_d, *Pout_d;

    // allocate memory on gpu for Pin_d and Pout_d
    int rgb_size = sizeof(float)*ROW*COL*3;
    int gray_size = sizeof(float)*ROW*COL;

    cudaMalloc((void **) &Pin_d, rgb_size);
    cudaMalloc((void **) &Pout_d, gray_size);

    // Copy input vector(s) to gpu: Pin_d -> source Pin_h
    cudaMemcpy(Pin_d, Pin_h, rgb_size, cudaMemcpyHostToDevice);

    // call the gpu kernel function
    dim3 grid_dim (std::ceil(ROW/16.0), std::ceil(COL/16.0), 1);
    dim3 block_dim (16, 16, 1);

    rgb_to_gray_scale_conversion_kernel<<<grid_dim, block_dim>>>(Pin_d, Pout_d, ROW, COL);

    // copy Pout_d from gpu to host Pout_h
    cudaMemcpy(Pout_h, Pout_d, gray_size, cudaMemcpyDeviceToHost);

    //  free memory in gpu Pout_d and Pin_d
    cudaFree(Pin_d);
    cudaFree(Pout_d);

    for(int idx=0; idx < ROW*COL; idx++){
        std::cout<<"Pout["<<idx<<"] = "<<Pout_h[idx]<<std::endl;
    }

    delete[] Pin_h;
    delete[] Pout_h;

    return 0;
}

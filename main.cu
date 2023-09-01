
#include<iostream>
#include "cuda_runtime.h"
#include<string>
#include<vector>
#include<fstream>
#include <cuda.h>
#include "device_launch_parameters.h"
#include <cuda_runtime_api.h>
#include <device_functions.h>

# define __syncthreads()
# define PI  3.14159265358979323846

static void HandleError(cudaError_t err, const char* file, int line) {
	if (err != cudaSuccess) {
		std::cout << cudaGetErrorString(err) << "in" << file << "at line" << line;
	}
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ )) 



__global__ void Eout_Ein_calculation(float* Eo_gpu, float* Ei_gpu, float* fo_gpu, float* fi_gpu, float* input_img_gpu, float* input, float* fout, float* fin, int img_w, int img_h, int img_l, int sigma) {

	
	size_t i = blockDim.y * blockIdx.y + threadIdx.y;	// calculate row index, point to the output  //width 
	size_t j = blockDim.x * blockIdx.x + threadIdx.x;	// calculate column index, point to the output //height
	size_t p = blockDim.z * blockIdx.z + threadIdx.z;
	if (i >= img_h || j >= img_w || p >= img_l) return;

	float s1 = 0;
	float s2 = 0;

	for (int u = -sigma; u <= sigma; u++)
	{
		for (int v = -sigma; v <= sigma; v++)
		{
			for (int w = -sigma; w <= sigma; w++)
			{
				if (((i + u) >= 0) && ((i + u) < img_h) && ((j + v) >= 0) && ((j + v) < img_w) && ((p + w) >= 0) && ((p + w) < img_l))
				{
					s1 = (s1 + ((1 / pow(((2 * sigma) + 1), 2)) * (input_img_gpu[((p + w) * img_w * img_h) + ((i + u) * img_w) + (j + v)] - fo_gpu[(p * img_w * img_h) + i * img_w + j]) * (input_img_gpu[((p + w) * img_w * img_h) + ((i + u) * img_w) + (j + v)] - fo_gpu[(p * img_w * img_h) + i * img_w + j])));
					s2 = (s2 + ((1 / pow(((2 * sigma) + 1), 2)) * (input_img_gpu[((p + w) * img_w * img_h) + ((i + u) * img_w) + (j + v)] - fi_gpu[(p * img_w * img_h) + i * img_w + j]) * (input_img_gpu[((p + w) * img_w * img_h) + ((i + u) * img_w) + (j + v)] - fi_gpu[(p * img_w * img_h) + i * img_w + j])));
				}
			}
		}
	}
	Eo_gpu[(p * img_w * img_h) + i * img_w + j] = s1;
	Ei_gpu[(p * img_w * img_h) + i * img_w + j] = s2;

	

}

__global__ void gradient_dx_cuda(float* gradient_dx, float* output, int img_w, int img_h, int img_l) {
	size_t i = blockDim.y * blockIdx.y + threadIdx.y;
	size_t j = blockDim.x * blockIdx.x + threadIdx.x;
	size_t p = blockDim.z * blockIdx.z + threadIdx.z;

	if (i >= img_h || j >= img_w || p >= img_l) return;

	size_t j_left = j - 1;
	if (j_left < 0) {
		j_left = 0;
	}
	size_t j_right = j + 1;
	if (j_right >= img_w) {
		j_right = img_w - 1;
	}

	double dist_grad_left = gradient_dx[(p * img_h * img_w) + (i * img_w) + j_left];
	double dist_grad_right = gradient_dx[(p * img_h * img_w) + (i * img_w) + j_right];

	double dist_grad = (dist_grad_right - dist_grad_left) / 2.0;

	output[(p * img_h * img_w) + (i * img_w) + j] = dist_grad;
}


__global__ void gradient_dy_cuda(float* gradient_dx, float* output, int img_w, int img_h, int img_l) {
	// Calculate gradient along dy (3D)
	int i = blockDim.y * blockIdx.y + threadIdx.y; // Calculate row index (height)
	int j = blockDim.x * blockIdx.x + threadIdx.x; // Calculate column index (width)
	int p = blockDim.z * blockIdx.z + threadIdx.z;

	if (i >= img_h || j >= img_w || p >= img_l) return;

	int i_left = i - 1;
	int i_right = i + 1;

	if (i_left < 0) {
		i_left = 0;
		i_right = 1;
	}
	else if (i_right >= img_h) {
		i_right = img_h - 1;
		i_left = i_right - 1;
	}

	double dist_grad = (gradient_dx[(p * img_h * img_w) + (i_right * img_w) + j] - gradient_dx[(p * img_h * img_w) + (i_left * img_w) + j]) / 2.0f;

	output[(p * img_h * img_w) + (i * img_w) + j] = dist_grad;


}

__global__ void gradient_dz_cuda(float* gradient_dx, float* output, int img_w, int img_h, int img_l) {
	size_t i = blockDim.y * blockIdx.y + threadIdx.y;
	size_t j = blockDim.x * blockIdx.x + threadIdx.x;
	size_t p = blockDim.z * blockIdx.z + threadIdx.z;

	if (i >= img_h || j >= img_w || p >= img_l) return;

	size_t p_left = (p > 0) ? p - 1 : 0;
	size_t p_right = (p < img_l - 1) ? p + 1 : img_l - 1;

	double dist_grad_left = gradient_dx[(p_left * img_h * img_w) + (i * img_w) + j];
	double dist_grad_right = gradient_dx[(p_right * img_h * img_w) + (i * img_w) + j];

	double dist_grad = (dist_grad_right - dist_grad_left) / 2.0;

	output[(p * img_h * img_w) + (i * img_w) + j] = dist_grad;
}

void gradient_cal(float* input, float* output_x, float* output_y, float* output_z, int img_w, int img_h, int img_l) {


	cudaDeviceProp props;
	HANDLE_ERROR(cudaGetDeviceProperties(&props, 0));


	float* input_gpu;
	float* output_x_gpu;
	float* output_y_gpu;
	float* output_z_gpu;
	
	size_t bytes = (img_w * img_h * img_l) * sizeof(float);
	HANDLE_ERROR(cudaMalloc(&input_gpu, bytes));  							    //allocate memory on device
	HANDLE_ERROR(cudaMalloc(&output_x_gpu, bytes));  							//allocate memory on device
	HANDLE_ERROR(cudaMalloc(&output_y_gpu, bytes));  							//allocate memory on device
	HANDLE_ERROR(cudaMalloc(&output_z_gpu, bytes));  							//allocate memory on device
		

	HANDLE_ERROR(cudaMemcpy(input_gpu, input, bytes, cudaMemcpyHostToDevice));     //copy the array from main memory to device


	size_t blockDim = sqrt(props.maxThreadsPerBlock);
	dim3 threads(blockDim, blockDim);
	dim3 blocks(img_w / threads.x + 1, img_h / threads.y + 1, img_l / threads.z + 1);

	gradient_dx_cuda << < blocks, threads >> > (input_gpu, output_x_gpu,  img_w, img_h, img_l);
	gradient_dy_cuda << < blocks, threads >> > (input_gpu, output_y_gpu, img_w, img_h, img_l);
	gradient_dz_cuda << < blocks, threads >> > (input_gpu, output_z_gpu, img_w, img_h, img_l);

	

	HANDLE_ERROR(cudaMemcpy(output_x, output_x_gpu, bytes, cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(output_y, output_y_gpu, bytes, cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(output_z, output_z_gpu, bytes, cudaMemcpyDeviceToHost));


	cudaFree(input_gpu);
	cudaFree(output_x_gpu);
	cudaFree(output_y_gpu);
	cudaFree(output_z_gpu);


}


__global__ void gradient_dx_cuda_shared(float* gradient_dx, float* output, int img_w, int img_h, int img_l) {
	extern __shared__ float shared_gradient_dx[];

	size_t i = blockDim.y * blockIdx.y + threadIdx.y;
	size_t j = blockDim.x * blockIdx.x + threadIdx.x;
	size_t p = blockDim.z * blockIdx.z + threadIdx.z;

	if (i >= img_h || j >= img_w || p >= img_l) return;

	
	size_t shared_idx = threadIdx.z * blockDim.y * blockDim.x + threadIdx.y * blockDim.x + threadIdx.x;
	size_t global_idx = (p * img_h * img_w) + (i * img_w) + j;
	shared_gradient_dx[shared_idx] = gradient_dx[global_idx];
	__syncthreads();

	size_t j_left = threadIdx.x - 1;
	size_t j_right = threadIdx.x + 1;

	
	if (threadIdx.x == 0) {
		j_left = 0;
	}
	if (threadIdx.x == blockDim.x - 1) {
		j_right = blockDim.x - 1;
	}


	float dist_grad_left = shared_gradient_dx[threadIdx.z * blockDim.y * blockDim.x + threadIdx.y * blockDim.x + j_left];
	float dist_grad_right = shared_gradient_dx[threadIdx.z * blockDim.y * blockDim.x + threadIdx.y * blockDim.x + j_right];

	float dist_grad = (dist_grad_right - dist_grad_left) / 2.0;

	
	output[global_idx] = dist_grad;
}


__global__ void gradient_dy_cuda_shared(float* gradient_dy, float* output, int img_w, int img_h, int img_l) {
	extern __shared__ float shared_gradient_dy[];

	size_t i = blockDim.y * blockIdx.y + threadIdx.y;
	size_t j = blockDim.x * blockIdx.x + threadIdx.x;
	size_t p = blockDim.z * blockIdx.z + threadIdx.z;

	if (i >= img_h || j >= img_w || p >= img_l) return;

	size_t shared_idx = threadIdx.z * blockDim.y * blockDim.x + threadIdx.y * blockDim.x + threadIdx.x;
	size_t global_idx = (p * img_h * img_w) + (i * img_w) + j;
	shared_gradient_dy[shared_idx] = gradient_dy[global_idx];
	__syncthreads();

	size_t i_above = threadIdx.y - 1;
	size_t i_below = threadIdx.y + 1;

	if (threadIdx.y == 0) {
		i_above = 0;
	}
	if (threadIdx.y == blockDim.y - 1) {
		i_below = blockDim.y - 1;
	}

	size_t idx_above = (threadIdx.z * blockDim.y + i_above) * blockDim.x + threadIdx.x;
	size_t idx_below = (threadIdx.z * blockDim.y + i_below) * blockDim.x + threadIdx.x;

	float dist_grad_above = shared_gradient_dy[idx_above];
	float dist_grad_below = shared_gradient_dy[idx_below];

	float dist_grad = (dist_grad_below - dist_grad_above) / 2.0;

	output[global_idx] = dist_grad;

}

__global__ void gradient_dz_cuda_shared(float* gradient_dz, float* output, int img_w, int img_h, int img_l) {
	extern __shared__ float shared_gradient_dz[];

	size_t i = blockDim.y * blockIdx.y + threadIdx.y;
	size_t j = blockDim.x * blockIdx.x + threadIdx.x;
	size_t p = blockDim.z * blockIdx.z + threadIdx.z;

	if (i >= img_h || j >= img_w || p >= img_l) return;

	size_t shared_idx = threadIdx.z * blockDim.y * blockDim.x + threadIdx.y * blockDim.x + threadIdx.x;
	size_t global_idx = (p * img_h * img_w) + (i * img_w) + j;
	shared_gradient_dz[shared_idx] = gradient_dz[global_idx];
	__syncthreads();

	size_t p_front = threadIdx.z - 1;
	size_t p_back = threadIdx.z + 1;

	if (threadIdx.z == 0) {
		p_front = 0;
	}
	if (threadIdx.z == blockDim.z - 1) {
		p_back = blockDim.z - 1;
	}

	float dist_grad_front = shared_gradient_dz[p_front * blockDim.y * blockDim.x + threadIdx.y * blockDim.x + threadIdx.x];
	float dist_grad_back = shared_gradient_dz[p_back * blockDim.y * blockDim.x + threadIdx.y * blockDim.x + threadIdx.x];

	float dist_grad = (dist_grad_back - dist_grad_front) / 2.0;

	output[global_idx] = dist_grad;
}

void gradient_cal_shared(float* input, float* output_x, float* output_y, float* output_z, int img_w, int img_h, int img_l) {


	cudaDeviceProp props;
	HANDLE_ERROR(cudaGetDeviceProperties(&props, 0));


	float* input_gpu;
	float* output_x_gpu;
	float* output_y_gpu;
	float* output_z_gpu;

	size_t bytes = (img_w * img_h * img_l) * sizeof(float);
	HANDLE_ERROR(cudaMalloc(&input_gpu, bytes));  							    //allocate memory on device
	HANDLE_ERROR(cudaMalloc(&output_x_gpu, bytes));  							//allocate memory on device
	HANDLE_ERROR(cudaMalloc(&output_y_gpu, bytes));  							//allocate memory on device
	HANDLE_ERROR(cudaMalloc(&output_z_gpu, bytes));  							//allocate memory on device


	HANDLE_ERROR(cudaMemcpy(input_gpu, input, bytes, cudaMemcpyHostToDevice));     //copy the array from main memory to device


	size_t blockDim = sqrt(props.maxThreadsPerBlock);
	dim3 threads(blockDim, blockDim);
	dim3 blocks(img_w / threads.x + 1, img_h / threads.y + 1, img_l / threads.z + 1);

	// calculate the size of shared memory
	size_t sharedmemory = blockDim * blockDim  * sizeof(float);
	if (props.sharedMemPerBlock < sharedmemory) {
		std::cout << "ERROR:  shared memory is insufficient " << std::endl;
		exit(1);
	}

	gradient_dx_cuda_shared << < blocks, threads, sharedmemory >> > (input_gpu, output_x_gpu, img_w, img_h, img_l);
	gradient_dy_cuda_shared << < blocks, threads, sharedmemory >> > (input_gpu, output_y_gpu, img_w, img_h, img_l);
	gradient_dz_cuda_shared << < blocks, threads, sharedmemory >> > (input_gpu, output_z_gpu, img_w, img_h, img_l);



	HANDLE_ERROR(cudaMemcpy(output_x, output_x_gpu, bytes, cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(output_y, output_y_gpu, bytes, cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(output_z, output_z_gpu, bytes, cudaMemcpyDeviceToHost));


	cudaFree(input_gpu);
	cudaFree(output_x_gpu);
	cudaFree(output_y_gpu);
	cudaFree(output_z_gpu);


}

__global__ void division_grad(float* output, float* input_1, float* input_2, int img_w, int img_h, int img_l) {


	size_t i = blockDim.y * blockIdx.y + threadIdx.y;	// calculate row index, point to the output  //width 
	size_t j = blockDim.x * blockIdx.x + threadIdx.x;	// calculate column index, point to the output //height
	size_t p = blockDim.z * blockIdx.z + threadIdx.z;
	if (i >= img_h || j >= img_w || p >= img_l) return;

	size_t index = i * img_w * img_l + j * img_l + p;
	float gradient_x = input_1[index];
	float gradient_y = input_2[index];
	
	float final = gradient_x / (gradient_y + 1e-6);
	output[index] = final;


}

__global__ void division(float* output, float* input_1, float* input_2, int img_w, int img_h, int img_l) {


	size_t i = blockDim.y * blockIdx.y + threadIdx.y;	// calculate row index, point to the output  //width 
	size_t j = blockDim.x * blockIdx.x + threadIdx.x;	// calculate column index, point to the output //height
	size_t p = blockDim.z * blockIdx.z + threadIdx.z;
	if (i >= img_h || j >= img_w || p >= img_l) return;

	size_t index = i * img_w * img_l + j * img_l + p;
	float gradient_x = input_1[index];
	float gradient_y = input_2[index];

	float final = gradient_x / (gradient_y);
	output[index] = final;


}


__global__ void multiplication(float* output, float* input_1, float* input_2, int img_w, int img_h, int img_l) {


	size_t i = blockDim.y * blockIdx.y + threadIdx.y;	// calculate row index, point to the output  //width 
	size_t j = blockDim.x * blockIdx.x + threadIdx.x;	// calculate column index, point to the output //height
	size_t p = blockDim.z * blockIdx.z + threadIdx.z;
	if (i >= img_h || j >= img_w || p >= img_l) return;

	size_t index = i * img_w * img_l + j * img_l + p;
	float gradient_x = input_1[index];
	float gradient_y = input_2[index];

	float final = gradient_x * (gradient_y);
	output[index] = final;


}

__global__ void plus(float* output, float* input_1, float* input_2, float* input_3, int img_w, int img_h, int img_l) {


	
	size_t i = blockDim.y * blockIdx.y + threadIdx.y;	// calculate row index, point to the output  //width 
	size_t j = blockDim.x * blockIdx.x + threadIdx.x;	// calculate column index, point to the output //height
	size_t p = blockDim.z * blockIdx.z + threadIdx.z;
	if (i >= img_h || j >= img_w || p >= img_l) return;

	size_t index = i * img_w * img_l + j * img_l + p;
	float gradient_x = input_1[index];
	float gradient_y = input_2[index];
	float gradient_z = input_3[index];

	float final = gradient_x + gradient_y + gradient_z;
	output[index] = final;

}


__global__ void heaviside(float* output, float* input, int img_w, int img_h, int img_l) {



	size_t i = blockDim.y * blockIdx.y + threadIdx.y;	// calculate row index, point to the output  //width 
	size_t j = blockDim.x * blockIdx.x + threadIdx.x;	// calculate column index, point to the output //height
	size_t p = blockDim.z * blockIdx.z + threadIdx.z;
	if (i >= img_h || j >= img_w || p >= img_l) return;

	size_t index = i * img_w * img_l + j * img_l + p;
	float epsilon = 0.2;
	float heaviside = (0.5 * (1 + (2 / 3.14159265358979323846) * atan(input[index] / epsilon)));
	
	output[index] = heaviside;

}

__global__ void deri_heaviside(float* output, float* input, int img_w, int img_h, int img_l) {



	size_t i = blockDim.y * blockIdx.y + threadIdx.y;	// calculate row index, point to the output  //width 
	size_t j = blockDim.x * blockIdx.x + threadIdx.x;	// calculate column index, point to the output //height
	size_t p = blockDim.z * blockIdx.z + threadIdx.z;
	if (i >= img_h || j >= img_w || p >= img_l) return;

	size_t index = i * img_w * img_l + j * img_l + p;
	float epsilon = 0.2;
	float deri_heaviside = (1 / 3.14159265358979323846) * (epsilon / ((epsilon * epsilon) + (input[index] * input[index])));

	output[index] = deri_heaviside;

}

//__global__ void border_padding(float* output, float* input, int w, int img_w, int img_h, int img_l) {
//
//	size_t i = blockIdx.y * blockDim.y + threadIdx.y;
//	size_t j = blockIdx.x * blockDim.x + threadIdx.x;
//	size_t p = blockIdx.z * blockDim.z + threadIdx.z;
//
//	if (i >= img_h + 2 * w || j >= img_w + 2 * w || p >= img_l + 2 * w) return;
//
//	if (i < w || j < w || p < w || i >= img_h + w || j >= img_w + w || p >= img_l + w) {
//		size_t n = i * (img_w + 2 * w) * (img_l + 2 * w) + j * (img_l + 2 * w) + p;
//		output[n] = 30.0f;
//	}
//	else {
//		size_t index = (i - w) * img_w * img_l + (j - w) * img_l + (p - w);
//		size_t n = i * (img_w + 2 * w) * (img_l + 2 * w) + j * (img_l + 2 * w) + p;
//		output[n] = input[index];
//	}
//
//}


//void border_padding_cal(float* output, float* input, int w, int img_w, int img_h, int img_l) {
//
//
//	cudaDeviceProp props;
//	HANDLE_ERROR(cudaGetDeviceProperties(&props, 0));
//
//
//	float* output_gpu;
//	float* input_gpu;
//
//
//	size_t bytes_in = (img_w * img_h * img_l) * sizeof(float);
//	size_t bytes_out = ((img_w +(w * 2)) * (img_h + (w * 2)) * (img_l + (w * 2))) * sizeof(float);
//	HANDLE_ERROR(cudaMalloc(&output_gpu, bytes_out));  							    //allocate memory on 
//	HANDLE_ERROR(cudaMalloc(&input_gpu, bytes_in));  							    //allocate memory on device
//
//	HANDLE_ERROR(cudaMemcpy(input_gpu, input, bytes_in, cudaMemcpyHostToDevice));     //copy the array from main memory to device
//
//	size_t blockDim = sqrt(props.maxThreadsPerBlock);
//	dim3 threads(blockDim, blockDim);
//	dim3 blocks(img_w / threads.x + 1, img_h / threads.y + 1, img_l / threads.z + 1);
//
//	border_padding << < blocks, threads >> > (output_gpu, input_gpu, w , img_w, img_h, img_l);
//
//	HANDLE_ERROR(cudaMemcpy(output, output_gpu, bytes_out, cudaMemcpyDeviceToHost));
//
//	cudaFree(output_gpu);
//	cudaFree(input_gpu);
//
//
//}

void division_cal(float* output, float* input_1, float* input_2, int img_w, int img_h, int img_l) {


	cudaDeviceProp props;
	HANDLE_ERROR(cudaGetDeviceProperties(&props, 0));


	float* output_gpu;
	float* input_1_gpu;
	float* input_2_gpu;
	


	size_t bytes = (img_w * img_h * img_l) * sizeof(float);
	HANDLE_ERROR(cudaMalloc(&output_gpu, bytes));  							    //allocate memory on 
	HANDLE_ERROR(cudaMalloc(&input_1_gpu, bytes));  							    //allocate memory on device
	HANDLE_ERROR(cudaMalloc(&input_2_gpu, bytes));  							    //allocate memory on device
	
	
	HANDLE_ERROR(cudaMemcpy(input_1_gpu, input_1, bytes, cudaMemcpyHostToDevice));     //copy the array from main memory to device
	HANDLE_ERROR(cudaMemcpy(input_2_gpu, input_2, bytes, cudaMemcpyHostToDevice));     //copy the array from main memory to device


	size_t blockDim = sqrt(props.maxThreadsPerBlock);
	dim3 threads(blockDim, blockDim);
	dim3 blocks(img_w / threads.x + 1, img_h / threads.y + 1, img_l / threads.z + 1);

	division << < blocks, threads >> > (output_gpu, input_1_gpu, input_2_gpu, img_w, img_h, img_l);
	
	HANDLE_ERROR(cudaMemcpy(output, output_gpu, bytes, cudaMemcpyDeviceToHost));
	
	cudaFree(output_gpu);
	cudaFree(input_1_gpu);
	cudaFree(input_2_gpu);



}


void heaviside_cal(float* output, float* input, int img_w, int img_h, int img_l) {


	cudaDeviceProp props;
	HANDLE_ERROR(cudaGetDeviceProperties(&props, 0));


	float* output_gpu;
	float* input_gpu;
	

	size_t bytes = (img_w * img_h * img_l) * sizeof(float);
	HANDLE_ERROR(cudaMalloc(&output_gpu, bytes));  							    //allocate memory on 
	HANDLE_ERROR(cudaMalloc(&input_gpu, bytes));  							    //allocate memory on device
	
	HANDLE_ERROR(cudaMemcpy(input_gpu, input, bytes, cudaMemcpyHostToDevice));     //copy the array from main memory to device
	
	size_t blockDim = sqrt(props.maxThreadsPerBlock);
	dim3 threads(blockDim, blockDim);
	dim3 blocks(img_w / threads.x + 1, img_h / threads.y + 1, img_l / threads.z + 1);

	heaviside << < blocks, threads >> > (output_gpu, input_gpu, img_w, img_h, img_l);

	HANDLE_ERROR(cudaMemcpy(output, output_gpu, bytes, cudaMemcpyDeviceToHost));

	cudaFree(output_gpu);
	cudaFree(input_gpu);
	

}

void deri_heaviside_cal(float* output, float* input, int img_w, int img_h, int img_l) {


	cudaDeviceProp props;
	HANDLE_ERROR(cudaGetDeviceProperties(&props, 0));


	float* output_gpu;
	float* input_gpu;


	size_t bytes = (img_w * img_h * img_l) * sizeof(float);
	HANDLE_ERROR(cudaMalloc(&output_gpu, bytes));  							    //allocate memory on 
	HANDLE_ERROR(cudaMalloc(&input_gpu, bytes));  							    //allocate memory on device

	HANDLE_ERROR(cudaMemcpy(input_gpu, input, bytes, cudaMemcpyHostToDevice));     //copy the array from main memory to device

	size_t blockDim = sqrt(props.maxThreadsPerBlock);
	dim3 threads(blockDim, blockDim);
	dim3 blocks(img_w / threads.x + 1, img_h / threads.y + 1, img_l / threads.z + 1);

	deri_heaviside << < blocks, threads >> > (output_gpu, input_gpu, img_w, img_h, img_l);

	HANDLE_ERROR(cudaMemcpy(output, output_gpu, bytes, cudaMemcpyDeviceToHost));

	cudaFree(output_gpu);
	cudaFree(input_gpu);


}

void multiplication_cal(float* output, float* input_1, float* input_2, int img_w, int img_h, int img_l) {


	cudaDeviceProp props;
	HANDLE_ERROR(cudaGetDeviceProperties(&props, 0));


	float* output_gpu;
	float* input_1_gpu;
	float* input_2_gpu;



	size_t bytes = (img_w * img_h * img_l) * sizeof(float);
	HANDLE_ERROR(cudaMalloc(&output_gpu, bytes));  							    //allocate memory on 
	HANDLE_ERROR(cudaMalloc(&input_1_gpu, bytes));  							    //allocate memory on device
	HANDLE_ERROR(cudaMalloc(&input_2_gpu, bytes));  							    //allocate memory on device


	HANDLE_ERROR(cudaMemcpy(input_1_gpu, input_1, bytes, cudaMemcpyHostToDevice));     //copy the array from main memory to device
	HANDLE_ERROR(cudaMemcpy(input_2_gpu, input_2, bytes, cudaMemcpyHostToDevice));     //copy the array from main memory to device


	size_t blockDim = sqrt(props.maxThreadsPerBlock);
	dim3 threads(blockDim, blockDim);
	dim3 blocks(img_w / threads.x + 1, img_h / threads.y + 1, img_l / threads.z + 1);

	multiplication << < blocks, threads >> > (output_gpu, input_1_gpu, input_2_gpu, img_w, img_h, img_l);

	HANDLE_ERROR(cudaMemcpy(output, output_gpu, bytes, cudaMemcpyDeviceToHost));

	cudaFree(output_gpu);
	cudaFree(input_1_gpu);
	cudaFree(input_2_gpu);



}

void DIV_cal( float* DIV, float* input_x, float* input_y, float* input_z, int img_w, int img_h, int img_l) {


	cudaDeviceProp props;
	HANDLE_ERROR(cudaGetDeviceProperties(&props, 0));


	float* input_x_gpu;
	float* input_y_gpu;
	float* input_z_gpu;
	float* output_x_gpu;
	float* output_y_gpu;
	float* output_z_gpu;
	float* DIV_gpu;

	size_t bytes = (img_w * img_h * img_l) * sizeof(float);
	HANDLE_ERROR(cudaMalloc(&input_x_gpu, bytes));  							    //allocate memory on 
	HANDLE_ERROR(cudaMalloc(&input_y_gpu, bytes));  							    //allocate memory on device
	HANDLE_ERROR(cudaMalloc(&input_z_gpu, bytes));  							    //allocate memory on device
	HANDLE_ERROR(cudaMalloc(&output_x_gpu, bytes));  							//allocate memory on device
	HANDLE_ERROR(cudaMalloc(&output_y_gpu, bytes));  							//allocate memory on device
	HANDLE_ERROR(cudaMalloc(&output_z_gpu, bytes));  							//allocate memory on device
	HANDLE_ERROR(cudaMalloc(&DIV_gpu, bytes));  							//allocate memory on device


	HANDLE_ERROR(cudaMemcpy(input_x_gpu, input_x, bytes, cudaMemcpyHostToDevice));     //copy the array from main memory to 
	HANDLE_ERROR(cudaMemcpy(input_y_gpu, input_y, bytes, cudaMemcpyHostToDevice));     //copy the array from main memory to device
	HANDLE_ERROR(cudaMemcpy(input_z_gpu, input_z, bytes, cudaMemcpyHostToDevice));     //copy the array from main memory to device


	size_t blockDim = sqrt(props.maxThreadsPerBlock);
	dim3 threads(blockDim, blockDim);
	dim3 blocks(img_w / threads.x + 1, img_h / threads.y + 1, img_l / threads.z + 1);

	gradient_dx_cuda << < blocks, threads >> > (input_x_gpu, output_x_gpu, img_w, img_h, img_l);
	gradient_dy_cuda << < blocks, threads >> > (input_y_gpu, output_y_gpu, img_w, img_h, img_l);
	gradient_dz_cuda << < blocks, threads >> > (input_z_gpu, output_z_gpu, img_w, img_h, img_l);
	plus << < blocks, threads >> > (DIV_gpu, output_x_gpu, output_y_gpu, output_z_gpu, img_w, img_h, img_l);


	HANDLE_ERROR(cudaMemcpy(DIV, DIV_gpu, bytes, cudaMemcpyDeviceToHost));
	

	cudaFree(DIV_gpu);
	cudaFree(output_x_gpu);
	cudaFree(output_y_gpu);
	cudaFree(output_z_gpu);
	cudaFree(input_x_gpu);
	cudaFree(input_y_gpu);
	cudaFree(input_z_gpu);
	


}


void division_nDIV( float* input_x, float* input_y, float* input_z, float* gradientmagnitude, float* output_x, float* output_y, float* output_z, int img_w, int img_h, int img_l) {


	cudaDeviceProp props;
	HANDLE_ERROR(cudaGetDeviceProperties(&props, 0));


	float* input_x_gpu;
	float* input_y_gpu;
	float* input_z_gpu;
	float* output_x_gpu;
	float* output_y_gpu;
	float* output_z_gpu;
	float* gradientmagnitude_gpu;
	

	size_t bytes = (img_w * img_h * img_l) * sizeof(float);
	HANDLE_ERROR(cudaMalloc(&input_x_gpu, bytes));  							    //allocate memory on 
	HANDLE_ERROR(cudaMalloc(&input_y_gpu, bytes));  							    //allocate memory on device
	HANDLE_ERROR(cudaMalloc(&input_z_gpu, bytes));  							    //allocate memory on device
	HANDLE_ERROR(cudaMalloc(&output_x_gpu, bytes));  							//allocate memory on device
	HANDLE_ERROR(cudaMalloc(&output_y_gpu, bytes));  							//allocate memory on device
	HANDLE_ERROR(cudaMalloc(&output_z_gpu, bytes));  							//allocate memory on device
	HANDLE_ERROR(cudaMalloc(&gradientmagnitude_gpu, bytes));  							//allocate memory on device


	HANDLE_ERROR(cudaMemcpy(input_x_gpu, input_x, bytes, cudaMemcpyHostToDevice));     //copy the array from main memory to 
	HANDLE_ERROR(cudaMemcpy(input_y_gpu, input_y, bytes, cudaMemcpyHostToDevice));     //copy the array from main memory to device
	HANDLE_ERROR(cudaMemcpy(input_z_gpu, input_z, bytes, cudaMemcpyHostToDevice));     //copy the array from main memory to device
	HANDLE_ERROR(cudaMemcpy(gradientmagnitude_gpu, gradientmagnitude, bytes, cudaMemcpyHostToDevice));     //copy the array from main memory to device


	size_t blockDim = sqrt(props.maxThreadsPerBlock);
	dim3 threads(blockDim, blockDim);
	dim3 blocks(img_w / threads.x + 1, img_h / threads.y + 1, img_l / threads.z + 1);

	division_grad << < blocks, threads >> > (output_x_gpu, input_x_gpu, gradientmagnitude_gpu, img_w, img_h, img_l);
	division_grad << < blocks, threads >> > (output_y_gpu, input_y_gpu, gradientmagnitude_gpu, img_w, img_h, img_l);
	division_grad << < blocks, threads >> > (output_z_gpu, input_z_gpu, gradientmagnitude_gpu, img_w, img_h, img_l);
	
	HANDLE_ERROR(cudaMemcpy(output_x, output_x_gpu, bytes, cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(output_y, output_y_gpu, bytes, cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(output_z, output_z_gpu, bytes, cudaMemcpyDeviceToHost));

	cudaFree(output_x_gpu);
	cudaFree(output_y_gpu);
	cudaFree(output_z_gpu);
	cudaFree(input_x_gpu);
	cudaFree(input_y_gpu);
	cudaFree(input_z_gpu);
	cudaFree(gradientmagnitude_gpu);



}

void normalized_div_plus( float* normalize_DIV , float* input_x, float* input_y, float* input_z, int img_w, int img_h, int img_l) {


	cudaDeviceProp props;
	HANDLE_ERROR(cudaGetDeviceProperties(&props, 0));


	float* input_x_gpu;
	float* input_y_gpu;
	float* input_z_gpu;
	float* normalize_DIV_gpu;

	size_t bytes = (img_w * img_h * img_l) * sizeof(float);
	HANDLE_ERROR(cudaMalloc(&input_x_gpu, bytes));  							    //allocate memory on 
	HANDLE_ERROR(cudaMalloc(&input_y_gpu, bytes));  							    //allocate memory on device
	HANDLE_ERROR(cudaMalloc(&input_z_gpu, bytes));  							    //allocate memory on device
	HANDLE_ERROR(cudaMalloc(&normalize_DIV_gpu, bytes));  							//allocate memory on device


	HANDLE_ERROR(cudaMemcpy(input_x_gpu, input_x, bytes, cudaMemcpyHostToDevice));     //copy the array from main memory to 
	HANDLE_ERROR(cudaMemcpy(input_y_gpu, input_y, bytes, cudaMemcpyHostToDevice));     //copy the array from main memory to device
	HANDLE_ERROR(cudaMemcpy(input_z_gpu, input_z, bytes, cudaMemcpyHostToDevice));     //copy the array from main memory to device


	size_t blockDim = sqrt(props.maxThreadsPerBlock);
	dim3 threads(blockDim, blockDim);
	dim3 blocks(img_w / threads.x + 1, img_h / threads.y + 1, img_l / threads.z + 1);

	plus << < blocks, threads >> > (normalize_DIV_gpu, input_x_gpu, input_y_gpu, input_z_gpu, img_w, img_h, img_l);

	HANDLE_ERROR(cudaMemcpy(normalize_DIV, normalize_DIV_gpu, bytes, cudaMemcpyDeviceToHost));
	

	cudaFree(input_x_gpu);
	cudaFree(input_y_gpu);
	cudaFree(input_z_gpu);
	cudaFree(normalize_DIV_gpu);



}

void normalized_DIV_cal_cross( float* normalize_DIV, float* gradientmagnitude, float* input_x, float* input_y, float* input_z, int img_w, int img_h, int img_l) {


	cudaDeviceProp props;
	HANDLE_ERROR(cudaGetDeviceProperties(&props, 0));


	float* input_x_gpu;
	float* input_y_gpu;
	float* input_z_gpu;
	float* output_x_gpu;
	float* output_y_gpu;
	float* output_z_gpu;
	float* gradientmagnitude_gpu;
	float* output_x_gpu_final;
	float* output_y_gpu_final;
	float* output_z_gpu_final;
	float* normalize_DIV_gpu;

	size_t bytes = (img_w * img_h * img_l) * sizeof(float);
	HANDLE_ERROR(cudaMalloc(&input_x_gpu, bytes));  							    //allocate memory on 
	HANDLE_ERROR(cudaMalloc(&input_y_gpu, bytes));  							    //allocate memory on device
	HANDLE_ERROR(cudaMalloc(&input_z_gpu, bytes));  							    //allocate memory on device
	HANDLE_ERROR(cudaMalloc(&output_x_gpu, bytes));  							//allocate memory on device
	HANDLE_ERROR(cudaMalloc(&output_y_gpu, bytes));  							//allocate memory on device
	HANDLE_ERROR(cudaMalloc(&output_z_gpu, bytes));  							//allocate memory on device
	HANDLE_ERROR(cudaMalloc(&gradientmagnitude_gpu, bytes));  							//allocate memory on device
	HANDLE_ERROR(cudaMalloc(&output_x_gpu_final, bytes));  							//allocate memory on device
	HANDLE_ERROR(cudaMalloc(&output_y_gpu_final, bytes));  							//allocate memory on device
	HANDLE_ERROR(cudaMalloc(&output_z_gpu_final, bytes));  							//allocate memory on device
	HANDLE_ERROR(cudaMalloc(&normalize_DIV_gpu, bytes));  							//allocate memory on device


	HANDLE_ERROR(cudaMemcpy(input_x_gpu, input_x, bytes, cudaMemcpyHostToDevice));     //copy the array from main memory to 
	HANDLE_ERROR(cudaMemcpy(input_y_gpu, input_y, bytes, cudaMemcpyHostToDevice));     //copy the array from main memory to device
	HANDLE_ERROR(cudaMemcpy(input_z_gpu, input_z, bytes, cudaMemcpyHostToDevice));     //copy the array from main memory to device
	HANDLE_ERROR(cudaMemcpy(gradientmagnitude_gpu, gradientmagnitude, bytes, cudaMemcpyHostToDevice));     //copy the array from main memory to device


	size_t blockDim = sqrt(props.maxThreadsPerBlock);
	dim3 threads(blockDim, blockDim);
	dim3 blocks(img_w / threads.x + 1, img_h / threads.y + 1, img_l / threads.z + 1);

	//gradientmagnitude << < blocks, threads >> > (gradientmagnitude_gpu, input_x_gpu, input_y_gpu, input_z_gpu, img_w, img_h, img_l);
	division_grad << < blocks, threads >> > (output_x_gpu, input_x_gpu, gradientmagnitude_gpu, img_w, img_h, img_l);
	division_grad << < blocks, threads >> > (output_y_gpu, input_y_gpu, gradientmagnitude_gpu, img_w, img_h, img_l);
	division_grad << < blocks, threads >> > (output_z_gpu, input_z_gpu, gradientmagnitude_gpu, img_w, img_h, img_l);
	gradient_dx_cuda << < blocks, threads >> > (output_x_gpu, output_x_gpu_final, img_w, img_h, img_l);
	gradient_dy_cuda << < blocks, threads >> > (output_y_gpu, output_y_gpu_final, img_w, img_h, img_l);
	gradient_dz_cuda << < blocks, threads >> > (output_z_gpu, output_z_gpu_final, img_w, img_h, img_l);
	plus << < blocks, threads >> > (normalize_DIV_gpu, output_x_gpu_final, output_y_gpu_final, output_z_gpu_final, img_w, img_h, img_l);


	HANDLE_ERROR(cudaMemcpy( normalize_DIV, normalize_DIV_gpu, bytes, cudaMemcpyDeviceToHost));
	
	cudaFree(normalize_DIV_gpu);
	cudaFree(gradientmagnitude_gpu);
	cudaFree(output_z_gpu_final);
	cudaFree(output_y_gpu_final);
	cudaFree(output_x_gpu_final);
	cudaFree(output_z_gpu);
	cudaFree(output_y_gpu);
	cudaFree(output_x_gpu);
	cudaFree(input_z_gpu);
	cudaFree(input_y_gpu);
	cudaFree(input_x_gpu);
	


}

void gradient_cal_3_input(float* input_x, float* input_y, float* input_z, float* output_x, float* output_y, float* output_z, int img_w, int img_h, int img_l) {


	cudaDeviceProp props;
	HANDLE_ERROR(cudaGetDeviceProperties(&props, 0));


	float* input_x_gpu;
	float* input_y_gpu;
	float* input_z_gpu;
	float* output_x_gpu;
	float* output_y_gpu;
	float* output_z_gpu;

	size_t bytes = (img_w * img_h * img_l) * sizeof(float);
	HANDLE_ERROR(cudaMalloc(&input_x_gpu, bytes));  							    //allocate memory on 
	HANDLE_ERROR(cudaMalloc(&input_y_gpu, bytes));  							    //allocate memory on device
	HANDLE_ERROR(cudaMalloc(&input_z_gpu, bytes));  							    //allocate memory on device
	HANDLE_ERROR(cudaMalloc(&output_x_gpu, bytes));  							//allocate memory on device
	HANDLE_ERROR(cudaMalloc(&output_y_gpu, bytes));  							//allocate memory on device
	HANDLE_ERROR(cudaMalloc(&output_z_gpu, bytes));  							//allocate memory on device


	HANDLE_ERROR(cudaMemcpy(input_x_gpu, input_x, bytes, cudaMemcpyHostToDevice));     //copy the array from main memory to 
	HANDLE_ERROR(cudaMemcpy(input_y_gpu, input_y, bytes, cudaMemcpyHostToDevice));     //copy the array from main memory to device
	HANDLE_ERROR(cudaMemcpy(input_z_gpu, input_z, bytes, cudaMemcpyHostToDevice));     //copy the array from main memory to device


	size_t blockDim = sqrt(props.maxThreadsPerBlock);
	dim3 threads(blockDim, blockDim);
	dim3 blocks(img_w / threads.x + 1, img_h / threads.y + 1, img_l / threads.z + 1);

	gradient_dx_cuda << < blocks, threads >> > (input_x_gpu, output_x_gpu, img_w, img_h, img_l);
	gradient_dy_cuda << < blocks, threads >> > (input_y_gpu, output_y_gpu, img_w, img_h, img_l);
	gradient_dz_cuda << < blocks, threads >> > (input_z_gpu, output_z_gpu, img_w, img_h, img_l);



	HANDLE_ERROR(cudaMemcpy(output_x, output_x_gpu, bytes, cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(output_y, output_y_gpu, bytes, cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(output_z, output_z_gpu, bytes, cudaMemcpyDeviceToHost));


	cudaFree(input_x_gpu);
	cudaFree(input_y_gpu);
	cudaFree(input_z_gpu);
	cudaFree(output_x_gpu);
	cudaFree(output_y_gpu);
	cudaFree(output_z_gpu);


}

//__global__ void Eout_Ein_calculation_shared_memory(float* Eo_gpu, float* Ei_gpu, float* fo_gpu, float* fi_gpu, float* input_img_gpu, float* input, float* fout, float* fin, int img_w, int img_h, int img_l, int sigma) {
//
//	extern __shared__ unsigned char sharedPtr[];
//	size_t i = blockDim.y * blockIdx.y + threadIdx.y;	// calculate row index, point to the output  //width 
//	size_t j = blockDim.x * blockIdx.x + threadIdx.x;	// calculate column index, point to the output //height
//	size_t p = blockDim.z * blockIdx.z + threadIdx.z;
//	size_t ti = threadIdx.y;							// calculate the i (row) index, point to the shared memory
//	size_t tj = threadIdx.x;							// calculate the j (column) index, point to the shared memory
//	size_t tp = threadIdx.z;							
//
//	if (i >= img_h || j >= img_w || p >= img_l) return;
//
//	int L = blockDim.x + sigma ;							// length of data required in one block
//
//	float s1 = 0;
//	float s2 = 0;
//
//	for (int u = -sigma; u <= sigma; u++)
//	{
//		for (int v = -sigma; v <= sigma; v++)
//		{
//			for (int w = -sigma; w <= sigma; w++)
//			{
//				if (((i + u) >= 0) && ((i + u) < img_h) && ((j + v) >= 0) && ((j + v) < img_w) && ((p + w) >= 0) && ((p + w) < img_l))
//				{
//					s1 = (s1 + ((1 / pow(((2 * sigma) + 1), 2)) * (input_img_gpu[((i + u) * img_w * img_h) + ((j + v) * img_w) + (p + w)] - fo_gpu[(i * img_w * img_h) + j * img_w + p]) * (input_img_gpu[((i + u) * img_w * img_h) + ((j + v) * img_w) + (p + w)] - fo_gpu[(i * img_w * img_h) + j * img_w + p])));
//					s2 = (s2 + ((1 / pow(((2 * sigma) + 1), 2)) * (input_img_gpu[((i + u) * img_w * img_h) + ((j + v) * img_w) + (p + w)] - fi_gpu[(i * img_w * img_h) + j * img_w + p]) * (input_img_gpu[((i + u) * img_w * img_h) + ((j + v) * img_w) + (p + w)] - fi_gpu[(i * img_w * img_h) + j * img_w + p])));
//				}
//			}
//		}
//	}
//	Eo_gpu[i * img_w * img_h + j * img_w + p] = s1;
//	Ei_gpu[i * img_w * img_h + j * img_w + p] = s2;
//
//
//
//}

//Eout_Ein_calculation_separable

__global__ void box_conv_separable_X(float* output_1_gpu, float* input_img_gpu, int img_w, int img_h, int img_l, float sigma) {


	size_t i = blockDim.y * blockIdx.y + threadIdx.y;	// calculate row index, point to the output  //width 
	size_t j = blockDim.x * blockIdx.x + threadIdx.x;	// calculate column index, point to the output //height
	size_t p = blockDim.z * blockIdx.z + threadIdx.z;
	if (i >= img_h || j >= img_w || p >= img_l) return;

	float s1 = 0;
	

	for (int u = -sigma; u <= sigma; u++)
	{
		if (((i + u) >= 0) && ((i + u) < img_h))
		{
			s1 = s1 + (input_img_gpu[((i + u) * img_w * img_h) + (j * img_w) + p]);
		}
	}
	output_1_gpu[i * img_w * img_h + j * img_w + p] = s1;
	
}

__global__ void gradientmagnitude(float* input, float* output_x, float* output_y, float* output_z, int img_w, int img_h, int img_l) {


	size_t i = blockDim.y * blockIdx.y + threadIdx.y;  // Row index
	size_t j = blockDim.x * blockIdx.x + threadIdx.x;  // Column index
	size_t p = blockDim.z * blockIdx.z + threadIdx.z;  // Depth index

	if (i < img_h && j < img_w && p < img_l) {
		size_t index = i * img_w * img_l + j * img_l + p;

		float gradient_x = output_x[index];
		float gradient_y = output_y[index];
		float gradient_z = output_z[index];

		float gradient_magnitude = sqrt(gradient_x * gradient_x + gradient_y * gradient_y + gradient_z * gradient_z);

		input[index] = gradient_magnitude;
	}

}

void gradientmagnitude_cal(float* output, float* input_x, float* input_y, float* input_z, int img_w, int img_h, int img_l) {


	cudaDeviceProp props;
	HANDLE_ERROR(cudaGetDeviceProperties(&props, 0));


	float* output_gpu;
	float* input_x_gpu;
	float* input_y_gpu;
	float* input_z_gpu;

	size_t bytes = (img_w * img_h * img_l) * sizeof(float);
	HANDLE_ERROR(cudaMalloc(&output_gpu, bytes));  							    //allocate memory on device
	HANDLE_ERROR(cudaMalloc(&input_x_gpu, bytes));  							//allocate memory on device
	HANDLE_ERROR(cudaMalloc(&input_y_gpu, bytes));  							//allocate memory on device
	HANDLE_ERROR(cudaMalloc(&input_z_gpu, bytes));  							//allocate memory on device



	HANDLE_ERROR(cudaMemcpy( input_x_gpu, input_x, bytes, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy( input_y_gpu, input_y, bytes, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy( input_z_gpu, input_z, bytes, cudaMemcpyHostToDevice));


	size_t blockDim = sqrt(props.maxThreadsPerBlock);
	dim3 threads(blockDim, blockDim);
	dim3 blocks(img_w / threads.x + 1, img_h / threads.y + 1, img_l / threads.z + 1);

	gradientmagnitude << < blocks, threads >> > (output_gpu, input_x_gpu, input_y_gpu, input_z_gpu, img_w, img_h, img_l);
	

	HANDLE_ERROR(cudaMemcpy( output, output_gpu, bytes, cudaMemcpyDeviceToHost));     //copy the array from main memory to device

	cudaFree(output_gpu);
	cudaFree(input_x_gpu);
	cudaFree(input_y_gpu);
	cudaFree(input_z_gpu);


}

__global__ void box_conv_separable_Y(float* output_1_gpu, float* input_img_gpu, int img_w, int img_h, int img_l, float sigma) {


	size_t i = blockDim.y * blockIdx.y + threadIdx.y;	// calculate row index, point to the output  //width 
	size_t j = blockDim.x * blockIdx.x + threadIdx.x;	// calculate column index, point to the output //height
	size_t p = blockDim.z * blockIdx.z + threadIdx.z;
	if (i >= img_h || j >= img_w || p >= img_l) return;

	float s1 = 0;


	for (int u = -sigma; u <= sigma; u++)
	{
		if (((j + u) >= 0) && ((j + u) < img_w))
		{
			s1 = s1 + input_img_gpu[(i  * img_w * img_h) + ((j + u) * img_w) + p ];

		}

	}
	output_1_gpu[i * img_w * img_h + j * img_w + p] = s1;

}

__global__ void box_conv_separable_Z(float* output_1_gpu, float* input_img_gpu,  int img_w, int img_h, int img_l, float sigma) {


	size_t i = blockDim.y * blockIdx.y + threadIdx.y;	// calculate row index, point to the output  //width 
	size_t j = blockDim.x * blockIdx.x + threadIdx.x;	// calculate column index, point to the output //height
	size_t p = blockDim.z * blockIdx.z + threadIdx.z;
	if (i >= img_h || j >= img_w || p >= img_l) return;

	float s1 = 0;


	for (int u = -sigma; u <= sigma; u++)
	{
		if (((p + u) >= 0) && ((p + u) < img_l))
		{
			s1 = s1 + input_img_gpu[(i * img_w * img_h) + (j * img_w) + (p + u)];

		}

	}
	output_1_gpu[i * img_w * img_h + j * img_w + p] = s1;

}


__global__ void cal_f_x(float* input_fx_gpu, float* output_fx_gpu, float* input_sq_gpu, float* input_2_gpu, float* input_3_gpu, int img_w, int img_h, int img_l, float sigma) {

	//tira::volume<float>input_sq(input.X(), input.Y(), input.Z());
	

	size_t i = blockDim.y * blockIdx.y + threadIdx.y;	// calculate row index, point to the output  //width 
	size_t j = blockDim.x * blockIdx.x + threadIdx.x;	// calculate column index, point to the output //height
	size_t p = blockDim.z * blockIdx.z + threadIdx.z;
	if (i >= img_h || j >= img_w || p >= img_l) return;

	/*for (int yi = 0; yi < input.Y(); yi++) {
		for (int xi = 0; xi < input.X(); xi++) {
			for (int zi = 0; zi < input.Z(); zi++) {

				input_sq(xi, yi, zi) = input(xi, yi, zi) * input(xi, yi, zi);
			}
		}
	}*/

	input_sq_gpu[(i * img_w * img_h) + (j * img_w) + p ] = input_fx_gpu[(i * img_w * img_h) + (j * img_w) + p] * input_fx_gpu[(i * img_w * img_h) + (j * img_w) + p];

	//tira::volume<float>input_2(input.X(), input.Y(), input.Z());
	float s1 = 0;
	for (int u = -sigma; u <= sigma; u++)
	{
		if (((i + u) >= 0) && ((i + u) < img_h))
		{
			s1 = s1 + input_sq_gpu[(i * img_w * img_h) + (j * img_w) + p];
		}
	}
	input_2_gpu[i * img_w * img_h + j * img_w + p] = s1;


	float s2 = 0;
	for (int u = -sigma; u <= sigma; u++)
	{
		if (((j + u) >= 0) && ((j + u) < img_w))
		{
			s2 = s2 + input_2_gpu[(i * img_w * img_h) + (j * img_w) + p];
		}
	}
	input_3_gpu[i * img_w * img_h + j * img_w + p] = s2;



	float s3 = 0;
	for (int u = -sigma; u <= sigma; u++)
	{
		if (((p + u) >= 0) && ((p + u) < img_l))
		{
			s3 = s3 + input_3_gpu[(i * img_w * img_h) + (j * img_w) + p];
		}
	}
	output_fx_gpu[i * img_w * img_h + j * img_w + p] = s3;
	
	/*for (int yi = 0; yi < input.Y(); yi++) {
		for (int xi = 0; xi < input.X(); xi++) {
			for (int zi = 0; zi < input.Z(); zi++) {
				float s1 = 0;
				for (int u = -sigma; u <= sigma; u++) {
					if ((yi + u >= 0) && (yi + u < input.Y()))
					{
						s1 += input_sq(xi, yi, zi);

					}
				}
				input_2(xi, yi, zi) = s1;
			}
		}
	}*/

	//tira::volume<float>input_3(input.X(), input.Y(), input.Z());

	/*float* input_3

	for (int yi = 0; yi < input.Y(); yi++) {
		for (int xi = 0; xi < input.X(); xi++) {
			for (int zi = 0; zi < input.Z(); zi++) {
				float s1 = 0;
				for (int u = -sigma; u <= sigma; u++) {
					if ((xi + u >= 0) && (xi + u < input.X()))
					{
						s1 += input_2(xi, yi, zi);

					}
				}
				input_3(xi, yi, zi) = s1;
			}
		}
	}*/


	//tira::volume<float>input_4(input.X(), input.Y(), input.Z());

	

	/*for (int yi = 0; yi < input.Y(); yi++) {
		for (int xi = 0; xi < input.X(); xi++) {
			for (int zi = 0; zi < input.Z(); zi++) {
				float s1 = 0;
				for (int u = -sigma; u <= sigma; u++) {
					if ((zi + u >= 0) && (zi + u < input.Z()))
					{
						s1 += input_3(xi, yi, zi);

					}
				}
				input_4(xi, yi, zi) = s1;
			}
		}
	}*/

	
}

void fx_cal_separable(float* input_fx, float* output_fx, int img_w, int img_h, int img_l, float sigma) {


	cudaDeviceProp props;
	HANDLE_ERROR(cudaGetDeviceProperties(&props, 0));


	float* input_fx_gpu;
	float* output_fx_gpu;
	float* input_sq_gpu;
	float* input_2_gpu;
	float* input_3_gpu;
	size_t bytes = (img_w * img_h * img_l) * sizeof(float);
	HANDLE_ERROR(cudaMalloc(&input_fx_gpu, bytes));  							    //allocate memory on device
	HANDLE_ERROR(cudaMalloc(&output_fx_gpu, bytes));  							//allocate memory on device
	HANDLE_ERROR(cudaMalloc(&input_sq_gpu, bytes));  							//allocate memory on device
	HANDLE_ERROR(cudaMalloc(&input_2_gpu, bytes));  							//allocate memory on device
	HANDLE_ERROR(cudaMalloc(&input_3_gpu, bytes));  							//allocate memory on device

	HANDLE_ERROR(cudaMemcpy(input_fx_gpu, input_fx, bytes, cudaMemcpyHostToDevice));     //copy the array from main memory to device


	size_t blockDim = sqrt(props.maxThreadsPerBlock);
	dim3 threads(blockDim, blockDim);
	dim3 blocks(img_w / threads.x + 1, img_h / threads.y + 1, img_l / threads.z + 1);

	cal_f_x << < blocks, threads >> > (input_fx_gpu, output_fx_gpu, input_sq_gpu, input_2_gpu, input_3_gpu, img_w, img_h, img_l, sigma);

	/*box_conv_separable_X << < blocks, threads >> > (input_box_convolved_gpu_1, input_img_gpu, img_w, img_h, img_l, sigma);
	box_conv_separable_Y << < blocks, threads >> > (input_box_convolved_gpu_2, input_box_convolved_gpu_1, img_w, img_h, img_l, sigma);
	box_conv_separable_Z << < blocks, threads >> > (input_box_convolved_gpu, input_box_convolved_gpu_2, img_w, img_h, img_l, sigma);*/
	// float* Eo_gpu, float* Ei_gpu, float* fo_gpu, float* fi_gpu, float* input_img_gpu, float* input, float* fout, float* fin, int img_w, int img_h, int img_l, int sigma

	HANDLE_ERROR(cudaMemcpy(output_fx, output_fx_gpu, bytes, cudaMemcpyDeviceToHost));


	cudaFree(input_fx_gpu);
	cudaFree(input_sq_gpu);
	cudaFree(input_2_gpu);
	cudaFree(input_3_gpu);
	cudaFree(output_fx_gpu);




}

void box_conv_separable(float* input, float*input_box_convolved, int img_w, int img_h, int img_l, float sigma) {


	cudaDeviceProp props;
	HANDLE_ERROR(cudaGetDeviceProperties(&props, 0));


	float* input_box_convolved_gpu;
	float* input_box_convolved_gpu_1;
	float* input_box_convolved_gpu_2;
	float* input_img_gpu;
	size_t bytes = (img_w * img_h * img_l) * sizeof(float);
	HANDLE_ERROR(cudaMalloc(&input_img_gpu, bytes));  							    //allocate memory on device
	HANDLE_ERROR(cudaMalloc(&input_box_convolved_gpu_1, bytes));  							//allocate memory on device
	HANDLE_ERROR(cudaMalloc(&input_box_convolved_gpu_2, bytes));  							//allocate memory on device
	HANDLE_ERROR(cudaMalloc(&input_box_convolved_gpu, bytes));  							//allocate memory on device
	
	HANDLE_ERROR(cudaMemcpy(input_img_gpu, input, bytes, cudaMemcpyHostToDevice));     //copy the array from main memory to device
	
	
	size_t blockDim = sqrt(props.maxThreadsPerBlock);
	dim3 threads(blockDim, blockDim );
	dim3 blocks(img_w / threads.x + 1, img_h / threads.y + 1, img_l / threads.z + 1);

	box_conv_separable_X << < blocks, threads >> > (input_box_convolved_gpu_1, input_img_gpu, img_w, img_h, img_l, sigma);
	box_conv_separable_Y << < blocks, threads >> > (input_box_convolved_gpu_2, input_box_convolved_gpu_1, img_w, img_h, img_l, sigma);
	box_conv_separable_Z << < blocks, threads >> > (input_box_convolved_gpu, input_box_convolved_gpu_2, img_w, img_h, img_l, sigma);
	// float* Eo_gpu, float* Ei_gpu, float* fo_gpu, float* fi_gpu, float* input_img_gpu, float* input, float* fout, float* fin, int img_w, int img_h, int img_l, int sigma

	HANDLE_ERROR(cudaMemcpy(input_box_convolved, input_box_convolved_gpu, bytes, cudaMemcpyDeviceToHost));
	
	cudaFree(input_img_gpu);
	cudaFree(input_box_convolved_gpu_1);
	cudaFree(input_box_convolved_gpu_2);
	cudaFree(input_box_convolved_gpu);

	


}


// convolution on device
__global__ void Convolution__on_device(float* out, float* img, float* kernel, int img_w, int img_l, int img_h, int out_w, int out_h, int out_l, int K) {
	size_t i = blockDim.y * blockIdx.y + threadIdx.y;
	size_t j = blockDim.x * blockIdx.x + threadIdx.x;
	size_t p = blockDim.z * blockIdx.z + threadIdx.z;

	// i and j being smaller than output's width and height, manage the edges perfectly
	if (i >= out_h || j >= out_w || p >= out_l ) return;

	float conv = 0;
	for (int ki = 0; ki < K; ki++)
		for (int kj = 0; kj < K; kj++)
			for (int kk = 0; kk < K; kk++)
			conv += img[  (i + ki) * img_w * img_h+ (j + kj) * img_w + (p + kk)] * kernel[ki * K * K + kj * K + kk ];

	out[i * out_w * out_h + j * out_w + p] = conv;

}

// convolution on device x
__global__ void Convolution__on_X(float* out, float* img, float* kernel, int img_w, int img_l, int img_h, int out_w, int out_h, int out_l, unsigned int K) {
	size_t i = blockDim.y * blockIdx.y + threadIdx.y;
	size_t j = blockDim.x * blockIdx.x + threadIdx.x;
	size_t p = blockDim.z * blockIdx.z + threadIdx.z;

	// i and j being smaller than output's width and height, manage the edges perfectly
	if (i >= out_h || j >= out_w || p >= out_l) return;

	float conv = 0;
	for (int ki = 0; ki < K; ki++)
				conv += (float)img[(p * img_h + i) * img_w + (j + ki)] * kernel[ki ];

	out[(p * out_h + i) * out_w + j] = (float)conv;

}

// convolution on device y
__global__ void Convolution__on_Y(float* out, float* img, float* kernel, int img_w, int img_l, int img_h, int out_w, int out_h, int out_l, unsigned int K) {
	size_t i = blockDim.y * blockIdx.y + threadIdx.y;
	size_t j = blockDim.x * blockIdx.x + threadIdx.x;
	size_t p = blockDim.z * blockIdx.z + threadIdx.z;

	// i and j being smaller than output's width and height, manage the edges perfectly
	if (i >= out_h || j >= out_w || p >= out_l) return;

	float conv = 0;
	for (int ki = 0; ki < K; ki++)
		conv += (float)img[(p * img_h + (i + ki)) * img_w + j] * kernel[ki];

	out[(p * out_h + i) * out_w + j] = (float)conv;

}


// convolution on device z
__global__ void Convolution__on_Z(float* out, float* img, float* kernel, int img_w, int img_l, int img_h, int out_w, int out_h, int out_l, unsigned int K) {
	size_t i = blockDim.y * blockIdx.y + threadIdx.y;
	size_t j = blockDim.x * blockIdx.x + threadIdx.x;
	size_t p = blockDim.z * blockIdx.z + threadIdx.z;

	// i and j being smaller than output's width and height, manage the edges perfectly
	if (i >= out_h || j >= out_w || p >= out_l) return;

	float conv = 0;
	for (int ki = 0; ki < K; ki++)
		conv += img[((p + ki) * img_h + i) * img_w + j] * kernel[ki];

	out[(p * out_h + i) * out_w + j] = conv;

}


// Convolution on device x with shared memory caching
__global__ void Convolution__on_X_shared(float* out, float* img, float* kernel, int img_w, int img_l, int img_h, int out_w, int out_h, int out_l, unsigned int K) {
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int tz = threadIdx.z;

	int bx = blockIdx.x;
	int by = blockIdx.y;
	int bz = blockIdx.z;

	int i = blockDim.y * by + ty;
	int j = blockDim.x * bx + tx;
	int p = blockDim.z * bz + tz;


	if (i >= out_h || j >= out_w || p >= out_l) return;
	
	int L = blockDim.x + K - 1;

	extern __shared__ float shared_mem[];

	// Load input image and kernel into shared memory
	for (int ki = 0; ki < K; ki++) {
		int img_idx = (p * img_h + i) * img_w + (j + ki);
		int shared_idx = (tz * blockDim.y + ty) * L + tx + ki;
		shared_mem[shared_idx] = img[img_idx];
	}

	__syncthreads();

	float conv = 0;
	for (int ki = 0; ki < K; ki++) {
		int shared_idx = (tz * blockDim.y + ty) * L + tx + ki;
		conv += shared_mem[shared_idx] * kernel[ki];
	}
	out[(p * out_h + i) * out_w + j] = conv;
}


// convolution on device x (shared memory)
__global__ void Convolution_on_Y_sharedwrong(float* out, float* img, float* kernel, int img_w, int img_l, int img_h, int out_w, int out_h, int out_l, unsigned int K) {

	extern __shared__ float sharedPtr[];
	size_t i = blockDim.y * blockIdx.y + threadIdx.y;
	size_t j = blockDim.x * blockIdx.x + threadIdx.x;
	size_t p = blockDim.z * blockIdx.z + threadIdx.z;

	size_t tp = threadIdx.z;
	size_t tj = threadIdx.x;
	size_t ti = threadIdx.y;

	// i, j, and p being smaller than output's width, length, and height, manage the edges perfectly
	if (i >= out_h || j >= out_w || p >= out_l) return;

	size_t L = blockDim.y + K - 1;							// length of data required in one block


	// Copy data to shared memory
	for (int m = ti; m < L; m += blockDim.y) {
		sharedPtr[(tp * L + m) * blockDim.x + tj] = img[ (p * img_w * img_h + (blockDim.y * blockIdx.y + m) * img_w + j)];
	}

	__syncthreads();



	float conv = 0;
	for (int ki = 0; ki < K; ki++) {
		conv += sharedPtr[(tp * img_h + (ti + ki)) * img_w + tj ] * kernel[ki];
	}

	out[(p * out_h + i) * out_w + j] = conv;

}


// convolution on device x (shared memory)
__global__ void Convolution_on_Z_sharedwrong(float* out, float* img, float* kernel, int img_w, int img_l, int img_h, int out_w, int out_h, int out_l, unsigned int K) {

	extern __shared__ float sharedPtr[];
	size_t i = blockDim.y * blockIdx.y + threadIdx.y;
	size_t j = blockDim.x * blockIdx.x + threadIdx.x;
	size_t p = blockDim.z * blockIdx.z + threadIdx.z;

	size_t tp = threadIdx.z;
	size_t tj = threadIdx.x;
	size_t ti = threadIdx.y;

	// i, j, and p being smaller than output's width, length, and height, manage the edges perfectly
	if (i >= out_h || j >= out_w || p >= out_l) return;

	size_t L = blockDim.z + K - 1;  // length of data required in one block

	// Copy data to shared memory
	for (int m = tp; m < L; m += blockDim.z) {
		sharedPtr[((ti * blockDim.x + tj) * L + m)] = img[ ((blockDim.z * blockIdx.z + m)* img_w * img_h + i * img_w  + j )];
	}

	__syncthreads();




	float conv = 0;
	for (int ki = 0; ki < K; ki++) {
		conv += sharedPtr[((tp+ki) * img_h + ti ) * img_w + tj] * kernel[ki];
	}

	out[(p * out_h + i) * out_w + j] = conv;

}

// Convolution on device y with shared memory caching
__global__ void Convolution__on_Y_shared(float* out, float* img, float* kernel, int img_w, int img_l, int img_h, int out_w, int out_h, int out_l, unsigned int K) {
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int tz = threadIdx.z;

	int bx = blockIdx.x;
	int by = blockIdx.y;
	int bz = blockIdx.z;

	int i = blockDim.y * by + ty;
	int j = blockDim.x * bx + tx;
	int p = blockDim.z * bz + tz;


	if (i >= out_h || j >= out_w || p >= out_l) return;
	
	int L = (blockDim.y + K - 1) * blockDim.x;

	extern __shared__ float shared_mem[];

	// Load input image and kernel into shared memory
	if (i < img_h && j < img_w && p < img_l) {
		for (int ki = 0; ki < K; ki++) {
			int img_idx = (p * img_h + (i + ki)) * img_w + j;
			int shared_idx = (tz * L + (ty + ki) * blockDim.x + tx);
			shared_mem[shared_idx] = img[img_idx];
		}
	}

	__syncthreads();

	
	float conv = 0;
	for (int ki = 0; ki < K; ki++) {
		int shared_idx = (tz * L + (ty + ki) * blockDim.x + tx);
		conv += shared_mem[shared_idx] * kernel[ki];
	}
	out[(p * out_h + i) * out_w + j] = conv;
}


__global__ void Convolution__on_Z_shared(float* out, float* img, float* kernel, int img_w, int img_l, int img_h, int out_w, int out_h, int out_l, unsigned int K) {
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int tz = threadIdx.z;

	int bx = blockIdx.x;
	int by = blockIdx.y;
	int bz = blockIdx.z;

	int i = blockDim.y * by + ty;
	int j = blockDim.x * bx + tx;
	int p = blockDim.z * bz + tz;

	if (i >= out_h || j >= out_w || p >= out_l) return;

	//length of data that are required in one block
	int L = (blockDim.z + K - 1) ;

	extern __shared__ float shared_mem[];

	// Load input image and kernel into shared memory
	if (i < img_h && j < img_w && p < img_l) {
		for (int ki = 0; ki < K; ki++) {
			int img_idx = (p + ki) * img_h * img_w + i * img_w + j;
			int shared_idx = ((tz + ki) * (blockDim.x - K + 1) + ty * (blockDim.x - K + 1) + tx );
			shared_mem[shared_idx] = img[img_idx];
		}
	}

	__syncthreads();


	float conv = 0;
	for (int ki = 0; ki < K; ki++) {
		int shared_idx = ((tz + ki)  * (blockDim.x - K +1) + ty * (blockDim.x - K + 1) + tx);
		conv += shared_mem[shared_idx] * kernel[ki];
	}
	out[(p * out_h * out_w) + i * out_w + j] = conv;
	//out[(p * out_h + i) * out_w + j] = conv;
}

void adddevice_convolution_seperable_shared(float* output, float* in_img, int img_w, int img_h, int img_l, float sigma, float* gkernel, unsigned int k_size) {


	cudaDeviceProp props;
	HANDLE_ERROR(cudaGetDeviceProperties(&props, 0));

	int x_height = img_h;
	int x_width = img_w - k_size + 1;
	int x_length = img_l;
	int x_size = x_height * x_width * x_length;

	int y_height = img_h - k_size + 1;
	int y_width = img_w - k_size + 1;
	int y_length = img_l;
	int y_size = y_height * y_width * y_length;

	int z_height = img_h - k_size + 1;
	int z_width = img_w - k_size + 1;
	int z_length = img_l - k_size + 1;
	int z_size = z_height * z_width * z_length;
	//y_output = (float*)malloc(y_size * sizeof(float));


	float* gkernel_gpu;
	float* input_img_gpu;
	float* gpu_output_x;
	float* gpu_output_y;
	float* gpu_output_z;
	size_t bytes = (img_w * img_h * img_l) * sizeof(float);


	HANDLE_ERROR(cudaMalloc(&gkernel_gpu, k_size * sizeof(float)));
	HANDLE_ERROR(cudaMalloc(&input_img_gpu, bytes));  							    //allocate memory on device
	HANDLE_ERROR(cudaMalloc(&gpu_output_x, x_size * sizeof(float)));  				//allocate memory on device
	HANDLE_ERROR(cudaMalloc(&gpu_output_y, y_size * sizeof(float)));  				//allocate memory on device
	HANDLE_ERROR(cudaMalloc(&gpu_output_z, z_size * sizeof(float)));  				//allocate memory on device

	HANDLE_ERROR(cudaMemcpy(input_img_gpu, in_img, bytes, cudaMemcpyHostToDevice));     //copy the array from main memory to device
	HANDLE_ERROR(cudaMemcpy(gkernel_gpu, gkernel, k_size * sizeof(float), cudaMemcpyHostToDevice));     //copy the array from main memory to device


	//// convolving along x
	//Convolution_x_on_device << < blocks, threads >> > (gpu_output_x, input_img_gpu, gkernel_gpu, img_w, x_height, x_width, k);
	//// convolving along y
	//Convolution__on_device(float* out, float* img, float* kernel, int img_w, int img_l, int out_w, int out_h, int out_l, int K)
	size_t blockDim = sqrt(props.maxThreadsPerBlock);
	dim3 threads(blockDim, blockDim);
	dim3 blocks(img_w / threads.x + 1, img_h / threads.y + 1, img_l );
	//dim3 blocks((x_width + blockDim - 1) / blockDim, (x_height + blockDim - 1) / blockDim, (x_length + blockDim - 1) / blockDim);
	// calculate the size of shared memory
	size_t sharedmemory = blockDim *  (blockDim + k_size - 1) * sizeof(float);
	if (props.sharedMemPerBlock < sharedmemory) {
		std::cout << "ERROR:  shared memory is insufficient " << std::endl;
		exit(1);
	}



	Convolution__on_X_shared << < blocks, threads, sharedmemory >> > (gpu_output_x, input_img_gpu, gkernel_gpu, img_w, img_l, img_h, x_width, x_height, x_length, k_size);
	Convolution__on_Y_shared << < blocks, threads, sharedmemory >> > (gpu_output_y, gpu_output_x, gkernel_gpu, x_width, x_length, x_height, y_width, y_height, y_length, k_size);
	//Convolution__on_Z_shared << < blocks, threads, sharedmemory >> > (gpu_output_z, gpu_output_y, gkernel_gpu, y_width, y_length, y_height, z_width, z_height, z_length, k_size);
	

	/*Convolution__on_X << < blocks, threads >> > (gpu_output_x, input_img_gpu, gkernel_gpu, img_w, img_l, img_h, x_width, x_height, x_length, k_size);
	Convolution__on_Y<< < blocks, threads >> > (gpu_output_y, gpu_output_x, gkernel_gpu, x_width, x_length, x_height, y_width, y_height, y_length, k_size);
	Convolution__on_Z<< < blocks, threads >> > (gpu_output_z, gpu_output_y, gkernel_gpu, y_width, y_length, y_height, z_width, z_height, z_length, k_size);*/

	//float* out, float* img, float* kernel, int img_w, int out_h, int out_w, int K
	// copy convolved outputs from Device to main memory
	//HANDLE_ERROR(cudaMemcpy(x_output, gpu_output_x, x_size * sizeof(float), cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(output, gpu_output_y, y_size * sizeof(float), cudaMemcpyDeviceToHost));




	cudaFree(gkernel_gpu);
	cudaFree(input_img_gpu);
	cudaFree(gpu_output_x);
	cudaFree(gpu_output_y);
	cudaFree(gpu_output_z);


}

//// convolution kernel along x running on device (with shared memory)
//__global__ void kernelConvolution_x(unsigned char* C, unsigned char* A, float* B, int M, int N, int M_x, int N_x, int K) {
//	extern __shared__ unsigned char sharedPtr[];
//	size_t i = blockDim.y * blockIdx.y + threadIdx.y;   // calculate the i (row) index, point to the outMatrix
//	size_t j = blockDim.x * blockIdx.x + threadIdx.x;   // calculate the j (column) index, point to the outMatrix
//	size_t ti = threadIdx.y;							// calculate the i (row) index, point to the shared memory
//	size_t tj = threadIdx.x;							// calculate the j (column) index, point to the shared memory
//	if (i >= M_x || j >= N_x) return;
//	int L = blockDim.x + K - 1;							// length of data required in one block
//	// copy data to shared memory
//	for (int l = tj; l < L; l += blockDim.x) {
//		sharedPtr[ (ti * L + l)] = A[ (i * N + blockDim.x * blockIdx.x + l)];
//	}
//	__syncthreads();
//
//	// initialize the register c to store the results
//	float c=0;
//	
//	// apply the convolution with Gaussian kernel along x axis
//	for (int k = 0; k < K; k++) {
//		c += (float)sharedPtr[ (ti * L + tj + k)] * B[k];
//		
//	}
//	// copy results from register to outMatrix
//	C[ (i * N_x + j)] = (unsigned char)c;
//	
//}





void adddevice_convolution_seperable(float* output, float* in_img, int img_w, int img_h, int img_l, float sigma, float* gkernel, unsigned int k_size) {


	cudaDeviceProp props;
	HANDLE_ERROR(cudaGetDeviceProperties(&props, 0));

	int x_height = img_h;
	int x_width = img_w - k_size + 1;
	int x_length = img_l;
	int x_size = x_height * x_width * x_length;

	int y_height = img_h - k_size + 1;
	int y_width = img_w - k_size + 1;
	int y_length = img_l;
	int y_size = y_height * y_width * y_length;

	int z_height = img_h - k_size + 1;
	int z_width = img_w - k_size + 1;
	int z_length = img_l - k_size + 1;
	int z_size = z_height * z_width * z_length;
	//y_output = (float*)malloc(y_size * sizeof(float));


	float* gkernel_gpu;
	float* input_img_gpu;
	float* gpu_output_x;
	float* gpu_output_y;
	float* gpu_output_z;
	size_t bytes = (img_w * img_h * img_l) * sizeof(float);


	HANDLE_ERROR(cudaMalloc(&gkernel_gpu, k_size * sizeof(float)));
	HANDLE_ERROR(cudaMalloc(&input_img_gpu, bytes));  							    //allocate memory on device
	HANDLE_ERROR(cudaMalloc(&gpu_output_x, x_size * sizeof(float)));  				//allocate memory on device
	HANDLE_ERROR(cudaMalloc(&gpu_output_y, y_size * sizeof(float)));  				//allocate memory on device
	HANDLE_ERROR(cudaMalloc(&gpu_output_z, z_size * sizeof(float)));  				//allocate memory on device

	HANDLE_ERROR(cudaMemcpy(input_img_gpu, in_img, bytes, cudaMemcpyHostToDevice));     //copy the array from main memory to device
	HANDLE_ERROR(cudaMemcpy(gkernel_gpu, gkernel, k_size * sizeof(float), cudaMemcpyHostToDevice));     //copy the array from main memory to device


	//// convolving along x
	//Convolution_x_on_device << < blocks, threads >> > (gpu_output_x, input_img_gpu, gkernel_gpu, img_w, x_height, x_width, k);
	//// convolving along y
	//Convolution__on_device(float* out, float* img, float* kernel, int img_w, int img_l, int out_w, int out_h, int out_l, int K)
	size_t blockDim = sqrt(props.maxThreadsPerBlock);
	dim3 threads(blockDim, blockDim);
	dim3 blocks(img_w / threads.x +1 , img_h / threads.y + 1, img_l / threads.z +1);
	//dim3 blocks((x_width + blockDim - 1) / blockDim, (x_height + blockDim - 1) / blockDim, (x_length + blockDim - 1) / blockDim);


	Convolution__on_X << < blocks, threads >> > (gpu_output_x, input_img_gpu, gkernel_gpu, img_w, img_l, img_h, x_width, x_height, x_length, k_size);
	Convolution__on_Y << < blocks, threads >> > (gpu_output_y, gpu_output_x, gkernel_gpu, x_width, x_length, x_height, y_width, y_height, y_length, k_size);
	Convolution__on_Z << < blocks, threads >> > (gpu_output_z, gpu_output_y, gkernel_gpu, y_width, y_length, y_height, z_width, z_height, z_length, k_size);
	//float* out, float* img, float* kernel, int img_w, int out_h, int out_w, int K
	// copy convolved outputs from Device to main memory
	//HANDLE_ERROR(cudaMemcpy(x_output, gpu_output_x, x_size * sizeof(float), cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(output, gpu_output_z, z_size * sizeof(float), cudaMemcpyDeviceToHost));



	
	cudaFree(gkernel_gpu);
	cudaFree(input_img_gpu);
	cudaFree(gpu_output_x);
	cudaFree(gpu_output_y);
	cudaFree(gpu_output_z);


}


void adddevice_convolution(float* y_output, float* in_img,  int img_w, int img_h,int img_l, float sigma, float* gkernel , unsigned int k_size) {


	cudaDeviceProp props;
	HANDLE_ERROR(cudaGetDeviceProperties(&props, 0));


	
	int y_height = img_h - k_size + 1;
	int y_width = img_w - k_size + 1;
	int y_length = img_l - k_size + 1;
	int y_size = y_height * y_width * y_length;
	//y_output = (float*)malloc(y_size * sizeof(float));

	
	float* gkernel_gpu;
	float* input_img_gpu;
	float* gpu_output_y;
	size_t bytes = (img_w * img_h * img_l) * sizeof(float);
	

	HANDLE_ERROR(cudaMalloc(&gkernel_gpu, k_size * k_size * k_size * sizeof(float)));
	HANDLE_ERROR(cudaMalloc(&input_img_gpu, bytes));  							    //allocate memory on device
	HANDLE_ERROR(cudaMalloc(&gpu_output_y, y_size * sizeof(float)));  				//allocate memory on device


	HANDLE_ERROR(cudaMemcpy(input_img_gpu, in_img, bytes, cudaMemcpyHostToDevice));     //copy the array from main memory to device
	HANDLE_ERROR(cudaMemcpy(gkernel_gpu, gkernel, k_size * k_size * k_size * sizeof(float), cudaMemcpyHostToDevice));     //copy the array from main memory to device


	//// convolving along x
	//Convolution_x_on_device << < blocks, threads >> > (gpu_output_x, input_img_gpu, gkernel_gpu, img_w, x_height, x_width, k);
	//// convolving along y
	//Convolution__on_device(float* out, float* img, float* kernel, int img_w, int img_l, int out_w, int out_h, int out_l, int K)
	size_t blockDim = sqrt(props.maxThreadsPerBlock);
	dim3 threads(blockDim, blockDim);
	dim3 blocks(img_w / threads.x , img_h / threads.y , img_l / threads.z );

	Convolution__on_device << < blocks, threads >> > (gpu_output_y, input_img_gpu, gkernel_gpu, img_w, img_l, img_h, y_width, y_height, y_length, k_size);
	//float* out, float* img, float* kernel, int img_w, int out_h, int out_w, int K
	// copy convolved outputs from Device to main memory
	//HANDLE_ERROR(cudaMemcpy(x_output, gpu_output_x, x_size * sizeof(float), cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(y_output, gpu_output_y, y_size * sizeof(float), cudaMemcpyDeviceToHost));

	
	
	cudaFree(gpu_output_y);
	cudaFree(gkernel_gpu);
	cudaFree(input_img_gpu); 


}

void adddevice(float* input, float* fout, float* fin, float* Eo, float* Ei, int img_w, int img_h, int img_l, float sigma) {
	
	
	cudaDeviceProp props;
	HANDLE_ERROR(cudaGetDeviceProperties(&props, 0));


	/*size_t blockDim = sqrt(props.maxThreadsPerBlock);
	dim3 threads(blockDim, blockDim);
	dim3 blocks(img_w / threads.x   , img_h / threads.y , img_l / threads.z );*/


	float* Eo_gpu;
	float* Ei_gpu;
	float* fo_gpu;
	float* fi_gpu;
	float* input_img_gpu;
	size_t bytes = (img_w * img_h * img_l) * sizeof(float);
	HANDLE_ERROR(cudaMalloc(&input_img_gpu, bytes));  							    //allocate memory on device
	HANDLE_ERROR(cudaMalloc(&fo_gpu,  bytes));  							//allocate memory on device
	HANDLE_ERROR(cudaMalloc(&fi_gpu, bytes));  							//allocate memory on device
	HANDLE_ERROR(cudaMalloc(&Eo_gpu, bytes));  							//allocate memory on device
	HANDLE_ERROR(cudaMalloc(&Ei_gpu, bytes));  							//allocate memory on device

	HANDLE_ERROR(cudaMemcpy(input_img_gpu, input, bytes, cudaMemcpyHostToDevice));     //copy the array from main memory to device
	HANDLE_ERROR(cudaMemcpy(fo_gpu, fout, bytes, cudaMemcpyHostToDevice));     //copy the array from main memory to device
	HANDLE_ERROR(cudaMemcpy(fi_gpu, fin, bytes, cudaMemcpyHostToDevice));     //copy the array from main memory to device

	size_t blockDim = sqrt(props.maxThreadsPerBlock);
	dim3 threads(blockDim, blockDim);
	dim3 blocks(img_w / threads.x + 1, img_h / threads.y + 1, img_l / threads.z + 1);

	Eout_Ein_calculation << < blocks, threads >> > (Eo_gpu, Ei_gpu, fo_gpu, fi_gpu, input_img_gpu, input, fout, fin, img_w, img_h, img_l, sigma);
	// float* Eo_gpu, float* Ei_gpu, float* fo_gpu, float* fi_gpu, float* input_img_gpu, float* input, float* fout, float* fin, int img_w, int img_h, int img_l, int sigma

	HANDLE_ERROR(cudaMemcpy(Eo , Eo_gpu, bytes, cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(Ei , Ei_gpu, bytes, cudaMemcpyDeviceToHost));

	/*for (int i = 0; i < 10; i++)
	{
		std::cout << Eo[i] << std::endl;
	}
	
	std::cout << "adddevice\n";*/
	//cudaDeviceSynchronize();
	/*free(input);
	free(fout);
	free(fin);
	free(Eo);
	free(Ei);*/
	cudaFree(Eo_gpu);
	cudaFree(Ei_gpu);
	cudaFree(fo_gpu);
	cudaFree(fi_gpu);
	cudaFree(input_img_gpu);


}


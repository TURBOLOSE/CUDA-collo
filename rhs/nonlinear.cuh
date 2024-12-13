#pragma once

//system_order=2
__device__ void rhs(double *y, double *yd, double t, double *par, size_t step, size_t iternum)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;

	yd[0]=y[1];
	yd[1]=pow(y[0],3)*pow(t+pos/100.,2);
}
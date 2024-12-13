#pragma once

//system_order=1
__device__ void rhs(double *y, double *yd, double t, double *par, size_t step, size_t iternum) 
{
		int pos = blockIdx.x * blockDim.x + threadIdx.x;

		yd[0]=pow(t+pos/100.,3);//y'=t^3    y=t^4/4
}
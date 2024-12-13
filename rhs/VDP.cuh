#pragma once

//system_order=2
__device__ void rhs(double *y, double *yd, double t, double *par, size_t step, size_t iternum) //VDP
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	double mu=2.0+pos/100.;
	yd[0]= mu*(y[0] -pow(y[0],3)/3.0 - y[1]);
	yd[1] =1/mu *y[0];
}
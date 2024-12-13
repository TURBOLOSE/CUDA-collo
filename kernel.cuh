#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <eigen3/Eigen/Dense>
#include "numm/numm.hpp"
#include <fstream>
#include <iostream>
#include <string>
#include <optional>



template <typename num_t, size_t system_order, size_t method_stage>
__global__ void kernel(num_t *dinputdata, num_t t0, num_t *doutputdata, std::optional<num_t *> par, num_t *dtime_nodes,
					   num_t *dsv_nodes, num_t *dinv_lsm, num_t h, size_t maxstep, size_t skipstep)
{
	// inputdata и outpudata для данных, nodes для узловых значений, rhs-- правые части, maxstep--число шагов
	// d=device

	using sv_t = Eigen::Matrix<num_t, system_order, 1>;
	using vector_t = Eigen::Matrix<num_t, method_stage, 1>;
	using matrix_t = Eigen::Matrix<num_t, method_stage, method_stage>;
	using sva_t = Eigen::Matrix<num_t, system_order, method_stage>;

	sva_t alphas;
	__shared__ vector_t time_nodes;
	__shared__  matrix_t sv_nodes;
	__shared__ matrix_t inv_lsm;
	sv_t y;
	sv_t y1, y_prev, y_temp;
	num_t dist, t;

	const size_t maxiter = 50;
	sva_t alphas_prev, f;
	constexpr size_t Nb = 13;
	extern __shared__ num_t parloc[Nb * method_stage * 3];

	int pos = blockIdx.x * blockDim.x + threadIdx.x;

	t = t0; // нач время


	for (size_t i = 0; i < system_order; i++)
		y(i) = dinputdata[pos * system_order + i]; 
	//different initial data for each thread

	for (size_t i = 0; i < method_stage; i++)
		time_nodes(i) = dtime_nodes[i];

	for (size_t i = 0; i < method_stage; i++)
	{ 
		for (size_t j = 0; j < method_stage; j++)
		{
			inv_lsm(i, j) = dinv_lsm[j * method_stage + i];
			sv_nodes(i, j) = dsv_nodes[j * method_stage + i];
		}
	}

	__syncthreads();

	alphas = sva_t::Zero();

	// main loop
	for (int step = 0; step < maxstep; step++)
	{

		__syncthreads();

		y_prev = y;

		if (par)
		{
			for (size_t i = 0; i < Nb * method_stage * 3; i++)
			{
				parloc[i] = par.value()[i + step * Nb * method_stage * 3];
			}
		}

		//do_step(y, t, h, 0, parloc);
		size_t iter = 0;
		do
		{
			alphas_prev = alphas;
			for (int i = 0; i < f.cols(); i++)
			{
				y1 = y + alphas * sv_nodes.row(i).transpose();
				rhs((y1).data(), (f.col(i)).data(), t + time_nodes(i) * h, parloc, 0, i);
			}

			alphas = f * inv_lsm * h;

			dist = (alphas - alphas_prev).norm();
			iter++;

		} while ((dist + num_t{2.0} != num_t{2.0}) && iter < maxiter);
		// y_prev=y;
		y += alphas(Eigen::all, Eigen::seq(0, Eigen::last, Eigen::fix<2>)).eval().rowwise().sum() * 2;





		t += h;

		if ((step % skipstep) == 0)
		{
			for (size_t j = 0; j < system_order; j++)
				doutputdata[(step / skipstep + j * maxstep / skipstep) + pos * maxstep * system_order / skipstep] = y(j);
			// output tables where every column = system_order, every row=maxstep, pos=number of table
		}


		if(pos==0 && ((int) (step*100.)/maxstep >  (int) ((step-1)*100.)/maxstep ))  //progress counter
		printf("\r progress:[%3d%%]", (int) (step*100.)/maxstep);

	}

	if(pos==0)
	printf("\n");
}

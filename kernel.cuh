#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <eigen3/Eigen/Dense>
#include "numm/numm.hpp"
#include <fstream>
#include <iostream>
#include <string>
#include <optional>
#include "rhs1.cuh"

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
	vector_t time_nodes;
	matrix_t sv_nodes;
	matrix_t inv_lsm;
	sv_t y;
	sv_t y1, y_prev, y_temp;
	num_t dist, t;

	const size_t maxiter = 50;
	// const int maxdepth=4;
	sva_t alphas_prev, f;
	// const num_t dE=0.00001; //max allowed derivation of energy
	// const num_t dE=10000; //max allowed derivation of energy
	constexpr size_t Nb = 13;
	__shared__ num_t parloc[Nb * method_stage * 3];

	int pos = blockIdx.x * blockDim.x + threadIdx.x;

	//=========================================
	num_t Gm[Nb] = {132712440041.279419, 22031.868551, 324858.592000, 398600.435507, 42828.375816, 126712764.100000, 37940584.841800,
					5794556.400000, 6836527.100580, 975.500000, 62.62890, 17.288245, 4902.800118}; // km^3 s^-2

	// num_t* par1=new num_t[Nb*system_order/2*method_stage];
	// matrix_t time_nodes1;

	for (size_t i = 0; i < Nb; i++)
	{
		Gm[i] *= 86400.0 * 86400 / pow(1.495978707e8, 3);
	}

	// num_t H1,H, R1, R, h1;
	//=========================================
	t = t0; // нач время

	// lambda for steps
	auto do_step = [&alphas, &alphas_prev, &maxiter, &y_prev, &y1, &dist, &inv_lsm, &f, &sv_nodes, &time_nodes](sv_t &y, num_t t, num_t h, int step, num_t *parloc)
	{
		size_t iter = 0;
		do
		{
			alphas_prev = alphas;
			for (int i = 0; i < f.cols(); i++)
			{
				y1 = y + alphas * sv_nodes.row(i).transpose();
				rhs((y1).data(), (f.col(i)).data(), t + time_nodes(i) * h, parloc, step, i);
			}

			alphas = f * inv_lsm * h;

			dist = (alphas - alphas_prev).norm();
			iter++;

		} while ((dist + num_t{2.0} != num_t{2.0}) && iter < maxiter);
		// y_prev=y;
		y += alphas(Eigen::all, Eigen::seq(0, Eigen::last, Eigen::fix<2>)).eval().rowwise().sum() * 2;
	};

	// lambda for full energy checks
	/*	auto E_full = [par, Gm](sv_t y, int step){
			num_t H,R;
			H=0.5*(y(3)*y(3)+y(4)*y(4)+y(5)*y(5));
			for (size_t i = 0; i < Nb; i++) // full energy
			{
				R=sqrt(pow((y(0)-par[Nb*system_order*method_stage/2*(step)+i*system_order/2]),2)+
				pow((y(1)-par[Nb*system_order*method_stage/2*(step)+i*system_order/2+1]),2)+
				pow((y(2)-par[Nb*system_order*method_stage/2*(step)+i*system_order/2+2]),2));
				H-=Gm[i]/R;
			}
			return H;
		};

		//lambda for new object positions in par1
		auto make_new_par = [&time_nodes, par, Nb, &time_nodes1](num_t* par1, int step, int substep, int depth){


			for (int i = 0; i <method_stage; i++){// new time_nodes
				for (int j = 0; j <method_stage; j++)
				{
					time_nodes1(i,j)=1;
					for (int k = 0; k <method_stage; k++){
						if(j!=k){
							time_nodes1(i,j)*=(((substep+time_nodes(i))/pow(2,depth))-time_nodes(k))/(time_nodes(j)-time_nodes(k));
						}
					}

				}
			}

			for (size_t i = 0; i <Nb*system_order/2; i++){// new planet positions
				for(size_t j = 0; j<method_stage; j++){
					par1[j*Nb*system_order/2+i]=0;
					for(size_t k = 0; k <method_stage; k++){
						par1[j*Nb*system_order/2+i]+=par[Nb*system_order*method_stage/2*step+ Nb*system_order/2*k +i]*time_nodes1(j,k);
					}
				}
			}

		};*/

	for (size_t i = 0; i < system_order; i++)
	{
		y(i) = dinputdata[pos * system_order + i]; // свои нач данные у каждого потока
	}

	for (size_t i = 0; i < method_stage; i++)
		time_nodes(i) = dtime_nodes[i];

	for (size_t i = 0; i < method_stage; i++)
	{ // тут нужно заполнять с другой стороны из-за странной работы с указателем
		for (size_t j = 0; j < method_stage; j++)
		{
			inv_lsm(i, j) = dinv_lsm[j * method_stage + i];
			sv_nodes(i, j) = dsv_nodes[j * method_stage + i];
		}
	}


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

		do_step(y, t, h, 0, parloc);

		// do_step(y,t,h,step,par);

		// variable step
		//====================================================================

		/*H=E_full(y, step+1); H1=E_full(y_prev,step);



		if( abs(1-H/H1)>=dE){ // energy bounds for variable step
			//printf("%i %i %f ", pos, step, H/H1);
			int depth=1;



			do{//depth = number of times  h1=h/2
				y_temp=y_prev;


				for (int substep = 0; substep < pow(2,depth); substep++){

					make_new_par(par1, step, substep, depth);

					h1=h/pow(2,depth);

					do_step(y_temp,t+substep*h1,h1,0,par1);
					}

					// temp check
				H=E_full(y_temp, step+1); H1=E_full(y_prev,step);
				//printf("%i %i %f ", pos, step, H/H1);

				y=y_temp;
				depth+=1;
				//printf("%i %i \n",depth, maxdepth);

			}while( (depth != maxdepth) && ( abs(1-H/H1)>=dE) );//  or energy condition is met

			//printf("\n");
		}*/

		t += h;

		if ((step % skipstep) == 0)
		{
			for (size_t j = 0; j < system_order; j++)
				doutputdata[(step / skipstep + j * maxstep / skipstep) + pos * maxstep * system_order / skipstep] = y(j);
			// output tables where every column = system_order, every row=maxstep, pos=number of table
		}
	}
}

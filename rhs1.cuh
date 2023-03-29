
// y= 3 координаты + 3 скорости одной частицы
// x- Au; V - au/s;
// par-вектор параметров(положения всех других планет на момент t)
// par одномерный вектор координат, по 3 значения на каждый объект
__device__ void rhs(double *y, double *yd, double t, double *par, size_t step, size_t iternum)
{

	constexpr int Nb = 13; // число тел в par
	constexpr int dim = 6; // размерность вектора
	// const double au=149597870;
	constexpr int method_stage = 7;

	double R3;

	double Gm[Nb] = {132712440041.279419, 22031.868551, 324858.592000, 398600.435507, 42828.375816, 126712764.100000, 37940584.841800,
					 5794556.400000, 6836527.100580, 975.500000, 62.62890, 17.288245, 4902.800118}; // km^3 s^-2

	for (size_t i = 0; i < Nb; i++)
	{
		Gm[i] *= 86400.0 * 86400 / pow(1.495978707e8, 3);
	}

	Eigen::Matrix<double, dim, 1> y0;
	Eigen::Matrix<double, dim, 1> yd0;
	yd0.setZero();

	for (size_t i = 0; i < dim; i++)
	{
		y0(i) = y[i];
	}

	for (size_t i = 0; i < dim / 2; i++)
	{
		yd0(i) = y0(i + 3); // скорости частицы
	}

	for (size_t i = 0; i < Nb; i++)
	{
		R3 = pow(
			pow((y0(0) - par[iternum * Nb * dim / 2 + Nb * dim / 2 * step * method_stage + i * dim / 2]), 2) +
				pow((y0(1) - par[iternum * Nb * dim / 2 + Nb * dim / 2 * step * method_stage + i * dim / 2 + 1]), 2) +
				pow((y0(2) - par[iternum * Nb * dim / 2 + Nb * dim / 2 * step * method_stage + i * dim / 2 + 2]), 2),
			3.0 / 2);

		// printf("%e \n",y0(0));

		// 2 за занятые 2 элемента, Nb*dim/2*step нужный шаг в таблице, i*dim/2 за нужный объект

		yd0(3) += -Gm[i] * (y0(0) - par[iternum * Nb * dim / 2 + Nb * dim / 2 * step * method_stage + i * dim / 2]) / (R3);
		yd0(4) += -Gm[i] * (y0(1) - par[iternum * Nb * dim / 2 + Nb * dim / 2 * step * method_stage + i * dim / 2 + 1]) / (R3);
		yd0(5) += -Gm[i] * (y0(2) - par[iternum * Nb * dim / 2 + Nb * dim / 2 * step * method_stage + i * dim / 2 + 2]) / (R3);
	}
	// non gravity part
	const double S0 = 1.3611e6;	  // solar constant g/s^3
	const double r0 = 3;		  // r of meteoroid cm
	const double m0 = 16;		  // mass of meteoroid g
	const double c = 299792458e2; // speed of light cm/c
	const double pi = atan(1);

	Eigen::Matrix<double, dim / 2, 1> y_sun;
	Eigen::Matrix<double, dim / 2, 1> dv;
	Eigen::Matrix<double, dim / 2, 1> v;
	Eigen::Matrix<double, dim / 2, 1> S;
	v(0) = y0[3];
	v(1) = y0[4];
	v(2) = y0[5];

	y_sun(0) = par[iternum * Nb * dim / 2 + Nb * dim / 2 * step * method_stage];
	y_sun(1) = par[iternum * Nb * dim / 2 + Nb * dim / 2 * step * method_stage + 1];
	y_sun(2) = par[iternum * Nb * dim / 2 + Nb * dim / 2 * step * method_stage + 2];

	double R_sun = pow(pow(y0(0) - y_sun(0), 2) + pow(y0(1) - y_sun(1), 2) + pow(y0(2) - y_sun(2), 2), 1. / 2);

	S(0) = (y0[0] - y_sun(0)) / R_sun;
	S(1) = (y0[1] - y_sun(1)) / R_sun;
	S(2) = (y0[2] - y_sun(2)) / R_sun;

	double v_r = v.dot(y_sun) / R_sun;

	dv = S0 * 2 * pi * r0 * r0 * 86400.0 * 86400 / (c * m0 * R_sun * R_sun * 1.495978707e13) * ((1 - v_r * 86400 / (c * 1.495978707e13)) * S - v * 86400 / (c * 1.495978707e13));

	yd0(3) += dv(0);
	yd0(4) += dv(1);
	yd0(5) += dv(2);

	for (size_t i = 0; i < dim; i++)
	{
		yd[i] = yd0(i);
	}
}

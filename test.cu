#include "collo.cuh"


using namespace std;

int main()
{
    
    constexpr size_t method_stage = 7;
    constexpr size_t system_order = 6;

    constexpr size_t Nb = 13;

    constexpr size_t maxstep = 365;
    constexpr size_t skipstep = 150; // записывать данные каждые skipstep шагов
    // skipstep должно быть делителем maxstep

    // constexpr size_t blocks = 878; constexpr size_t threads = 32;

    constexpr size_t blocks = 2;
    constexpr size_t threads = 32;

    constexpr size_t plength = Nb * 3 * maxstep * method_stage; // длина вектора параметров

    // maxstep = 1000; 3 за размерность

    double h = -1;
    double t0 = 0.0;

    std::ifstream ind;
    ind.open("input/input.dat"); //par input
    std::ifstream indata("input/coords.csv");//in data input


    double *par = new double[plength]; // вектор параметров

    for (size_t i = 0; i < plength; i++)
        ind >> par[i];

    // входной вектор y0 длины system_order*blocks*threads, разрезается на нужные куски длины system_order уже внутри интегратора



    double y0[system_order * blocks * threads];

    

    for (size_t i = 0; i < blocks * threads * system_order; i++)
    {
        indata >> y0[i];
        // std::cout<<y0(i)<<std::endl;
    }

    collocation<double, system_order, method_stage> test(y0, t0, h, maxstep, skipstep, par, plength, blocks, threads);
    std::cout << "copying done" << std::endl;

    test.do_steps();
    test.write();



    delete par;
}

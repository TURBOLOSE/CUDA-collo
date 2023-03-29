#include "kernel.cuh"

template <typename num_t, size_t system_order, size_t method_stage>
class collocation
{

private:
    Eigen::Matrix<num_t, method_stage, method_stage> inv_lsm;
    Eigen::Matrix<num_t, method_stage, 1> time_nodes;
    Eigen::Matrix<num_t, method_stage, method_stage> sv_nodes;
    num_t *inv_lsmd;
    num_t *time_nodesd;
    num_t *sv_nodesd;
    num_t *y0d;
    num_t *pard;
    num_t *outd;

public:
    num_t *y0;
    // Eigen::Matrix<num_t, maxstep/skipstep, system_order> out;
    num_t *out;
    // std::optional<num_t *>  par;
    // std::optional<size_t> plength;
    num_t *par;
    size_t plength;
    num_t h, t0;
    size_t maxstep, skipstep, blocks, threads;

    collocation(num_t *y0p, num_t t0h, num_t hp,
                size_t maxstep, size_t skipstep, std::optional<num_t *> parp, std::optional<size_t> plength0, size_t blocks, size_t threads) : y0(y0p), par(parp.value()), t0(t0h), h(hp), maxstep(maxstep), skipstep(skipstep), plength(plength0.value()), blocks(blocks), threads(threads)
    {
        using matrix_t = Eigen::Matrix<double, method_stage, method_stage>;
        out = new num_t[maxstep * system_order * blocks * threads / skipstep];
        if(!parp) // if no parameters passed
        plength=0;
        /*std::ifstream ind("temp.dat");
        Eigen::Matrix<num_t, method_stage, method_stage> inv_lsm0;
        Eigen::Matrix<num_t, method_stage, 1> time_nodes0;
        Eigen::Matrix<num_t, method_stage, method_stage> sv_nodes0;

        for (size_t i = 0; i < method_stage; i++) // ввод констант
            ind >> time_nodes0(i);

        for (size_t i = 0; i < method_stage; i++)
        {
            for (size_t j = 0; j < method_stage; j++)
                ind >> sv_nodes0(i, j);
        }

        for (size_t i = 0; i < method_stage; i++)
        {
            for (size_t j = 0; j < method_stage; j++)
                ind >> inv_lsm0(i, j);
        }*/

        Eigen::Matrix<num_t, method_stage, 1> time_nodes;
        auto time_nodes0 = numm::roots_ilegendre_sh<method_stage - 1, double>();
        time_nodes=Eigen::Matrix<num_t, method_stage, 1>(time_nodes0.data());

        auto sv_nodes0 = numm::legendre_sh<method_stage>(time_nodes0);
        auto base = numm::legendre_sh<method_stage>(0.0);
        for (auto &row : sv_nodes0)
            for (std::size_t i = 0; i <= method_stage; ++i)
                row[i] -= base[i];


        Eigen::Matrix<num_t, method_stage, method_stage> sv_nodes;
        for (size_t i = 0; i < method_stage; i++)
        {
            for (size_t j = 0; j < method_stage; j++)
                sv_nodes(i, j)=sv_nodes0[i][j];
        }
        



        std::array<double, method_stage * method_stage> lsm{};
        auto it = std::begin(lsm);
        for (auto &row : numm::dlegendre_sh<method_stage>(time_nodes0))
            it = std::copy(std::next(row.begin()), row.end(), it);
        static const matrix_t inv_lsm = matrix_t(lsm.data()).inverse();

  

        cudaMalloc((void **)&time_nodesd, sizeof(Eigen::Matrix<num_t, method_stage, 1>)); // выделение памяти
        cudaMalloc((void **)&sv_nodesd, sizeof(Eigen::Matrix<num_t, method_stage, method_stage>));
        cudaMalloc((void **)&inv_lsmd, sizeof(Eigen::Matrix<num_t, method_stage, method_stage>));
        cudaMalloc((void **)&pard, sizeof(num_t) * plength);
        cudaMalloc((void **)&outd, sizeof(num_t) * maxstep * system_order * blocks * threads / skipstep);
        // cudaMalloc((void**)&y0d, sizeof (Eigen::Matrix<num_t, system_order*blocks*threads, 1>));
        cudaMalloc((void **)&y0d, sizeof(num_t) * system_order * blocks * threads);

        cudaMemcpy(time_nodesd, time_nodes.data(), sizeof(Eigen::Matrix<num_t, method_stage, 1>), cudaMemcpyKind::cudaMemcpyHostToDevice); // копирование
        cudaMemcpy(sv_nodesd, sv_nodes.data(), sizeof(Eigen::Matrix<num_t, method_stage, method_stage>), cudaMemcpyKind::cudaMemcpyHostToDevice);
        cudaMemcpy(inv_lsmd, inv_lsm.data(), sizeof(Eigen::Matrix<num_t, method_stage, method_stage>), cudaMemcpyKind::cudaMemcpyHostToDevice);
        cudaMemcpy(pard, par, sizeof(num_t) * plength, cudaMemcpyKind::cudaMemcpyHostToDevice);
        // cudaMemcpy(y0d, y0.data(), sizeof(Eigen::Matrix<num_t, system_order*blocks*threads, 1>), cudaMemcpyKind::cudaMemcpyHostToDevice);
        cudaMemcpy(y0d, y0, sizeof(num_t) * system_order * blocks * threads, cudaMemcpyKind::cudaMemcpyHostToDevice);
    };

    void do_steps()
    {

        kernel<num_t, system_order, method_stage><<<blocks, threads>>>(y0d, t0, outd, pard, time_nodesd, sv_nodesd, inv_lsmd, h, maxstep, skipstep);

        cudaDeviceSynchronize();
        //=========================================
        /*int maxActiveBlocks;
        int device;
        cudaDeviceProp props;
        int minGridSize;
        int blockSize=threads;


        cudaOccupancyMaxActiveBlocksPerMultiprocessor( &maxActiveBlocks,  kernel<double,system_order, method_stage, maxstep, skipstep>, blockSize, 0);

        //cudaOccupancyMaxPotentialBlockSize( &minGridSize, &blockSize, kernel<double,system_order, method_stage, maxstep, skipstep>, 0, 0);
        cudaGetDevice(&device);
        cudaGetDeviceProperties(&props, device);
        float occupancy = (maxActiveBlocks * blockSize / props.warpSize) /
                        (float)(props.maxThreadsPerMultiProcessor /
                                props.warpSize);

        std::cout<<blockSize<<std::endl;
        std::cout<<minGridSize<<std::endl;

             printf("Launched blocks of size %d. Theoretical occupancy: %f\n", blockSize, occupancy);*/
        //=========================================

        cudaError_t c_ret;
        c_ret = cudaGetLastError();
        if (c_ret)
            std::cout << "Error: " << cudaGetErrorString(c_ret) << "-->";

        // cudaMemcpy(out.data(), outd,  sizeof (Eigen::Matrix<double, maxstep/skipstep, system_order>), cudaMemcpyDeviceToHost);
        cudaMemcpy(out, outd, sizeof(num_t) * maxstep * system_order * blocks * threads / skipstep, cudaMemcpyDeviceToHost);
    }

    void print()
    {

        for (size_t i = 0; i < maxstep / skipstep; i++)
        {
            // cout<<t0+(i+1)*h;
            for (size_t j = 0; j < system_order; j++)
            {
                // std::cout<<" "<<out(i,j);
                std::cout << " " << out[i + j * maxstep / skipstep];
            }
            std::cout << std::endl;
        }
    }

    void write()
    {

        std::ofstream outfile[blocks * threads];

        std::string path0 = "result/res";
        std::string path1 = ".dat";

        for (size_t nres = 0; nres < blocks * threads; nres++) // nres -- number of res_n file
        {

            // out is 3 tensor with dimenions: maxstep/skipstep; system_order; blocks*threads
            outfile[nres].open(path0 + std::to_string(nres) + path1);
            for (size_t i = 0; i < maxstep / skipstep; i++)
            {

                for (size_t j = 0; j < system_order; j++)
                {
                    outfile[nres] << " " << out[i + j * maxstep / skipstep + nres * maxstep * system_order / skipstep];
                }
                outfile[nres] << "\n";
            }
            outfile[nres].close();
        }
    }

    void write_last_steps()
    {

        std::ofstream outfile;

        std::string path0 = "result/res_full.dat";

        outfile.open(path0);
        for (size_t nres = 0; nres < blocks * threads; nres++) // nres -- number of res_n file
        {
            for (size_t j = 0; j < system_order; j++)
            {
                outfile << " " << out[((maxstep - 1) / skipstep) + j * maxstep / skipstep + nres * maxstep * system_order / skipstep];
                // doutputdata[(step/skipstep +  j*maxstep/skipstep)+pos*maxstep*system_order/skipstep]=y(j);
                // output tables where every column = system_order, every row=maxstep, pos=number of table
            }
            outfile << "\n";
        }
        outfile.close();
    }
};
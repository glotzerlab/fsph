
#include "tensorflow_op_gpu.cu.hpp"
#include "internal.hpp"
#include "../cuda_complex/cuda_complex.hpp"

namespace fsph{
    template<typename Real, typename Complex>
    __global__ void SphericalHarmonicSeriesKernel(
        const Real *__restrict__ phitheta, const unsigned int N,
        const unsigned int lmax, const bool full_m,
        Complex *__restrict__ output)
    {
        if(blockIdx.x*blockDim.y + threadIdx.y > N)
            return;

        const Real phi(phitheta[2*(blockIdx.x*blockDim.y + threadIdx.y)]);
        Real sphi, cphi;
        sincosf(phi, &sphi, &cphi);
        const Real theta(phitheta[2*(blockIdx.x*blockDim.y + threadIdx.y) + 1]);

        unsigned int m(threadIdx.x);
        unsigned int l(m);
        fsph::internal::SPHSeriesEvaluator<Real, Complex> eval(m);

        for(unsigned int i(0); i < lmax + 1; i++)
        {
            if(l > lmax)
            {
                m = lmax - threadIdx.x;
                l = m;
                eval.reset(m);
            }

            const Complex Ylm(eval.get(sphi, cphi, theta, l));

            const unsigned int sph_per_point(
                full_m? (lmax + 1)*(lmax + 2)/2 + lmax*(lmax + 1)/2:
                        (lmax + 1)*(lmax + 2)/2);
            const unsigned int sph_index(
                (blockIdx.x*blockDim.y + threadIdx.y)*sph_per_point +
                l*(l + 1)/2 + m);
            output[sph_index] = Ylm;

            if(full_m)
                output[sph_index + l] = eval.get_negative_m(Ylm, theta);

            l++;
        }
    }

    template<typename Real, typename Complex>
    void SphericalHarmonicSeriesKernelLauncher(
        const Real *phitheta, const unsigned int N,
        const unsigned int lmax, const bool full_m,
        Complex *output)
    {
        const unsigned int points_per_block(32);
        const unsigned int num_blocks((N + points_per_block - 1)/points_per_block);
        const dim3 block_dim(lmax + 1, points_per_block);

        SphericalHarmonicSeriesKernel<Real, Complex><<<num_blocks, block_dim>>>(
            phitheta, N, lmax, full_m, output);
    }
}

template void fsph::SphericalHarmonicSeriesKernelLauncher<float, complex<float>>(
    const float *phitheta, const unsigned int N,
    const unsigned int lmax, const bool full_m,
    complex<float> *output);

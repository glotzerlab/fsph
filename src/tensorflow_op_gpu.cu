
#include "tensorflow_op_gpu.cu.hpp"
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
        Real jacobi_before_last(0);

        Real last_jacobi(1);
        for(unsigned int i(1); i <= m; i++)
            last_jacobi *= 1 + 0.5/i;
        last_jacobi = sqrt(last_jacobi);
        last_jacobi /= sqrt(2.0);

        for(unsigned int i(0); i < lmax + 1; i++)
        {
            if(l > lmax)
            {
                m = lmax - threadIdx.x;
                l = m;
                jacobi_before_last = 0;
                last_jacobi = 1;
                for(unsigned int j(1); j <= m; j++)
                    last_jacobi *= 1 + 0.5/j;
                last_jacobi = sqrt(last_jacobi);
                last_jacobi /= sqrt(2.0);
            }

            const Real k(l - m);
            const Real prefactor_1(
                2*sqrt((1.0 + (m - 0.5)/k)*(1.0 - (m - 0.5)/(k + 2.0*m))));
            const Real prefactor_2(
                sqrt((1.0 + 4.0/(2.0*k + 2.0*m - 3.0))*(1.0 - 1.0/k)*(1.0 - 1.0/(k + 2.0*m))));
            const Real jacobi(
                prefactor_1*cphi*last_jacobi - prefactor_2*jacobi_before_last);

            Complex Ylm(pow(sphi, m)/sqrt(2*M_PI)*jacobi);
            Ylm *= exp(Complex(0, m*theta));

            jacobi_before_last = last_jacobi;
            last_jacobi = jacobi;

            const unsigned int sph_per_point(
                full_m? (lmax + 1)*(lmax + 2)/2 + lmax*(lmax + 1)/2:
                        (lmax + 1)*(lmax + 2)/2);
            const unsigned int sph_index(
                (blockIdx.x*blockDim.y + threadIdx.y)*sph_per_point +
                l*(l + 1)/2 + m);
            output[sph_index] = Ylm;

            if(full_m)
                output[sph_index + l] = Ylm*exp(Complex(0, -2.0*m*theta));

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

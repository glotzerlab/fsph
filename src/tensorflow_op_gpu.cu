
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

    template<typename Real, typename Complex>
    __global__ void SphericalHarmonicSeriesGradKernel(
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
        fsph::internal::SPHSeriesEvaluator<Real, Complex> evalmp1(m + 1);

        for(unsigned int i(0); i < lmax + 1; i++)
        {
            if(l > lmax)
            {
                m = lmax - threadIdx.x;
                l = m;
                eval.reset(m);
                evalmp1.reset(m + 1);
            }

            const Complex Ylm(eval.get(sphi, cphi, theta, l));
            const Complex Ylmp1(evalmp1.get(sphi, cphi, theta, l));

            const unsigned int sph_per_point(
                full_m? (lmax + 1)*(lmax + 2)/2 + lmax*(lmax + 1)/2:
                        (lmax + 1)*(lmax + 2)/2);
            const unsigned int sph_index(
                (blockIdx.x*blockDim.y + threadIdx.y)*sph_per_point +
                l*(l + 1)/2 + m);
            output[2*sph_index] = eval.grad_phi(cphi, sphi, theta, l, Ylm, Ylmp1, full_m);
            output[2*sph_index + 1] = eval.grad_theta(Ylm, full_m);

            if(full_m)
            {
                const Complex Ylnegm(eval.get_negative_m(Ylm, theta));
                const Complex Ylnegmp1(eval.get_negative_m(Ylmp1, theta));

                output[2*(sph_index + l)] = eval.grad_phi(cphi, sphi, theta, l, Ylnegm, Ylnegmp1, full_m);
                output[2*(sph_index + l) + 1] = eval.grad_theta(Ylnegm, full_m);
            }

            l++;
        }
    }

    template<typename Real, typename Complex>
    void SphericalHarmonicSeriesGradKernelLauncher(
        const Real *phitheta, const unsigned int N,
        const unsigned int lmax, const bool full_m,
        Complex *output)
    {
        const unsigned int points_per_block(32);
        const unsigned int num_blocks((N + points_per_block - 1)/points_per_block);
        const dim3 block_dim(lmax + 1, points_per_block);

        SphericalHarmonicSeriesGradKernel<Real, Complex><<<num_blocks, block_dim>>>(
            phitheta, N, lmax, full_m, output);
    }
}

template void fsph::SphericalHarmonicSeriesKernelLauncher<float, complex<float>>(
    const float *phitheta, const unsigned int N,
    const unsigned int lmax, const bool full_m,
    complex<float> *output);

template void fsph::SphericalHarmonicSeriesGradKernelLauncher<float, complex<float>>(
    const float *phitheta, const unsigned int N,
    const unsigned int lmax, const bool full_m,
    complex<float> *output);


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
#define POINT_INDEX (blockIdx.x*blockDim.y + threadIdx.y)
#define INITIAL_M (blockIdx.y*blockDim.x + threadIdx.x)
        if(POINT_INDEX > N || INITIAL_M > (lmax + 1)/2 + 1)
            return;

        const Real phi(phitheta[2*POINT_INDEX]);
        Real sphi, cphi;
        sincosf(phi, &sphi, &cphi);
        const Real theta(phitheta[2*POINT_INDEX + 1]);

        unsigned int m(INITIAL_M);
        unsigned int l(m);
        fsph::internal::SPHSeriesEvaluator<Real, Complex> eval(m, cphi, sphi, theta, false);

        for(unsigned int i(0); i < lmax + 1; i++)
        {
            if(l > lmax)
            {
                m = lmax - INITIAL_M + 1;
                l = m;
                eval.reset(m, cphi, sphi, theta, false);
            }

            const Complex Ylm(eval.get(l));

            const unsigned int sph_per_point(
                (lmax + 1)*(lmax + 2)/2 + full_m*lmax*(lmax + 1)/2);
            const unsigned int sph_index(
                POINT_INDEX*sph_per_point +
                l*(l + 1)/2 + full_m*(l - 1)*l/2 + m);
            output[sph_index] = Ylm;

            if(full_m && m)
                output[sph_index + l] = eval.get_negative_m(Ylm);

            l++;
        }
    }

    template<typename Real>
    void SphericalHarmonicSeriesKernelLauncher(
        const Real *phitheta, const unsigned int N,
        const unsigned int lmax, const bool full_m,
        std::complex<Real> *output)
    {
        const unsigned int points_per_block(32);
        const unsigned int num_blocks((N + points_per_block - 1)/points_per_block);
        const unsigned int m_workers_needed((lmax + 1)/2 + 1);
        const dim3 block_dim(min(m_workers_needed, 1024/points_per_block), points_per_block);
        const dim3 grid_dim(num_blocks, (m_workers_needed + block_dim.x - 1)/block_dim.x);

        SphericalHarmonicSeriesKernel<Real, complex<Real> ><<<grid_dim, block_dim>>>(
            phitheta, N, lmax, full_m, (complex<Real>*) output);
    }

    template<typename Real, typename Complex>
    __global__ void SphericalHarmonicSeriesGradKernel(
        const Real *__restrict__ phitheta, const unsigned int N,
        const unsigned int lmax, const bool full_m,
        Complex *__restrict__ output)
    {
#define POINT_INDEX (blockIdx.x*blockDim.y + threadIdx.y)
#define INITIAL_M (blockIdx.y*blockDim.x + threadIdx.x)
        if(POINT_INDEX > N || INITIAL_M > (lmax + 1)/2 + 1)
            return;

        const Real phi(phitheta[2*POINT_INDEX]);
        Real sphi, cphi;
        sincosf(phi, &sphi, &cphi);
        const Real theta(phitheta[2*POINT_INDEX + 1]);

        unsigned int m(INITIAL_M);
        unsigned int l(m);
        fsph::internal::SPHSeriesEvaluator<Real, Complex> eval(m, cphi, sphi, theta, true);
        fsph::internal::SPHSeriesEvaluator<Real, Complex> evalmp1(m + 1, cphi, sphi, theta, false);
        fsph::internal::SPHSeriesEvaluator<Real, Complex> evalmm1(m? m - 1: 0, cphi, sphi, theta, false);

        for(unsigned int i(0); i < lmax + 1; i++)
        {
            if(l > lmax)
            {
                m = lmax - INITIAL_M + 1;
                l = m;
                eval.reset(m, cphi, sphi, theta, true);
                evalmp1.reset(m + 1, cphi, sphi, theta, false);
                evalmm1.reset(m? m - 1: 0, cphi, sphi, theta, false);
            }

            const Complex Ylm(eval.get(l));
            const Complex Ylmp1(evalmp1.get(l));

            const unsigned int sph_per_point(
                (lmax + 1)*(lmax + 2)/2 + full_m*lmax*(lmax + 1)/2);
            const unsigned int sph_index(
                POINT_INDEX*sph_per_point +
                l*(l + 1)/2 + full_m*(l - 1)*l/2 + m);
            output[2*sph_index] = eval.grad_phi(l, Ylm, Ylmp1, false);
            output[2*sph_index + 1] = eval.grad_theta(Ylm, false);

            if(full_m && m)
            {
                const Complex Ylmm1(evalmm1.get(l));
                const Complex Ylnegm(eval.get_negative_m(Ylm));
                const Complex Ylnegmp1(evalmm1.get_negative_m(Ylmm1));

                output[2*(sph_index + l)] = eval.grad_phi(l, Ylnegm, Ylnegmp1, true);
                output[2*(sph_index + l) + 1] = eval.grad_theta(Ylnegm, true);
            }

            l++;
        }
    }

    template<typename Real>
    void SphericalHarmonicSeriesGradKernelLauncher(
        const Real *phitheta, const unsigned int N,
        const unsigned int lmax, const bool full_m,
        std::complex<Real> *output)
    {
        const unsigned int points_per_block(32);
        const unsigned int num_blocks((N + points_per_block - 1)/points_per_block);
        const unsigned int m_workers_needed((lmax + 1)/2 + 1);
        const dim3 block_dim(min(m_workers_needed, 1024/points_per_block), points_per_block);
        const dim3 grid_dim(num_blocks, (m_workers_needed + block_dim.x - 1)/block_dim.x);

        SphericalHarmonicSeriesGradKernel<Real, complex<Real> ><<<grid_dim, block_dim>>>(
            phitheta, N, lmax, full_m, (complex<Real>*) output);
    }
}

template void fsph::SphericalHarmonicSeriesKernelLauncher<float>(
    const float *phitheta, const unsigned int N,
    const unsigned int lmax, const bool full_m,
    std::complex<float> *output);

template void fsph::SphericalHarmonicSeriesGradKernelLauncher<float>(
    const float *phitheta, const unsigned int N,
    const unsigned int lmax, const bool full_m,
    std::complex<float> *output);

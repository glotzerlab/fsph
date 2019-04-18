
#include "tensorflow_op_gpu.cu.hpp"
#include "internal.hpp"

#define THETA_HARMONICS_OFFSET 0
#define RECURRENCE_PREFACTORS_OFFSET THETA_HARMONICS_OFFSET + lmax + 1
#define SIN_POWERS_OFFSET RECURRENCE_PREFACTORS_OFFSET + 2*(lmax + 1)*lmax
#define JACOBI_OFFSET SIN_POWERS_OFFSET + (lmax + 1)*blockDim.x
#define LEGENDRE_OFFSET JACOBI_OFFSET + (lmax + 1)*(lmax + 1)*blockDim.x

namespace fsph{
    template<typename Real, typename Complex>
    __global__ void SphericalHarmonicSeriesKernel(
        const Real *__restrict__ phitheta, const unsigned int N,
        const unsigned int lmax, const bool full_m,
        Complex *__restrict__ output)
    {
        extern __shared__ Real sh[];

        if(blockIdx.x*blockDim.x + threadIdx.x > N)
            return;

        // we are writing the same info multiple times, but the
        // alternative is a syncthreads
        internal::evaluatePrefactors<Real, Complex>(lmax, &sh[RECURRENCE_PREFACTORS_OFFSET]);
        const Real phi(phitheta[2*(blockIdx.x*blockDim.x + threadIdx.x)]);
        Real sphi, cphi;
        sincosf(phi, &sphi, &cphi);

        internal::compute_sinpows<Real, Complex>(
            sphi, lmax, &sh[SIN_POWERS_OFFSET + (lmax + 1)*threadIdx.x]);

        const Real theta(phitheta[2*(blockIdx.x*blockDim.x + threadIdx.x) + 1]);

        internal::compute_thetaHarmonics<Real, Complex>(
            theta, lmax, (Complex*) &sh[THETA_HARMONICS_OFFSET + (lmax + 1)*threadIdx.x]);

        internal::compute_jacobis<Real, Complex>(
            cphi, lmax, &sh[RECURRENCE_PREFACTORS_OFFSET],
            &sh[JACOBI_OFFSET + (lmax + 1)*(lmax + 1)*threadIdx.x]);

        internal::compute_legendres<Real, Complex>(
            lmax, &sh[SIN_POWERS_OFFSET + (lmax + 1)*threadIdx.x],
            &sh[JACOBI_OFFSET + (lmax + 1)*(lmax + 1)*threadIdx.x],
            &sh[LEGENDRE_OFFSET + internal::sphCount(lmax)*threadIdx.x]);

        unsigned int l(0), m(0), num_sphs(0);
        const unsigned int sph_stride(
            full_m? (lmax + 1)*(lmax + 2)/2 + lmax*(lmax + 1)/2:
                    (lmax + 1)*(lmax + 2)/2);

        while(l <= lmax)
        {
            const unsigned int sph_index(
                sph_stride*(blockIdx.x*blockDim.x + threadIdx.x) + num_sphs++);
            output[sph_index] = internal::iterator_get<Real, Complex>(
                l, m, &sh[LEGENDRE_OFFSET + internal::sphCount(lmax)*threadIdx.x],
                (Complex*) &sh[THETA_HARMONICS_OFFSET + (lmax + 1)*threadIdx.x]);

            internal::iterator_increment<Real, Complex>(full_m, l, m);
        }
    }

    template<typename Real, typename Complex>
    void SphericalHarmonicSeriesKernelLauncher(
        const Real *phitheta, const unsigned int N, const unsigned int points_per_block,
        const unsigned int lmax, const bool full_m, const unsigned int shm_size,
        Complex *output)
    {
        const unsigned int num_blocks((N + points_per_block - 1)/points_per_block);

        // blockIdx.x*blockDim.x + threadIdx.x: point index
        SphericalHarmonicSeriesKernel<Real, Complex><<<num_blocks, points_per_block, shm_size>>>(
            phitheta, N, lmax, full_m, output);
    }
}

template void fsph::SphericalHarmonicSeriesKernelLauncher<float, complex<float>>(
    const float *phitheta, const unsigned int N, const unsigned int points_per_block,
    const unsigned int lmax, const bool full_m, const unsigned int shm_size,
    complex<float> *output);

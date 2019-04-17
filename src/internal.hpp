

//#include <math.h>

#ifndef __FSPH_INTERNAL__
#define __FSPH_INTERNAL__

#ifndef __NVCC__
#define __device__
#define __host__
#include <complex>
#else
#include "../cuda_complex/cuda_complex.hpp"
#endif

namespace fsph{
    namespace internal{
        template<typename T>
        __device__ inline T index2d(const T &w, const T &i, const int &j)
        {
            return w*i + j;
        }

        template<typename T>
        __host__ __device__ inline T sphCount(const T &lmax)
        {
            return (lmax + 1)*(lmax + 2)/2;
        }

        template<typename T>
        __device__ inline T sphIndex(const T &l, const T &m)
        {
            if(l > 0)
                return sphCount(l - 1) + m;
            else
                return 0;
        }

        template<typename Real, typename Complex>
        __device__ inline void evaluatePrefactors(const unsigned int lmax, Real *recurrencePrefactors)
        {
            const unsigned int f1Count(index2d(lmax, lmax + 1, 0));

            for(unsigned int m(0); m < lmax + 1; ++m)
            {
                for(unsigned int l(1); l < lmax + 1; ++l)
                {
                    const unsigned int idx = index2d(lmax, m, l - 1);
                    recurrencePrefactors[idx] =
                        2*sqrt(1 + (m - 0.5)/l)*sqrt(1 - (m - 0.5)/(l + 2*m));
                }
            }

            for(unsigned int m(0); m < lmax + 1; ++m)
            {
                recurrencePrefactors[f1Count + index2d(lmax, m, 0)] = 0;
                for(unsigned int l(2); l < lmax + 1; ++l)
                {
                    const unsigned int idx = f1Count + index2d(lmax, m, l - 1);
                    recurrencePrefactors[idx] =
                        -sqrt(1.0 + 4.0/(2*l + 2*m - 3))*sqrt(1 - 1.0/l)*sqrt(1.0 - 1.0/(l + 2*m));
                }
            }
        }

        template<typename Real, typename Complex>
        __device__ inline void compute_sinpows(const Real &sphi, const unsigned int lmax, Real *sinPowers)
        {
            sinPowers[0] = 1;
            for(unsigned int i(1); i < lmax + 1; ++i)
                sinPowers[i] = sinPowers[i - 1]*sphi;
        }

        template<typename Real, typename Complex>
        __device__ inline void compute_thetaHarmonics(const Real &theta, const unsigned int lmax, Complex *thetaHarmonics)
        {
            thetaHarmonics[0] = Complex(1, 0);
            for(unsigned int i(0); i < lmax + 1; ++i)
                thetaHarmonics[i] = exp(Complex(0, i*theta));
        }

        template<typename Real, typename Complex>
        __device__ inline void compute_jacobis(const Real &cphi, const unsigned int lmax, const Real *recurrencePrefactors, Real *jacobi)
        {
            const unsigned int f1Count(index2d(lmax, lmax + 1, 0));

            for(unsigned int m(0); m < lmax + 1; ++m)
            {
                if(m > 0)
                    jacobi[index2d(lmax + 1, m, 0)] =
                        jacobi[index2d(lmax + 1, m - 1, 0)]*sqrt(1 + 1.0/2/m);
                else
                    jacobi[index2d(lmax + 1, (unsigned int) 0, 0)] = 1/sqrt(2.0);

                if(lmax > 0)
                    jacobi[index2d(lmax + 1, m, 1)] =
                        cphi*recurrencePrefactors[index2d(lmax, m, 0)]*jacobi[index2d(lmax + 1, m, 0)];

                for(unsigned int l(2); l < lmax + 1; ++l)
                {
                    jacobi[index2d(lmax + 1, m, l)] =
                        (cphi*recurrencePrefactors[index2d(lmax, m, l - 1)]*jacobi[index2d(lmax + 1, m, l - 1)] +
                         recurrencePrefactors[f1Count + index2d(lmax, m, l - 1)]*jacobi[index2d(lmax + 1, m, l - 2)]);
                }
            }
        }

        template<typename Real, typename Complex>
        __device__ inline void compute_legendres(const unsigned int lmax, const Real *sinPowers, const Real *jacobi, Real *legendre)
        {
            for(unsigned int l(0); l < lmax + 1; ++l)
            {
                for(unsigned int m(0); m < l + 1; ++m)
                {
                    legendre[sphIndex(l, m)] = sinPowers[m]*jacobi[index2d(lmax + 1, m, l - m)];
                }
            }
        }

        template<typename Real, typename Complex>
        __device__ inline void iterator_increment(const bool full_m, unsigned int &l, unsigned int &m)
        {
            unsigned int mCount;
            if(full_m)
                mCount = 2*l + 1;
            else
                mCount = l + 1;

            ++m;
            const unsigned int deltaL(m/mCount);
            m %= mCount;
            l += deltaL;
        }

        template<typename Real, typename Complex>
        __device__ inline Complex iterator_get(const unsigned int l, const unsigned int m, const Real *legendre, const Complex *thetaHarmonics)
        {
            // give negative m result
            if(m > l)
            {
                const unsigned int abs_m(m - l);
                return (Complex(legendre[sphIndex(l, abs_m)]/sqrt(2*M_PI))*
                        conj(thetaHarmonics[abs_m]));
            }
            else // positive m
            {
                return (Complex(legendre[sphIndex(l, m)]/sqrt(2*M_PI))*
                        thetaHarmonics[m]);
            }
        }

        template<typename Real, typename Complex>
        __device__ inline Complex iterator_get_grad_phi(const unsigned int l, const unsigned int m_index, const Real phi, const Real theta, const Real *legendre, const Complex *thetaHarmonics)
        {
            const int m(m_index > l? l - m_index: m_index);
            Complex result(m/tan(phi), 0);
            result *= iterator_get(l, m, legendre, thetaHarmonics);

            if(m != (int)l)
            {
                Complex additional(sqrt((Real) (l - m)*(l + m + 1)), 0);
                additional *= exp(Complex(0, -theta));

                unsigned int abs_m(m_index + 1);
                if(m < 0)
                {
                    abs_m = m_index - l - 1;
                    additional *= (Complex(legendre[sphIndex(l, abs_m)]/sqrt(2*M_PI))*
                                   conj(thetaHarmonics[abs_m]));
                }
                else
                {
                    additional *= -(Complex(legendre[sphIndex(l, abs_m)]/sqrt(2*M_PI))*
                                    thetaHarmonics[abs_m]);
                }

                result += additional;
            }

            return result;
        }

        template<typename Real, typename Complex>
        __device__ inline Complex iterator_get_grad_theta(const unsigned int l, const unsigned int m_index, const Real *legendre, const Complex *thetaHarmonics)
        {
            const int m(m_index > l? l - m_index: m_index);
            return Complex(0, m)*iterator_get(l, m, legendre, thetaHarmonics);
        }
    }
}

#endif


#ifndef __FSPH_INTERNAL__
#define __FSPH_INTERNAL__

#ifndef __NVCC__
#define __device__
#define __host__
#endif

namespace fsph{
    namespace internal{

        template<typename Real, typename Complex>
        class SPHSeriesEvaluator
        {
        public:
            __device__ SPHSeriesEvaluator(const unsigned int &m)
            {
                reset(m);
            }

            __device__ void reset(const unsigned int &m)
            {
                m_m = m;
                m_jacobi_before_last = 0;

                m_last_jacobi = 1.0;
                for(unsigned int i(1); i <= m; i++)
                    m_last_jacobi *= 1 + 0.5/i;
                m_last_jacobi = sqrt(m_last_jacobi);
                m_last_jacobi /= sqrt(2.0);
            }

            // NOTE: l *must* be incremented afterward to keep a proper series
            __device__ Complex get(
                const Real &cphi, const Real &sphi, const Real &theta,
                const unsigned int &l)
            {
                const Real k(l - m_m);
                const Real prefactor_1(
                    2*sqrt((1.0 + (m_m - 0.5)/k)*(1.0 - (m_m - 0.5)/(k + 2.0*m_m))));
                const Real prefactor_2(
                    sqrt((1.0 + 4.0/(2.0*k + 2.0*m_m - 3.0))*(1.0 - 1.0/k)*(1.0 - 1.0/(k + 2.0*m_m))));
                const Real jacobi(
                    prefactor_1*cphi*m_last_jacobi - prefactor_2*m_jacobi_before_last);

                Complex Ylm(pow(sphi, m_m)/sqrt(2*M_PI));

                if(l > m_m)
                {
                    Ylm *= jacobi;
                    m_jacobi_before_last = m_last_jacobi;
                    m_last_jacobi = jacobi;
                }
                else
                    Ylm *= m_last_jacobi;

                Ylm *= exp(Complex(0, m_m*theta));

                return Ylm;
            }

            __device__ Complex get_negative_m(
                const Complex &Ylm, const Real &theta) const
            {
                return exp(Complex(0, -2.0*m_m*theta))*Ylm;
            }

            __device__ Complex grad_phi(
                const Real &cphi, const Real &sphi, const Real &theta,
                const unsigned int &l, const Complex &Ylm, const Complex &Ylmp1,
                const bool &negative_m) const
            {
                const int m(negative_m? -m_m: m_m);
                Complex result(m*cphi/sphi, 0);
                result *= Ylm;

                if(m_m != l)
                {
                    Complex additional(sqrt((l - m)*(l + m + 1)), 0);
                    additional *= exp(Complex(0, -theta));
                    additional *= Ylmp1;

                    if(!negative_m)
                        additional *= -1;

                    result += additional;
                }
                else if(negative_m)
                    result *= -1;

                return result;
            }

            __device__ Complex grad_theta(const Complex &Ylm, const bool &negative_m) const
            {
                const int m(negative_m? -m_m: m_m);
                return Complex(0, m)*Ylm;
            }

        private:
            unsigned int m_m;
            Real m_last_jacobi;
            Real m_jacobi_before_last;
        };
    }
}

#endif

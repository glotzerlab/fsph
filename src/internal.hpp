
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
            __device__ SPHSeriesEvaluator(const unsigned int &m, const Real &cphi,
                                          const Real &sphi, const Real &theta,
                                          bool compute_gradient)
            {
                reset(m, cphi, sphi, theta, compute_gradient);
            }

            __device__ void reset(const unsigned int &m, const Real &cphi,
                                  const Real &sphi, const Real &theta,
                                  bool compute_gradient)
            {
                m_m = m;
                m_jacobi_before_last = 0;

                m_last_jacobi = 1.0;
                for(unsigned int i(1); i <= m; i++)
                    m_last_jacobi *= 1 + 0.5/i;
                m_last_jacobi = sqrt(0.5*m_last_jacobi);

                m_Ylm_prefactor = (Complex(pow(sphi, m_m)/sqrt(2*M_PI))*
                                   exp(Complex(0, m_m*theta)));
                m_negative_m_prefactor = exp(Complex(0, -2.0*m_m*theta));
                m_cphi = cphi;

                if(compute_gradient)
                {
                    m_gradphi_theta_rotation = exp(Complex(0, -theta));
                    m_cotphi = cphi/sphi;

                    // disregard gradient when phi == 0, for example
                    if(!isfinite(m_cotphi))
                        m_cotphi = 0;
                }
            }

            // NOTE: l *must* be incremented afterward to keep a proper series
            __device__ Complex get(const unsigned int &l)
            {
                const Real k(l - m_m);
                const Real prefactor_1(
                    2*sqrt((1.0 + (m_m - 0.5)/k)*(1.0 - (m_m - 0.5)/(k + 2.0*m_m))));
                const Real prefactor_2(
                    sqrt((1.0 + 4.0/(2.0*k + 2.0*m_m - 3.0))*(1.0 - 1.0/k)*(1.0 - 1.0/(k + 2.0*m_m))));
                const Real jacobi(
                    prefactor_1*m_cphi*m_last_jacobi - prefactor_2*m_jacobi_before_last);

                Complex Ylm(m_Ylm_prefactor);

                if(l > m_m)
                {
                    Ylm *= jacobi;
                    m_jacobi_before_last = m_last_jacobi;
                    m_last_jacobi = jacobi;
                }
                else
                    Ylm *= m_last_jacobi;

                return Ylm;
            }

            __device__ Complex get_negative_m(const Complex &Ylm) const
            {
                return m_negative_m_prefactor*Ylm;
            }

            __device__ Complex grad_phi(
                const unsigned int &l, const Complex &Ylm, const Complex &Ylmp1,
                const bool &negative_m) const
            {
                const int m(negative_m? -m_m: m_m);
                Complex result(m*m_cotphi, 0);

                result *= Ylm;

                if(m_m != l)
                {
                    Complex additional(sqrt((Real) (l - m)*(l + m + 1)), 0);
                    additional *= m_gradphi_theta_rotation;
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
            Real m_cphi;
            Real m_cotphi;
            Complex m_Ylm_prefactor;
            Complex m_negative_m_prefactor;
            Complex m_gradphi_theta_rotation;
        };
    }
}

#endif

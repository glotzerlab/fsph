
#include "internal.hpp"
#include "SharedArray.hpp"

#ifndef __FSPH_SPHERICAL_HARMONICS__
#define __FSPH_SPHERICAL_HARMONICS__

namespace fsph{

    unsigned int sphCount(const unsigned int lmax)
    {
        return internal::sphCount(lmax);
    }

    template<typename Real>
    class PointSPHEvaluator
    {
    public:

        class iterator: public std::iterator<std::input_iterator_tag, std::complex<Real> >
        {
        public:
            iterator(const PointSPHEvaluator &generator, bool full_m):
                m_generator(generator), m_l(0), m_m(0), m_full_m(full_m)
            {}

            iterator(const iterator &rhs):
                m_generator(rhs.m_generator), m_l(rhs.m_l), m_m(rhs.m_m), m_full_m(rhs.m_full_m)
            {}

            iterator(const PointSPHEvaluator &generator, unsigned int l, unsigned int m, bool full_m):
                m_generator(generator), m_l(l), m_m(m), m_full_m(full_m)
            {}

            inline bool operator==(const iterator &rhs) const
            {
                return rhs.m_l == m_l && rhs.m_m == m_m;
            }

            inline bool operator!=(const iterator &rhs) const
            {
                return !(*this == rhs);
            }

            inline iterator& operator++()
            {
                fsph::internal::iterator_increment<Real, std::complex<Real> >(m_full_m, m_l, m_m);

                return *this;
            }

            inline std::complex<Real> operator*() const
            {
                return fsph::internal::iterator_get<Real, std::complex<Real> >(m_l, m_m, m_generator.m_legendre.get(), m_generator.m_thetaHarmonics.get());
            }

            inline std::complex<Real> grad_phi(Real phi, Real theta) const
            {
                return fsph::internal::iterator_get_grad_phi<Real, std::complex<Real> >(m_l, m_m, phi, theta, m_generator.m_legendre.get(), m_generator.m_thetaHarmonics.get());
            }

            inline std::complex<Real> grad_theta() const
            {
                return fsph::internal::iterator_get_grad_theta<Real, std::complex<Real> >(m_l, m_m, m_generator.m_legendre.get(), m_generator.m_thetaHarmonics.get());
            }

        private:
            const PointSPHEvaluator &m_generator;
            unsigned int m_l;
            unsigned int m_m;
            bool m_full_m;
        };

        PointSPHEvaluator(unsigned int lmax):
            m_lmax(lmax), m_sinPowers(new Real[lmax + 1], lmax + 1),
            m_thetaHarmonics(new std::complex<Real>[lmax + 1], lmax + 1),
            m_recurrencePrefactors(new Real[2*(lmax + 1)*lmax], 2*(lmax + 1)*lmax),
            m_jacobi(new Real[(lmax + 1)*(lmax + 1)], (lmax + 1)*(lmax + 1)),
            m_legendre(new Real[internal::sphCount(lmax)], internal::sphCount(lmax))
        {
            evaluatePrefactors();
        }

        iterator begin(bool full_m) const
        {
            return iterator(*this, full_m);
        }

        iterator end() const
        {
            return iterator(*this, m_lmax + 1, 0, 0);
        }

        iterator begin_l(unsigned int l, unsigned int m, bool full_m) const
        {
            return iterator(*this, l, m, full_m);
        }

        // phi in [0, pi]; theta in [0, 2*pi]
        void compute(Real phi, Real theta)
        {
            const Real sphi(sin(phi));
            compute_sinpows(sphi);

            compute_thetaHarmonics(theta);

            const Real cphi(cos(phi));
            compute_jacobis(cphi);

            compute_legendres();
        }
    private:
        const unsigned int m_lmax;
        // powers of sin(phi)
        SharedArray<Real> m_sinPowers;
        // harmonics of theta (e^(1j*m*theta))
        SharedArray<std::complex<Real> > m_thetaHarmonics;
        // prefactors for the Jacobi recurrence relation
        SharedArray<Real> m_recurrencePrefactors;
        // Jacobi polynomials
        SharedArray<Real> m_jacobi;
        // Associated Legendre polynomials
        SharedArray<Real> m_legendre;

        void evaluatePrefactors()
        {
            fsph::internal::evaluatePrefactors<Real, std::complex<Real> >(m_lmax, m_recurrencePrefactors.get());
        }

        void compute_sinpows(const Real &sphi)
        {
            fsph::internal::compute_sinpows<Real, std::complex<Real> >(sphi, m_lmax, m_sinPowers.get());
        }

        void compute_thetaHarmonics(const Real &theta)
        {
            fsph::internal::compute_thetaHarmonics<Real, std::complex<Real> >(theta, m_lmax, m_thetaHarmonics.get());
        }

        void compute_jacobis(const Real &cphi)
        {
            fsph::internal::compute_jacobis<Real, std::complex<Real> >(cphi, m_lmax, m_recurrencePrefactors.get(), m_jacobi.get());
        }

        void compute_legendres()
        {
            fsph::internal::compute_legendres<Real, std::complex<Real> >(m_lmax, m_sinPowers.get(), m_jacobi.get(), m_legendre.get());
        }
    };

    template<typename Real>
    void evaluate_SPH(std::complex<Real> *target, unsigned int lmax, const Real *phi, const Real *theta, unsigned int N, bool full_m)
    {
        PointSPHEvaluator<Real> eval(lmax);

        unsigned int j(0);
        for(unsigned int i(0); i < N; ++i)
        {
            eval.compute(phi[i], theta[i]);

            for(typename PointSPHEvaluator<Real>::iterator iter(eval.begin(full_m));
                iter != eval.end(); ++iter)
            {
                target[j] = *iter;
                ++j;
            }
        }
    }
}

#endif

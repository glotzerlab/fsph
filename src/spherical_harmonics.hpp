#include <iostream>
#include <complex>
#include <math.h>
#include <vector>
#include "SharedArray.hpp"
#include "internal.hpp"

#ifndef __FSPH_SPHERICAL_HARMONICS__
#define __FSPH_SPHERICAL_HARMONICS__

namespace fsph{

    template<typename T>
    inline T index2d(const T &w, const T &i, const int &j)
    {
        return w*i + j;
    }

    template<typename T>
    inline T sphCount(const T &lmax)
    {
        return (lmax + 1)*(lmax + 2)/2;
    }

    template<typename T>
    inline T sphIndex(const T &l, const T &m)
    {
        if(l > 0)
            return sphCount(l - 1) + m;
        else
            return 0;
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
            {
                setup(0, 0);
            }

            iterator(const iterator &rhs):
                m_generator(rhs.m_generator), m_l(0), m_m(0), m_full_m(rhs.m_full_m)
            {
                setup(rhs.m_l, rhs.m_m);
            }

            iterator(const PointSPHEvaluator &generator, unsigned int l, unsigned int m, bool full_m):
                m_generator(generator), m_l(0), m_m(0), m_full_m(full_m)
            {
                setup(l, m);
            }

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
                unsigned int mCount;
                if(m_full_m)
                    mCount = 2*m_l + 1;
                else
                    mCount = m_l + 1;

                ++m_m;
                const unsigned int deltaL(m_m/mCount);
                m_m %= mCount;
                m_l += deltaL;

                if(deltaL && m_l <= m_generator.m_lmax)
                {
                    for(unsigned int m(0); m <= m_l; ++m)
                    {
                        const std::complex<Real> Ylm(
                            m_evaluators[m].get(m_generator.m_cphi, m_generator.m_sphi, m_generator.m_theta, m_l));
                        m_cached_Ylm[m] = Ylm;

                        if(m_full_m)
                            m_cached_Ylm[m + m_l] = m_evaluators[m].get_negative_m(
                                Ylm, m_generator.m_theta);
                    }

                }

                return *this;
            }

            inline std::complex<Real> operator*() const
            {
                return m_cached_Ylm[m_m];
            }

            inline std::complex<Real> grad_phi(Real phi, Real theta) const
            {
                std::complex<Real> Ylmp1(0, 0);
                unsigned int abs_m(m_m);
                if(m_m < m_l)
                    Ylmp1 = m_cached_Ylm[m_m + 1];
                else if(m_m > m_l)
                {
                    abs_m -= m_l;

                    if(abs_m == 1)
                        Ylmp1 = m_cached_Ylm[0];
                    else
                        Ylmp1 = m_cached_Ylm[m_m - 1];
                }

                return m_evaluators[abs_m].grad_phi(
                    m_generator.m_cphi, m_generator.m_sphi, m_generator.m_theta,
                    m_l, this->operator*(), Ylmp1, m_m > m_l);
            }

            inline std::complex<Real> grad_theta() const
            {
                const unsigned int abs_m(m_m > m_l? m_m - m_l: m_m);
                return m_evaluators[abs_m].grad_theta(this->operator*(), m_m > m_l);
            }

        private:
            const PointSPHEvaluator &m_generator;
            unsigned int m_l;
            unsigned int m_m;
            bool m_full_m;
            // per-m (with m >= 0) spherical harmonic series
            // evaluators (each produces for a series of l with a
            // constant m)
            std::vector<internal::SPHSeriesEvaluator<Real, std::complex<Real> > > m_evaluators;
            // per-m spherical harmonics, cached for producing the m<0 values
            std::vector<std::complex<Real> > m_cached_Ylm;

            void setup(unsigned int l, unsigned int m)
            {
                // don't allocate anything when this is just being
                // used as an end iterator
                if(l > m_generator.m_lmax)
                {
                    m_l = l;
                    m_m = m;
                    return;
                }

                for(unsigned int m(0); m < m_generator.m_lmax + 1; ++m)
                {
                    m_evaluators.push_back(internal::SPHSeriesEvaluator<Real, std::complex<Real> >(m));
                }

                // for simplicity, resize to a definitely safe size
                m_cached_Ylm.resize(2*m_generator.m_lmax + 1);

                m_cached_Ylm[0] = 0.5/sqrt(M_PI);

                while(m_l < l && m_m < m)
                    this->operator++();
            }
        };

        PointSPHEvaluator(unsigned int lmax):
            m_lmax(lmax), m_cphi(1), m_sphi(0), m_theta(0)
        {
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
            const Real cphi(cos(phi));
            const Real sphi(sin(phi));
            m_cphi = cphi;
            m_sphi = sphi;
            m_theta = theta;
        }
    private:
        const unsigned int m_lmax;
        // last computed cos(phi)
        Real m_cphi;
        // last computed sin(phi)
        Real m_sphi;
        // last computed theta
        Real m_theta;
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

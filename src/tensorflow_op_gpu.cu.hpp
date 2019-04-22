#include <complex>

namespace fsph{
    template<typename Real>
    void SphericalHarmonicSeriesKernelLauncher(
        const Real *phitheta, const unsigned int N,
        const unsigned int lmax, const bool full_m,
        std::complex<Real> *output);

    template<typename Real>
    void SphericalHarmonicSeriesGradKernelLauncher(
        const Real *phitheta, const unsigned int N,
        const unsigned int lmax, const bool full_m,
        std::complex<Real> *output);
}

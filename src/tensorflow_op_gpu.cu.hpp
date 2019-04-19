
namespace fsph{
    template<typename Real, typename Complex>
    void SphericalHarmonicSeriesKernelLauncher(
        const Real *phitheta, const unsigned int N,
        const unsigned int lmax, const bool full_m,
        Complex *output);
}

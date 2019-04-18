
namespace fsph{
    template<typename Real, typename Complex>
    void SphericalHarmonicSeriesKernelLauncher(
        const Real *phitheta, const unsigned int N, const unsigned int points_per_block,
        const unsigned int lmax, const bool full_m, const unsigned int shm_size,
        Complex *output);
}

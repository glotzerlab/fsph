
#include <boost/math/special_functions/spherical_harmonic.hpp>
#include <iostream>

#include "../src/spherical_harmonics.hpp"

using namespace fsph;
using namespace boost::math;

int main(int argc, char **argv)
{
    unsigned int lmax(5);
    float phi(0.25), theta(1.5*M_PI);

    PointSPHEvaluator<float> eval(lmax);
    eval.compute(phi, theta);
    PointSPHEvaluator<float>::iterator iter(eval.begin(true));

    double error(0);
    unsigned int N(0);

    for(unsigned int l(0); l <= lmax; ++l)
    {
        for(unsigned int m(0); m <= l; ++m)
        {
            std::complex<float> from_fsph(*iter);
            // boost names phi and theta using the opposite convention
            std::complex<float> from_boost(spherical_harmonic(l, m, phi, theta));
            from_boost *= pow(-1, m);
            error += abs(from_fsph - from_boost);
            ++N;
            ++iter;
        }
        for(unsigned int m(1); m <= l; ++m)
        {
            std::complex<float> from_fsph(*iter);
            // boost names phi and theta using the opposite convention
            std::complex<float> from_boost(spherical_harmonic(l, -(int)m, phi, theta));
            error += abs(from_fsph - from_boost);
            ++N;
            ++iter;
        }
    }

    return error/N > 1e-7;
}

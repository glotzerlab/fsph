
#include "spherical_harmonics.hpp"
#include "tensorflow_op_gpu.cu.hpp"

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

template<typename Real, typename Complex>
class SphericalHarmonicSeriesGPUOp : public OpKernel {
public:
    explicit SphericalHarmonicSeriesGPUOp(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* context) override {
        const Tensor& input_tensor = context->input(0);
        auto input_flat = input_tensor.flat<Real>();

        // Input coordinates consiste of alternating phi (0..pi) and
        // theta (0..2pi) values
        const unsigned int N(input_flat.size()/2);

        const Tensor& lmax_tensor = context->input(1);
        const int lmax = lmax_tensor.flat<int>()(0);

        const Tensor& negative_m_tensor = context->input(2);
        const bool negative_m = negative_m_tensor.flat<bool>()(0);

        int num_sphs(fsph::sphCount(lmax));
        if(negative_m && lmax >= 1)
            num_sphs += fsph::sphCount(lmax - 1);

        TensorShape outputShape(input_tensor.shape());
        outputShape.RemoveLastDims(1);
        outputShape.AddDim(num_sphs);

        // Create an output tensor
        Tensor* output_tensor = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, outputShape,
                                                         &output_tensor));
        auto output_flat = output_tensor->flat<Complex>();

        const unsigned int points_per_block(1);
        const unsigned int base_mem_elements(2*(lmax + 1)*lmax); // recurrence prefactors
        const unsigned int base_mem_bytes(sizeof(Real)*base_mem_elements);
        const unsigned int mem_elements_per_point(
            lmax + 1 + // sin powers
            2*(lmax + 1) + // theta harmonics (*2 to hold complex)
            (lmax + 1)*(lmax + 1) + // jacobi
            internal::sphCount(lmax) // legendre
            );
        const unsigned int mem_bytes_per_point(sizeof(Real)*mem_elements_per_point);
        const unsigned int mem_bytes(base_mem_bytes + points_per_block*mem_bytes_per_point);

        SphericalHarmonicSeriesLauncher<Real, Complex>(
            input_flat.data(), N, points_per_block, lmax, negative_m, mem_bytes,
            output_flat.data());
    }
};

REGISTER_KERNEL_BUILDER(
    Name("SphericalHarmonicSeries")
    .Device(DEVICE_GPU),
    SphericalHarmonicSeriesGPUOp<float, complex<float> >);

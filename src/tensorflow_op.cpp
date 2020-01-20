
#include "spherical_harmonics.hpp"

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

template<typename Real>
class SphericalHarmonicSeriesOp : public OpKernel {
public:
    explicit SphericalHarmonicSeriesOp(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* context) override {
        const Tensor& input_tensor = context->input(0);
        auto input_flat = input_tensor.flat<Real>();

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
        auto output_flat = output_tensor->flat<std::complex<Real> >();

        // Input coordinates consiste of alternating phi (0..pi) and
        // theta (0..2pi) values
        const unsigned int N(input_flat.size()/2);

        fsph::PointSPHEvaluator<Real> eval(lmax);

        for (unsigned int i(0); i < N; i++) {
            const Real phi(input_flat(2*i));
            const Real theta(input_flat(2*i + 1));

            eval.compute(phi, theta);

            int offset(i*num_sphs);
            for(typename fsph::PointSPHEvaluator<Real>::iterator iter(eval.begin(negative_m));
                iter != eval.end(); ++iter)
                output_flat(offset++) = *iter;
        }

    }
};

REGISTER_OP("SphericalHarmonicSeries")
.Input("coords: float")
.Input("lmax: int32")
.Input("negative_m: bool")
.Output("sphs: complex64")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    ::tensorflow::shape_inference::ShapeHandle coords_input;
    TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(0), 1, &coords_input));

    ::tensorflow::shape_inference::DimensionHandle last_dim_handle;
    TF_RETURN_IF_ERROR(c->WithValue(c->Dim(coords_input, -1), 2, &last_dim_handle));

    ::tensorflow::shape_inference::ShapeHandle lmax_input;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &lmax_input));

    ::tensorflow::shape_inference::ShapeHandle negative_m_input;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &negative_m_input));

    ::tensorflow::shape_inference::ShapeHandle output_handle;
    TF_RETURN_IF_ERROR(c->ReplaceDim(coords_input, -1, c->UnknownDim(), &output_handle));
    c->set_output(0, output_handle);

    return Status::OK();
});

REGISTER_KERNEL_BUILDER(
    Name("SphericalHarmonicSeries")
    .Device(DEVICE_CPU),
    SphericalHarmonicSeriesOp<float>);

template<typename Real>
class SphericalHarmonicSeriesGradOp : public OpKernel {
public:
    explicit SphericalHarmonicSeriesGradOp(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* context) override {
        const Tensor& input_tensor = context->input(0);
        auto input_flat = input_tensor.flat<Real>();

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
        outputShape.AddDim(2);

        // Create an output tensor
        Tensor* output_tensor = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, outputShape,
                                                         &output_tensor));
        auto output_flat = output_tensor->flat<std::complex<Real> >();

        // Input coordinates consiste of alternating phi (0..pi) and
        // theta (0..2pi) values
        const unsigned int N(input_flat.size()/2);

        fsph::PointSPHEvaluator<Real> eval(lmax);

        for (unsigned int i(0); i < N; i++) {
            const Real phi(input_flat(2*i));
            const Real theta(input_flat(2*i + 1));

            eval.compute(phi, theta);

            int offset(i*num_sphs*2);
            for(typename fsph::PointSPHEvaluator<Real>::iterator iter(eval.begin(negative_m));
                iter != eval.end(); ++iter)
            {
                output_flat(offset++) = iter.grad_phi();
                output_flat(offset++) = iter.grad_theta();
            }
        }

    }
};

REGISTER_OP("SphericalHarmonicSeriesGrad")
.Input("coords: float")
.Input("lmax: int32")
.Input("negative_m: bool")
.Output("sphs: complex64")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    ::tensorflow::shape_inference::ShapeHandle coords_input;
    TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(0), 1, &coords_input));

    ::tensorflow::shape_inference::DimensionHandle last_dim_handle;
    TF_RETURN_IF_ERROR(c->WithValue(c->Dim(coords_input, -1), 2, &last_dim_handle));

    ::tensorflow::shape_inference::ShapeHandle lmax_input;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &lmax_input));

    ::tensorflow::shape_inference::ShapeHandle negative_m_input;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &negative_m_input));

    ::tensorflow::shape_inference::ShapeHandle output_handle;
    ::tensorflow::shape_inference::ShapeHandle extra_dim = c->MakeShape({c->MakeDim(2)});
    TF_RETURN_IF_ERROR(c->ReplaceDim(coords_input, -1, c->UnknownDim(), &output_handle));
    TF_RETURN_IF_ERROR(c->Concatenate(output_handle, extra_dim, &output_handle));
    c->set_output(0, output_handle);

    return Status::OK();
});

REGISTER_KERNEL_BUILDER(
    Name("SphericalHarmonicSeriesGrad")
    .Device(DEVICE_CPU),
    SphericalHarmonicSeriesGradOp<float>);

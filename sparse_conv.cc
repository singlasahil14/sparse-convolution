#define EIGEN_USE_THREADS

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_slice.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/util/padding.h"
#include "tensorflow/core/util/tensor_format.h"
#include "tensorflow/core/util/use_cudnn.h"
#include "tensorflow/core/kernels/bounds_check.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace tensorflow {

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

REGISTER_OP("SparseConv")
  .Input("input: T")
  .Input("sp_values: T")
  .Input("sp_indices: Tindices")
  .Input("sp_shape: int64")
  .Input("filter_shape: int32")
  .Output("output: T")
  .Attr("T: {float, double}")
  .Attr("Tindices: {int32,int64} = DT_INT64")
  .Attr("strides: list(int)=[1,1,1,1]")
  .Attr("use_cudnn_on_gpu: bool = true")
  .Attr(GetPaddingAttrString()+" = 'SAME'")
  .Attr(GetConvnetDataFormatAttrString())
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    tensorflow::shape_inference::ShapeHandle input_shape;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &input_shape));

    tensorflow::shape_inference::ShapeHandle unused;
    tensorflow::shape_inference::ShapeHandle sp_shape;
    tensorflow::shape_inference::ShapeHandle filter_shape;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &unused));
    TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 2, &unused));
    TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensor(3, &sp_shape));
    TF_RETURN_IF_ERROR(c->WithRank(sp_shape, 2, &sp_shape));
    TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensor(4, &filter_shape));
    TF_RETURN_IF_ERROR(c->WithRank(filter_shape, 4, &filter_shape));

    string data_format;
    Status s = c->GetAttr("data_format", &data_format);

    std::vector<int32> strides;
    TF_RETURN_IF_ERROR(c->GetAttr("strides", &strides));

    if (strides.size() != 4) {
      return errors::InvalidArgument(
          "Conv2D requires the stride attribute to contain 4 values, but got: ",
          strides.size());
    }

    int32 stride_rows, stride_cols;

    if (s.ok() && data_format == "NCHW") {
      // Convert input shape to default NHWC for inference
      auto dim = [&](char dimension) {
        return c->Dim(input_shape, GetTensorDimIndex<2>(FORMAT_NCHW, dimension));
      };
      input_shape = c->MakeShape({{dim('N'), dim('0'), dim('1'), dim('C')}});
      stride_rows = strides[2];
      stride_cols = strides[3];
    } else {
      stride_rows = strides[1];
      stride_cols = strides[2];
    }

    tensorflow::shape_inference::DimensionHandle unused_dim;
    tensorflow::shape_inference::DimensionHandle batch_size_dim = c->Dim(input_shape, 0);
    tensorflow::shape_inference::DimensionHandle in_rows_dim = c->Dim(input_shape, 1);
    tensorflow::shape_inference::DimensionHandle in_cols_dim = c->Dim(input_shape, 2);
    tensorflow::shape_inference::DimensionHandle filter_rows_dim = c->Dim(filter_shape, 0);
    tensorflow::shape_inference::DimensionHandle filter_cols_dim = c->Dim(filter_shape, 1);
    tensorflow::shape_inference::DimensionHandle input_depth_dim = c->Dim(filter_shape, 2);
    tensorflow::shape_inference::DimensionHandle output_depth_dim = c->Dim(filter_shape, 3);
    tensorflow::shape_inference::DimensionHandle filter_input_dim;

    TF_RETURN_IF_ERROR(c->Multiply(filter_rows_dim, filter_cols_dim, &filter_input_dim));
    TF_RETURN_IF_ERROR(c->Multiply(filter_input_dim, input_depth_dim, &filter_input_dim));
    TF_RETURN_IF_ERROR(c->Merge(c->Dim(input_shape, 3), c->Dim(filter_shape, 2), &unused_dim));
    TF_RETURN_IF_ERROR(c->Merge(filter_input_dim, c->Dim(sp_shape, 0), &unused_dim));

    Padding padding;
    TF_RETURN_IF_ERROR(c->GetAttr("padding", &padding));

    tensorflow::shape_inference::DimensionHandle output_rows, output_cols;
    TF_RETURN_IF_ERROR(GetWindowedOutputSizeFromDims(
        c, in_rows_dim, filter_rows_dim, stride_rows, padding, &output_rows));
    TF_RETURN_IF_ERROR(GetWindowedOutputSizeFromDims(
        c, in_cols_dim, filter_cols_dim, stride_cols, padding, &output_cols));

    tensorflow::shape_inference::ShapeHandle output_shape;
    if (data_format == "NCHW") {
      output_shape = c->MakeShape(
          {batch_size_dim, output_depth_dim, output_rows, output_cols});
    } else {
      output_shape = c->MakeShape(
          {batch_size_dim, output_rows, output_cols, output_depth_dim});
    }

    c->set_output(0, output_shape);
    return Status::OK();
  });

template <typename T, typename Tindices>
class SparseConvOp : public OpKernel {
  public:
    explicit SparseConvOp(OpKernelConstruction* context) : OpKernel(context) {
      OP_REQUIRES_OK(context, context->GetAttr("strides", &strides_));
      string data_format;
      OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format));
      OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
                  errors::InvalidArgument("Invalid data format"));
      OP_REQUIRES_OK(context, context->GetAttr("use_cudnn_on_gpu", &use_cudnn_));
      use_cudnn_ &= CanUseCudnn();
      cudnn_use_autotune_ = CudnnUseAutotune();
      OP_REQUIRES(context, strides_.size() == 4,
                  errors::InvalidArgument("Sliding window strides field must "
                                          "specify 4 dimensions"));
      const int64 stride_n = GetTensorDim(strides_, data_format_, 'N');
      const int64 stride_c = GetTensorDim(strides_, data_format_, 'C');
      OP_REQUIRES(
          context, stride_n == 1 && stride_c == 1,
          errors::InvalidArgument("Current implementation does not yet support "
                                  "strides in the batch and depth dimensions."));
      OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);

    // Input filter is of the following dimensions:
    // [ filter_rows, filter_cols, in_depth, out_depth]
    const Tensor& filter_shape = context->input(4);

    // For 2D convolution, there should be 4 dimensions.
    OP_REQUIRES(context, input.dims() == 4,
                errors::InvalidArgument("input must be 4-dimensional",
                                        input.shape().DebugString()));
    OP_REQUIRES(context, filter_shape.NumElements() == 4,
                errors::InvalidArgument("filter must be 4-dimensional: ",
                                        filter_shape.DebugString()));

    auto filter_shape_flat = filter_shape.flat<int32>();
    for (int i = 0; i < 3; i++) {
      OP_REQUIRES(context,
                  FastBoundsCheck(filter_shape_flat(i), std::numeric_limits<int>::max()),
                  errors::InvalidArgument("filter too large"));
    }

    // The last dimension for input is in_depth. It must be the same as the
    // filter's in_depth.
    const int64 in_depth = GetTensorDim(input, data_format_, 'C');
    OP_REQUIRES(context, in_depth == filter_shape_flat(2),
                errors::InvalidArgument(
                    "input and filter must have the same depth: ", in_depth,
                    " vs ", filter_shape_flat(2)));

    // The last dimension for filter is out_depth.
    const int out_depth = static_cast<int>(filter_shape_flat(3));

    // The second dimension for input is rows/height.
    // The first dimension for filter is rows/height.
    const int64 input_rows_raw = GetTensorDim(input, data_format_, 'H');
    OP_REQUIRES(
        context,
        FastBoundsCheck(input_rows_raw, std::numeric_limits<int>::max()),
        errors::InvalidArgument("Input rows too large"));
    const int input_rows = static_cast<int>(input_rows_raw);
    const int filter_rows = static_cast<int>(filter_shape_flat(0));

    // The third dimension for input is columns/width.
    // The second dimension for filter is columns/width.
    const int64 input_cols_raw = GetTensorDim(input, data_format_, 'W');
    OP_REQUIRES(
        context,
        FastBoundsCheck(input_cols_raw, std::numeric_limits<int>::max()),
        errors::InvalidArgument("Input cols too large"));
    const int input_cols = static_cast<int>(input_cols_raw);
    const int filter_cols = static_cast<int>(filter_shape_flat(1));

    // The first dimension for input is batch.
    const int64 batch_raw = GetTensorDim(input, data_format_, 'N');
    OP_REQUIRES(context,
                FastBoundsCheck(batch_raw, std::numeric_limits<int>::max()),
                errors::InvalidArgument("batch is too large"));
    const int batch = static_cast<int>(batch_raw);

    // For now we take the stride from the second and third dimensions only (we
    // do not support striding on the batch or depth dimension).
    const int stride_rows = GetTensorDim(strides_, data_format_, 'H');
    const int stride_cols = GetTensorDim(strides_, data_format_, 'W');

    int64 out_rows_raw = 0, out_cols_raw = 0, pad_rows_raw = 0, pad_cols_raw = 0;
    OP_REQUIRES_OK(context,
                   GetWindowedOutputSize(input_rows, filter_rows, stride_rows,
                                         padding_, &out_rows_raw, &pad_rows_raw));
    OP_REQUIRES_OK(context,
                   GetWindowedOutputSize(input_cols, filter_cols, stride_cols,
                                         padding_, &out_cols_raw, &pad_cols_raw));

    const int64 out_rows = static_cast<int64>(out_rows_raw);
    const int64 out_cols = static_cast<int64>(out_cols_raw);
    const int64 pad_rows = static_cast<int64>(pad_rows_raw);
    const int64 pad_cols = static_cast<int64>(pad_cols_raw);

    const Tensor* sp_indices;
    const Tensor* sp_values;
    const Tensor* sp_shape;
    OP_REQUIRES_OK(context, context->input("sp_indices", &sp_indices));
    OP_REQUIRES_OK(context, context->input("sp_values", &sp_values));
    OP_REQUIRES_OK(context, context->input("sp_shape", &sp_shape));

    OP_REQUIRES(context, TensorShapeUtils::IsVector(sp_shape->shape()),
                errors::InvalidArgument("Tensor 'sp_shape' is not a vector"));
    OP_REQUIRES(context, sp_shape->NumElements() == 2,
                errors::InvalidArgument("Tensor 'sp_shape' must have 2 elements"));

    OP_REQUIRES(context, TensorShapeUtils::IsVector(sp_values->shape()),
                errors::InvalidArgument("Tensor 'sp_values' is not a vector"));
    OP_REQUIRES(context, TensorShapeUtils::IsMatrix(sp_indices->shape()),
                errors::InvalidArgument("Tensor 'sp_indices' is not a matrix"));

    const int64 nnz = sp_indices->shape().dim_size(0);
    OP_REQUIRES(context, nnz == sp_values->NumElements(),
                errors::InvalidArgument("Number of rows of sp_indices does not "
                                        "match number of entries in sp_values"));

    OP_REQUIRES(context, sp_indices->shape().dim_size(1) == sp_shape->NumElements(),
                errors::InvalidArgument("Number of columns of sp_indices does not match "
                                        "number of entries in sp_shape"));

    auto sp_indices_mat = sp_indices->matrix<int64>();
    auto sp_values_flat = sp_values->flat<T>();

    const int padding_rows =
        std::max<int>(0, (out_rows - 1) * stride_rows + filter_rows - input_rows);
    const int padding_cols =
        std::max<int>(0, (out_cols - 1) * stride_cols + filter_cols - input_cols);

    const int pad_top = padding_rows/2;
    const int pad_bottom = padding_rows - pad_top;

    const int pad_left = padding_cols/2;
    const int pad_right = padding_cols - pad_left;

    const std::array<std::pair<int, int>, 4> paddings{{std::make_pair(0, 0), std::make_pair(pad_top, pad_bottom), std::make_pair(pad_left, pad_right), std::make_pair(0, 0)}};

    auto input_tensor = input.tensor<T, 4>();
    TensorShape out_shape = ShapeFromFormat(data_format_, batch, out_rows, out_cols, out_depth);

    const Eigen::Tensor<T, 4, Eigen::RowMajor> transformed_input = input_tensor.pad(paddings);
    const int64 transformed_input_rows = transformed_input.dimension(1);
    const int64 transformed_input_cols = transformed_input.dimension(2);

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &output));
    auto output_flat = output->flat<T>();

    auto input_matrix = Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor >> (input.flat<T>().data(), batch, input_rows*input_cols*in_depth);
    auto transformed_input_matrix = Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor >> (transformed_input.data(), batch, transformed_input_rows*transformed_input_cols*in_depth);
    auto output_matrix = Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor >> (output->flat<T>().data(), batch, out_rows*out_cols*out_depth); 

    for(int j=0; j<out_rows; j++)
    {
      for(int k=0; k<out_cols; k++)
      {
        for(int l=0; l<nnz; l++)
        {
          long long int filter_idx = sp_indices_mat(l, 0);
          long long int out_channel = sp_indices_mat(l, 1);
          long long int o = filter_idx%in_depth;
          long long int n = ((filter_idx-o)/in_depth) % filter_cols;
          long long int m = filter_idx/(filter_cols*in_depth);
          long long int out_col_idx = ((j*out_cols) + k)*out_depth + out_channel;
          long long int pos_j_tfmed = j*stride_rows + m;
          long long int pos_k_tfmed = k*stride_cols + n;
          long long int in_col_tfmed_idx = ((pos_j_tfmed*transformed_input_cols) + pos_k_tfmed)*in_depth + o;

          output_matrix.col(out_col_idx) += transformed_input_matrix.col(in_col_tfmed_idx)*sp_values_flat(l);
        }
      }
    }
}
  private:
  std::vector<int32> strides_;
  bool use_cudnn_;
  Padding padding_;
  TensorFormat data_format_;
  bool cudnn_use_autotune_;
};

#define REGISTER_CPU(TypeT, TypeIndex)           \
  REGISTER_KERNEL_BUILDER(                       \
      Name("SparseConv")                         \
          .Device(DEVICE_CPU)                    \
          .TypeConstraint<TypeT>("T")            \
          .TypeConstraint<TypeIndex>("Tindices") \
          .HostMemory("sp_shape"),               \
      SparseConvOp<TypeT, TypeIndex>);

#define REGISTER_KERNELS_CPU(T) \
  REGISTER_CPU(T, int64);       \
  REGISTER_CPU(T, int32)

REGISTER_KERNELS_CPU(float);
REGISTER_KERNELS_CPU(double);

}
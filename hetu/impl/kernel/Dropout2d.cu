#include "hetu/core/ndarray.h"
#include "hetu/impl/cuda/CUDARand.h"
#include "hetu/impl/stream/CUDAStream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/cuda_utils.h"
#include "hetu/impl/utils/offset_calculator.cuh"

namespace hetu {
namespace impl {

template <typename spec_t, typename mask_t>
__global__ void dropout2d_kernel(const spec_t* input, spec_t* output, mask_t* mask,
                                 float drop_rate, size_t size,
                                 size_t last_two,
                                 const OffsetCalculator* in_offset_calculator,
                                 const OffsetCalculator* out_offset_calculator,
                                 const OffsetCalculator* mask_offset_calculator) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;
  size_t leader = size_t(idx / last_two) * last_two;
  auto out_offset = out_offset_calculator->get(leader);
  mask_t keep_mask = (mask_t) (output[out_offset] >= drop_rate);
  auto in_offset = in_offset_calculator->get(idx);
  out_offset = out_offset_calculator->get(idx);
  auto mask_offset = mask_offset_calculator->get(idx);
  output[out_offset] = input[in_offset] * keep_mask / (1 - drop_rate);
  mask[mask_offset] = keep_mask;
}

template <typename spec_t, typename mask_t>
__global__ void dropout2d_gradient_kernel(const spec_t* grad,
                                          const mask_t* fw_mask, spec_t* output,
                                          float drop_rate, size_t size,
                                          const OffsetCalculator* grad_offset_calculator,
                                          const OffsetCalculator* fw_mask_offset_calculator,
                                          const OffsetCalculator* out_offset_calculator) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;
  auto mask_offset = fw_mask_offset_calculator->get(idx);
  auto grad_offset = grad_offset_calculator->get(idx);
  auto out_offset = out_offset_calculator->get(idx);
  output[out_offset] = grad[grad_offset] * fw_mask[mask_offset]  / (1 - drop_rate);
}

void Dropout2dCuda(const NDArray& input, double drop_rate, uint64_t seed,
                   NDArray& output, NDArray& mask, const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output);
  HT_ASSERT_SAME_DEVICE(input, mask);
  HT_ASSERT_SAME_SHAPE(input, output);
  HT_ASSERT_SAME_SHAPE(input, mask);
  size_t size = input->numel();
  if (size == 0)
    return;
  HT_ASSERT(input->ndim() == 4);
  size_t last_two_size = 1;
  last_two_size *= input->shape(input->ndim() - 1);
  last_two_size *= input->shape(input->ndim() - 2);

  dim3 blocks, threads;
  threads.x = MIN(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  blocks.x = DIVUP(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  CUDAStream cuda_stream(stream);
  hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
  NDArray in_offset_calculator_arr, out_offset_calculator_arr,
          mask_offset_calculator_arr;
  OffsetCalculator *in_offset_calculator, *out_offset_calculator,
                   *mask_offset_calculator;
  std::tie(in_offset_calculator_arr, in_offset_calculator) =
    AllocOffsetCalculator(input, stream);
  std::tie(out_offset_calculator_arr, out_offset_calculator) = 
    AllocOffsetCalculator(output, stream);
  std::tie(mask_offset_calculator_arr, mask_offset_calculator) =
    AllocOffsetCalculator(mask, stream);
  curandGenerator_t gen;
  CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_PHILOX4_32_10));
  CURAND_CALL(curandSetStream(gen, cuda_stream));
  CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, seed));
  HT_DISPATCH_FLOATING_TYPES(input->dtype(), spec_t, "Dropout2dCuda", [&]() {
    curand_gen_uniform<spec_t>(gen, output->data_ptr<spec_t>(), size);
    dropout2d_kernel<spec_t><<<blocks, threads, 0, cuda_stream>>>(
      input->data_ptr<spec_t>(), output->data_ptr<spec_t>(),
      mask->data_ptr<bool>(), static_cast<float>(drop_rate), size,
      last_two_size, in_offset_calculator,
      out_offset_calculator, mask_offset_calculator);
  });
  CURAND_CALL(curandDestroyGenerator(gen));
  NDArray::MarkUsedBy({input, output, mask, in_offset_calculator_arr,
                      out_offset_calculator_arr, mask_offset_calculator_arr}, stream);
}

void Dropout2dGradientCuda(const NDArray& grad, const NDArray& fw_mask,
                           double drop_rate, NDArray& output,
                           const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(grad);
  HT_ASSERT_SAME_DEVICE(grad, output);
  HT_ASSERT_SAME_DEVICE(grad, fw_mask);
  HT_ASSERT_SAME_SHAPE(grad, output);
  HT_ASSERT_SAME_SHAPE(grad, fw_mask);
  size_t size = grad->numel();
  if (size == 0)
    return;

  dim3 blocks, threads;
  threads.x = MIN(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  blocks.x = DIVUP(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  CUDAStream cuda_stream(stream);
  hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
  NDArray grad_offset_calculator_arr, fw_mask_offset_calculator_arr,
          out_offset_calculator_arr;
  OffsetCalculator *grad_offset_calculator, *fw_mask_offset_calculator,
                   *out_offset_calculator;
  std::tie(grad_offset_calculator_arr, grad_offset_calculator) =
    AllocOffsetCalculator(grad, stream);
  std::tie(fw_mask_offset_calculator_arr, fw_mask_offset_calculator) =
    AllocOffsetCalculator(fw_mask, stream);
  std::tie(out_offset_calculator_arr, out_offset_calculator) = 
    AllocOffsetCalculator(output, stream);
  HT_DISPATCH_FLOATING_TYPES(grad->dtype(), spec_t, "Dropout2dGradientCuda", [&]() {
    dropout2d_gradient_kernel<spec_t><<<blocks, threads, 0, cuda_stream>>>(
      grad->data_ptr<spec_t>(), fw_mask->data_ptr<spec_t>(),
      output->data_ptr<spec_t>(), static_cast<float>(drop_rate), size,
      grad_offset_calculator, fw_mask_offset_calculator,
      out_offset_calculator);
  });
  NDArray::MarkUsedBy({grad, fw_mask, output, grad_offset_calculator_arr,
                      fw_mask_offset_calculator_arr, out_offset_calculator_arr}, stream);
}

} // namespace impl
} // namespace hetu

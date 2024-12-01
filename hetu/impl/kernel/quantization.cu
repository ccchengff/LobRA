#include "hetu/core/ndarray.h"
#include "hetu/impl/stream/CUDAStream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/cuda_utils.h"
#include "hetu/impl/utils/cuda_math.h"
#include "hetu/impl/utils/offset_calculator.cuh"
#include "hetu/impl/kernel/Vectorized.cuh"
#include "kernels.cuh"

namespace hetu {
namespace impl {

void QuantizationCuda(const NDArray& input, NDArray& absmax,
                      const NDArray& code, NDArray& output, 
                      int64_t blocksize, bool stochastic, const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output);
  HT_ASSERT_SAME_SHAPE(input, output);

  size_t size = input->numel();
  if (size == 0)
    return;
  int device_id = input->device().index();
  hetu::cuda::CUDADeviceGuard guard(device_id);
  CUDAStream cuda_stream(stream);
  int num_blocks = size / blocksize;
  num_blocks = size % blocksize == 0 ? num_blocks : num_blocks + 1;
  if (input->dtype() == kFloat32) {
      if (output->dtype() == DataType::INT8) {
        switch(blocksize) {
          case 64:
            kQuantizeBlockwise<float, 64, 2, 0, 0><<<num_blocks, 32>>>(code->data_ptr<float>(), input->data_ptr<float>(), 
                                                                       absmax->data_ptr<float>(), output->data_ptr<unsigned char>(), 
                                                                       NULL, 0, size);
            break;
          case 128:
            kQuantizeBlockwise<float, 128, 2, 0, 0><<<num_blocks, 64>>>(code->data_ptr<float>(), input->data_ptr<float>(), 
                                                                      absmax->data_ptr<float>(), output->data_ptr<unsigned char>(), 
                                                                      NULL, 0, size);
            break;
          case 256: 
            kQuantizeBlockwise<float, 256, 2, 0, 0><<<num_blocks, 128>>>(code->data_ptr<float>(), input->data_ptr<float>(), 
                                                                      absmax->data_ptr<float>(), output->data_ptr<unsigned char>(), 
                                                                      NULL, 0, size);
            break;                                                          
          case 512: 
            kQuantizeBlockwise<float, 512, 2, 0, 0><<<num_blocks, 128>>>(code->data_ptr<float>(), input->data_ptr<float>(), 
                                                                      absmax->data_ptr<float>(), output->data_ptr<unsigned char>(), 
                                                                      NULL, 0, size);
            break;                                                           
          case 1024:
            kQuantizeBlockwise<float, 1024, 4, 0, 0><<<num_blocks, 256>>>(code->data_ptr<float>(), input->data_ptr<float>(), 
                                                                           absmax->data_ptr<float>(), output->data_ptr<unsigned char>(), 
                                                                           NULL, 0, size);
            break;
          case 2048:
            kQuantizeBlockwise<float, 2048, 4, 0, 0><<<num_blocks, 512>>>(code->data_ptr<float>(), input->data_ptr<float>(), 
                                                                           absmax->data_ptr<float>(), output->data_ptr<unsigned char>(), 
                                                                           NULL, 0, size);
            break;
          case 4096:
            kQuantizeBlockwise<float, 4096, 4, 0, 0><<<num_blocks, 1024>>>(code->data_ptr<float>(), input->data_ptr<float>(), 
                                                                      absmax->data_ptr<float>(), output->data_ptr<unsigned char>(), 
                                                                      NULL, 0, size);
            break;
          default:
            HT_NOT_IMPLEMENTED << "Invalid blocksize:" << blocksize;
        }
      }
      else if (output->dtype() == DataType::FLOAT4) {
        switch(blocksize) {
          case 64:
            kQuantizeBlockwise<float, 64, 2, 0, 1><<<num_blocks, 32>>>(NULL, input->data_ptr<float>(), 
                                                                      absmax->data_ptr<float>(), output->data_ptr<unsigned char>(), 
                                                                      NULL, 0, size);
            break;
          case 128:
            kQuantizeBlockwise<float, 128, 2, 0, 1><<<num_blocks, 64>>>(NULL, input->data_ptr<float>(), 
                                                                      absmax->data_ptr<float>(), output->data_ptr<unsigned char>(), 
                                                                      NULL, 0, size);
            break;
          case 256: 
            kQuantizeBlockwise<float, 256, 2, 0, 1><<<num_blocks, 128>>>(NULL, input->data_ptr<float>(), 
                                                                      absmax->data_ptr<float>(), output->data_ptr<unsigned char>(), 
                                                                      NULL, 0, size);
            break;                                                          
          case 512: 
            kQuantizeBlockwise<float, 512, 2, 0, 1><<<num_blocks, 128>>>(NULL, input->data_ptr<float>(), 
                                                                      absmax->data_ptr<float>(), output->data_ptr<unsigned char>(), 
                                                                      NULL, 0, size);
            break;                                                           
          case 1024:
            kQuantizeBlockwise<float, 1024, 4, 0, 1><<<num_blocks, 256>>>(NULL, input->data_ptr<float>(), 
                                                                      absmax->data_ptr<float>(), output->data_ptr<unsigned char>(), 
                                                                      NULL, 0, size);
            break;
          case 2048:
            kQuantizeBlockwise<float, 2048, 4, 0, 1><<<num_blocks, 512>>>(NULL, input->data_ptr<float>(), 
                                                                      absmax->data_ptr<float>(), output->data_ptr<unsigned char>(), 
                                                                      NULL, 0, size);
            break;
          case 4096:
            kQuantizeBlockwise<float, 4096, 4, 0, 1><<<num_blocks, 1024>>>(NULL, input->data_ptr<float>(), 
                                                                      absmax->data_ptr<float>(), output->data_ptr<unsigned char>(), 
                                                                      NULL, 0, size);
            break;
          default:
            HT_NOT_IMPLEMENTED << "Invalid blocksize:" << blocksize;
        }
      }
      else if (output->dtype() == DataType::NFLOAT4) {
        switch(blocksize) {
          case 64:
            kQuantizeBlockwise<float, 64, 2, 0, 2><<<num_blocks, 32>>>(NULL, input->data_ptr<float>(), 
                                                                      absmax->data_ptr<float>(), output->data_ptr<unsigned char>(), 
                                                                      NULL, 0, size);
            break;
          case 128:
            kQuantizeBlockwise<float, 128, 2, 0, 2><<<num_blocks, 64>>>(NULL, input->data_ptr<float>(), 
                                                                      absmax->data_ptr<float>(), output->data_ptr<unsigned char>(), 
                                                                      NULL, 0, size);
            break;
          case 256: 
            kQuantizeBlockwise<float, 256, 2, 0, 2><<<num_blocks, 128>>>(NULL, input->data_ptr<float>(), 
                                                                      absmax->data_ptr<float>(), output->data_ptr<unsigned char>(), 
                                                                      NULL, 0, size);
            break;                                                          
          case 512: 
            kQuantizeBlockwise<float, 512, 2, 0, 2><<<num_blocks, 256>>>(NULL, input->data_ptr<float>(), 
                                                                      absmax->data_ptr<float>(), output->data_ptr<unsigned char>(), 
                                                                      NULL, 0, size);
            break;                                                           
          case 1024:
            kQuantizeBlockwise<float, 1024, 4, 0, 2><<<num_blocks, 256>>>(NULL, input->data_ptr<float>(), 
                                                                      absmax->data_ptr<float>(), output->data_ptr<unsigned char>(), 
                                                                      NULL, 0, size);
            break;
          case 2048:
            kQuantizeBlockwise<float, 2048, 4, 0, 2><<<num_blocks, 512>>>(NULL, input->data_ptr<float>(), 
                                                                      absmax->data_ptr<float>(), output->data_ptr<unsigned char>(), 
                                                                      NULL, 0, size);
            break;
          case 4096:
            kQuantizeBlockwise<float, 4096, 4, 0, 2><<<num_blocks, 1024>>>(NULL, input->data_ptr<float>(), 
                                                                      absmax->data_ptr<float>(), output->data_ptr<unsigned char>(), 
                                                                      NULL, 0, size);
            break;
          default:
            HT_NOT_IMPLEMENTED << "Invalid blocksize:" << blocksize;
        }
      }
      else 
        HT_NOT_IMPLEMENTED << "Not support this quantization type:" << output->dtype();
  }
  else if (input->dtype() == kFloat16) {
    if (output->dtype() == DataType::INT8) {
        switch(blocksize) {
          case 64:
            kQuantizeBlockwise<half, 64, 2, 0, 0><<<num_blocks, 32>>>(code->data_ptr<float>(), input->data_ptr<half>(), 
                                                                       absmax->data_ptr<float>(), output->data_ptr<unsigned char>(), 
                                                                       NULL, 0, size);
            break;
          case 128:
            kQuantizeBlockwise<half, 128, 2, 0, 0><<<num_blocks, 64>>>(code->data_ptr<float>(), input->data_ptr<half>(), 
                                                                      absmax->data_ptr<float>(), output->data_ptr<unsigned char>(), 
                                                                      NULL, 0, size);
            break;
          case 256: 
            kQuantizeBlockwise<half, 256, 2, 0, 0><<<num_blocks, 128>>>(code->data_ptr<float>(), input->data_ptr<half>(), 
                                                                      absmax->data_ptr<float>(), output->data_ptr<unsigned char>(), 
                                                                      NULL, 0, size);
            break;                                                          
          case 512: 
            kQuantizeBlockwise<half, 512, 2, 0, 0><<<num_blocks, 128>>>(code->data_ptr<float>(), input->data_ptr<half>(), 
                                                                      absmax->data_ptr<float>(), output->data_ptr<unsigned char>(), 
                                                                      NULL, 0, size);
            break;                                                           
          case 1024:
            kQuantizeBlockwise<half, 1024, 4, 0, 0><<<num_blocks, 256>>>(code->data_ptr<float>(), input->data_ptr<half>(), 
                                                                           absmax->data_ptr<float>(), output->data_ptr<unsigned char>(), 
                                                                           NULL, 0, size);
            break;
          case 2048:
            kQuantizeBlockwise<half, 2048, 4, 0, 0><<<num_blocks, 512>>>(code->data_ptr<float>(), input->data_ptr<half>(), 
                                                                           absmax->data_ptr<float>(), output->data_ptr<unsigned char>(), 
                                                                           NULL, 0, size);
            break;
          case 4096:
            kQuantizeBlockwise<half, 4096, 4, 0, 0><<<num_blocks, 1024>>>(code->data_ptr<float>(), input->data_ptr<half>(), 
                                                                      absmax->data_ptr<float>(), output->data_ptr<unsigned char>(), 
                                                                      NULL, 0, size);
            break;
          default:
            HT_NOT_IMPLEMENTED << "Invalid blocksize:" << blocksize;
        }
      }
      else if (output->dtype() == DataType::FLOAT4) {
        switch(blocksize) {
          case 64:
            kQuantizeBlockwise<half, 64, 2, 0, 1><<<num_blocks, 32>>>(NULL, input->data_ptr<half>(), 
                                                                      absmax->data_ptr<float>(), output->data_ptr<unsigned char>(), 
                                                                      NULL, 0, size);
            break;
          case 128:
            kQuantizeBlockwise<half, 128, 2, 0, 1><<<num_blocks, 64>>>(NULL, input->data_ptr<half>(), 
                                                                      absmax->data_ptr<float>(), output->data_ptr<unsigned char>(), 
                                                                      NULL, 0, size);
            break;
          case 256: 
            kQuantizeBlockwise<half, 256, 2, 0, 1><<<num_blocks, 128>>>(NULL, input->data_ptr<half>(), 
                                                                      absmax->data_ptr<float>(), output->data_ptr<unsigned char>(), 
                                                                      NULL, 0, size);
            break;                                                          
          case 512: 
            kQuantizeBlockwise<half, 512, 2, 0, 1><<<num_blocks, 128>>>(NULL, input->data_ptr<half>(), 
                                                                      absmax->data_ptr<float>(), output->data_ptr<unsigned char>(), 
                                                                      NULL, 0, size);
            break;                                                           
          case 1024:
            kQuantizeBlockwise<half, 1024, 4, 0, 1><<<num_blocks, 256>>>(NULL, input->data_ptr<half>(), 
                                                                      absmax->data_ptr<float>(), output->data_ptr<unsigned char>(), 
                                                                      NULL, 0, size);
            break;
          case 2048:
            kQuantizeBlockwise<half, 2048, 4, 0, 1><<<num_blocks, 512>>>(NULL, input->data_ptr<half>(), 
                                                                      absmax->data_ptr<float>(), output->data_ptr<unsigned char>(), 
                                                                      NULL, 0, size);
            break;
          case 4096:
            kQuantizeBlockwise<half, 4096, 4, 0, 1><<<num_blocks, 1024>>>(NULL, input->data_ptr<half>(), 
                                                                      absmax->data_ptr<float>(), output->data_ptr<unsigned char>(), 
                                                                      NULL, 0, size);
            break;
          default:
            HT_NOT_IMPLEMENTED << "Invalid blocksize:" << blocksize;
        }
      }
      else if (output->dtype() == DataType::NFLOAT4) {
        switch(blocksize) {
          case 64:
            kQuantizeBlockwise<half, 64, 2, 0, 2><<<num_blocks, 32>>>(NULL, input->data_ptr<half>(), 
                                                                      absmax->data_ptr<float>(), output->data_ptr<unsigned char>(), 
                                                                      NULL, 0, size);
            break;
          case 128:
            kQuantizeBlockwise<half, 128, 2, 0, 2><<<num_blocks, 64>>>(NULL, input->data_ptr<half>(), 
                                                                      absmax->data_ptr<float>(), output->data_ptr<unsigned char>(), 
                                                                      NULL, 0, size);
            break;
          case 256: 
            kQuantizeBlockwise<half, 256, 2, 0, 2><<<num_blocks, 128>>>(NULL, input->data_ptr<half>(), 
                                                                      absmax->data_ptr<float>(), output->data_ptr<unsigned char>(), 
                                                                      NULL, 0, size);
            break;                                                          
          case 512: 
            kQuantizeBlockwise<half, 512, 2, 0, 2><<<num_blocks, 256>>>(NULL, input->data_ptr<half>(), 
                                                                      absmax->data_ptr<float>(), output->data_ptr<unsigned char>(), 
                                                                      NULL, 0, size);
            break;                                                           
          case 1024:
            kQuantizeBlockwise<half, 1024, 4, 0, 2><<<num_blocks, 256>>>(NULL, input->data_ptr<half>(), 
                                                                      absmax->data_ptr<float>(), output->data_ptr<unsigned char>(), 
                                                                      NULL, 0, size);
            break;
          case 2048:
            kQuantizeBlockwise<half, 2048, 4, 0, 2><<<num_blocks, 512>>>(NULL, input->data_ptr<half>(), 
                                                                      absmax->data_ptr<float>(), output->data_ptr<unsigned char>(), 
                                                                      NULL, 0, size);
            break;
          case 4096:
            kQuantizeBlockwise<half, 4096, 4, 0, 2><<<num_blocks, 1024>>>(NULL, input->data_ptr<half>(), 
                                                                      absmax->data_ptr<float>(), output->data_ptr<unsigned char>(), 
                                                                      NULL, 0, size);
            break;
          default:
            HT_NOT_IMPLEMENTED << "Invalid blocksize:" << blocksize;
        }
      }
      else 
        HT_NOT_IMPLEMENTED << "Not support this quantization type:" << output->dtype();
  }
  else if (input->dtype() == kBFloat16) {
    if (output->dtype() == DataType::INT8) {
        switch(blocksize) {
          case 64:
            kQuantizeBlockwise<nv_bfloat16, 64, 2, 0, 0><<<num_blocks, 32>>>(code->data_ptr<float>(), input->data_ptr<nv_bfloat16>(), 
                                                                       absmax->data_ptr<float>(), output->data_ptr<unsigned char>(), 
                                                                       NULL, 0, size);
            break;
          case 128:
            kQuantizeBlockwise<nv_bfloat16, 128, 2, 0, 0><<<num_blocks, 64>>>(code->data_ptr<float>(), input->data_ptr<nv_bfloat16>(), 
                                                                      absmax->data_ptr<float>(), output->data_ptr<unsigned char>(), 
                                                                      NULL, 0, size);
            break;
          case 256: 
            kQuantizeBlockwise<nv_bfloat16, 256, 2, 0, 0><<<num_blocks, 128>>>(code->data_ptr<float>(), input->data_ptr<nv_bfloat16>(), 
                                                                      absmax->data_ptr<float>(), output->data_ptr<unsigned char>(), 
                                                                      NULL, 0, size);
            break;                                                          
          case 512: 
            kQuantizeBlockwise<nv_bfloat16, 512, 2, 0, 0><<<num_blocks, 128>>>(code->data_ptr<float>(), input->data_ptr<nv_bfloat16>(), 
                                                                      absmax->data_ptr<float>(), output->data_ptr<unsigned char>(), 
                                                                      NULL, 0, size);
            break;                                                           
          case 1024:
            kQuantizeBlockwise<nv_bfloat16, 1024, 4, 0, 0><<<num_blocks, 256>>>(code->data_ptr<float>(), input->data_ptr<nv_bfloat16>(), 
                                                                           absmax->data_ptr<float>(), output->data_ptr<unsigned char>(), 
                                                                           NULL, 0, size);
            break;
          case 2048:
            kQuantizeBlockwise<nv_bfloat16, 2048, 4, 0, 0><<<num_blocks, 512>>>(code->data_ptr<float>(), input->data_ptr<nv_bfloat16>(), 
                                                                           absmax->data_ptr<float>(), output->data_ptr<unsigned char>(), 
                                                                           NULL, 0, size);
            break;
          case 4096:
            kQuantizeBlockwise<nv_bfloat16, 4096, 4, 0, 0><<<num_blocks, 1024>>>(code->data_ptr<float>(), input->data_ptr<nv_bfloat16>(), 
                                                                      absmax->data_ptr<float>(), output->data_ptr<unsigned char>(), 
                                                                      NULL, 0, size);
            break;
          default:
            HT_NOT_IMPLEMENTED << "Invalid blocksize:" << blocksize;
        }
      }
      else if (output->dtype() == DataType::FLOAT4) {
        switch(blocksize) {
          case 64:
            kQuantizeBlockwise<nv_bfloat16, 64, 2, 0, 1><<<num_blocks, 32>>>(NULL, input->data_ptr<nv_bfloat16>(), 
                                                                      absmax->data_ptr<float>(), output->data_ptr<unsigned char>(), 
                                                                      NULL, 0, size);
            break;
          case 128:
            kQuantizeBlockwise<nv_bfloat16, 128, 2, 0, 1><<<num_blocks, 64>>>(NULL, input->data_ptr<nv_bfloat16>(), 
                                                                      absmax->data_ptr<float>(), output->data_ptr<unsigned char>(), 
                                                                      NULL, 0, size);
            break;
          case 256: 
            kQuantizeBlockwise<nv_bfloat16, 256, 2, 0, 1><<<num_blocks, 128>>>(NULL, input->data_ptr<nv_bfloat16>(), 
                                                                      absmax->data_ptr<float>(), output->data_ptr<unsigned char>(), 
                                                                      NULL, 0, size);
            break;                                                          
          case 512: 
            kQuantizeBlockwise<nv_bfloat16, 512, 2, 0, 1><<<num_blocks, 128>>>(NULL, input->data_ptr<nv_bfloat16>(), 
                                                                      absmax->data_ptr<float>(), output->data_ptr<unsigned char>(), 
                                                                      NULL, 0, size);
            break;                                                           
          case 1024:
            kQuantizeBlockwise<nv_bfloat16, 1024, 4, 0, 1><<<num_blocks, 256>>>(NULL, input->data_ptr<nv_bfloat16>(), 
                                                                      absmax->data_ptr<float>(), output->data_ptr<unsigned char>(), 
                                                                      NULL, 0, size);
            break;
          case 2048:
            kQuantizeBlockwise<nv_bfloat16, 2048, 4, 0, 1><<<num_blocks, 512>>>(NULL, input->data_ptr<nv_bfloat16>(), 
                                                                      absmax->data_ptr<float>(), output->data_ptr<unsigned char>(), 
                                                                      NULL, 0, size);
            break;
          case 4096:
            kQuantizeBlockwise<nv_bfloat16, 4096, 4, 0, 1><<<num_blocks, 1024>>>(NULL, input->data_ptr<nv_bfloat16>(), 
                                                                      absmax->data_ptr<float>(), output->data_ptr<unsigned char>(), 
                                                                      NULL, 0, size);
            break;
          default:
            HT_NOT_IMPLEMENTED << "Invalid blocksize:" << blocksize;
        }
      }
      else if (output->dtype() == DataType::NFLOAT4) {
        switch(blocksize) {
          case 64:
            kQuantizeBlockwise<nv_bfloat16, 64, 2, 0, 2><<<num_blocks, 32>>>(NULL, input->data_ptr<nv_bfloat16>(), 
                                                                      absmax->data_ptr<float>(), output->data_ptr<unsigned char>(), 
                                                                      NULL, 0, size);
            break;
          case 128:
            kQuantizeBlockwise<nv_bfloat16, 128, 2, 0, 2><<<num_blocks, 64>>>(NULL, input->data_ptr<nv_bfloat16>(), 
                                                                      absmax->data_ptr<float>(), output->data_ptr<unsigned char>(), 
                                                                      NULL, 0, size);
            break;
          case 256: 
            kQuantizeBlockwise<nv_bfloat16, 256, 2, 0, 2><<<num_blocks, 128>>>(NULL, input->data_ptr<nv_bfloat16>(), 
                                                                      absmax->data_ptr<float>(), output->data_ptr<unsigned char>(), 
                                                                      NULL, 0, size);
            break;                                                          
          case 512: 
            kQuantizeBlockwise<nv_bfloat16, 512, 2, 0, 2><<<num_blocks, 256>>>(NULL, input->data_ptr<nv_bfloat16>(), 
                                                                      absmax->data_ptr<float>(), output->data_ptr<unsigned char>(), 
                                                                      NULL, 0, size);
            break;                                                           
          case 1024:
            kQuantizeBlockwise<nv_bfloat16, 1024, 4, 0, 2><<<num_blocks, 256>>>(NULL, input->data_ptr<nv_bfloat16>(), 
                                                                      absmax->data_ptr<float>(), output->data_ptr<unsigned char>(), 
                                                                      NULL, 0, size);
            break;
          case 2048:
            kQuantizeBlockwise<nv_bfloat16, 2048, 4, 0, 2><<<num_blocks, 512>>>(NULL, input->data_ptr<nv_bfloat16>(), 
                                                                      absmax->data_ptr<float>(), output->data_ptr<unsigned char>(), 
                                                                      NULL, 0, size);
            break;
          case 4096:
            kQuantizeBlockwise<nv_bfloat16, 4096, 4, 0, 2><<<num_blocks, 1024>>>(NULL, input->data_ptr<nv_bfloat16>(), 
                                                                      absmax->data_ptr<float>(), output->data_ptr<unsigned char>(), 
                                                                      NULL, 0, size);
            break;
          default:
            HT_NOT_IMPLEMENTED << "Invalid blocksize:" << blocksize;
        }
      }
      else 
        HT_NOT_IMPLEMENTED << "Not support this quantization type:" << output->dtype();
  }
  NDArray::MarkUsedBy({input, absmax, output}, stream);
}

void DeQuantizationCuda(const NDArray& input, NDArray& absmax, 
                        const NDArray& code, NDArray& output, 
                        int64_t blocksize, const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output);
  HT_ASSERT_SAME_SHAPE(input, output);

  size_t size = output->numel();
  if (size == 0)
    return;
  int device_id = input->device().index();
  hetu::cuda::CUDADeviceGuard guard(device_id);
  CUDAStream cuda_stream(stream);
  int num_blocks = size / blocksize;
  num_blocks = size % blocksize == 0 ? num_blocks : num_blocks + 1;
  int tile_size = (input->dtype() != DataType::INT8) ? 1024 : 512;
  int blocks = (size + tile_size - 1) / tile_size;
  if (input->dtype() == DataType::INT8) {
      switch(output->dtype()) {
        case kFloat32:
          kDequantizeBlockwise<float, 512, 64, 8, 0><<<blocks, 64>>>(code->data_ptr<float>(), input->data_ptr<unsigned char>(), absmax->data_ptr<float>(), 
                                                                     output->data_ptr<float>(), blocksize, size);
          break;
        case kFloat16:
          kDequantizeBlockwise<half, 512, 64, 8, 0><<<blocks, 64>>>(code->data_ptr<float>(), input->data_ptr<unsigned char>(), absmax->data_ptr<float>(), 
                                                                    output->data_ptr<half>(), blocksize, size);
          break;                                                        
        case kBFloat16:
          kDequantizeBlockwise<nv_bfloat16, 512, 64, 8, 0><<<blocks, 64>>>(code->data_ptr<float>(), input->data_ptr<unsigned char>(), absmax->data_ptr<float>(), 
                                                                           output->data_ptr<nv_bfloat16>(), blocksize, size);
          break;
        default:
          HT_NOT_IMPLEMENTED << "Not support dequantization to this type:" << output->dtype();
      }   
  }
  else if (input->dtype() == DataType::FLOAT4) {
    switch(output->dtype()) {
        case kFloat32:
          kDequantizeBlockwise<float, 512, 64, 8, 1><<<blocks, 64>>>(NULL, input->data_ptr<unsigned char>(), absmax->data_ptr<float>(), 
                                                                     output->data_ptr<float>(), blocksize / 2, size);
          break;
        case kFloat16:
          kDequantizeBlockwise<half, 512, 64, 8, 1><<<blocks, 64>>>(NULL, input->data_ptr<unsigned char>(), absmax->data_ptr<float>(), 
                                                                    output->data_ptr<half>(), blocksize / 2, size);
          break;                                                        
        case kBFloat16:
          kDequantizeBlockwise<nv_bfloat16, 512, 64, 8, 1><<<blocks, 64>>>(NULL, input->data_ptr<unsigned char>(), absmax->data_ptr<float>(), 
                                                                           output->data_ptr<nv_bfloat16>(), blocksize / 2, size);
          break;
        default:
          HT_NOT_IMPLEMENTED << "Not support dequantization to this type:" << output->dtype();
      }
  }
  else if (input->dtype() == DataType::NFLOAT4) {
    switch(output->dtype()) {
        case kFloat32:
          kDequantizeBlockwise<float, 512, 64, 8, 2><<<blocks, 64>>>(NULL, input->data_ptr<unsigned char>(), absmax->data_ptr<float>(), 
                                                                     output->data_ptr<float>(), blocksize / 2, size);
          break;
        case kFloat16:
          kDequantizeBlockwise<half, 512, 64, 8, 2><<<blocks, 64>>>(NULL, input->data_ptr<unsigned char>(), absmax->data_ptr<float>(), 
                                                                    output->data_ptr<half>(), blocksize / 2, size);
          break;                                                        
        case kBFloat16:
          kDequantizeBlockwise<nv_bfloat16, 512, 64, 8, 2><<<blocks, 64>>>(NULL, input->data_ptr<unsigned char>(), absmax->data_ptr<float>(), 
                                                                           output->data_ptr<nv_bfloat16>(), blocksize / 2, size);
          break;
        default:
          HT_NOT_IMPLEMENTED << "Not support dequantization to this type:" << output->dtype();
      }
  }
  else 
    HT_NOT_IMPLEMENTED << "Not support this dequantization type:" << input->dtype();
  NDArray::MarkUsedBy({input, absmax, output}, stream);
}

void MatMul4BitCuda(const NDArray& A, bool trans_a, const NDArray& B, bool trans_b,
                    const NDArray& absmax, const NDArray& datatype, NDArray& out,
                    int blocksize, const Stream& stream)
{
  HT_ASSERT_CUDA_DEVICE(A);
  HT_ASSERT_SAME_DEVICE(A, B);
  HT_ASSERT_SAME_DEVICE(A, out);
  int n = trans_a? A->shape(1) : A->shape(0);
  int m = trans_b? B->shape(0) : B->shape(1);
  int k = trans_b? B->shape(1) : B->shape(0);
  int lda = trans_b? B->shape(0) : B->shape(1);
  int ldc = trans_b? B->shape(0) : B->shape(1);
  int ldb = (trans_a? A->shape(0) : A->shape(1) + 1) / 2;

  size_t size = out->numel();
  if (size == 0)
    return;
  int device_id = A->device().index();
  CUDAStream cuda_stream(stream);
  hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
  int num_blocks = (m + 3) / 4;
  switch(A->dtype()) {
    case kFloat32:
      kgemm_4bit_inference_naive<float, 128, 32><<<num_blocks, 128, 0, cuda_stream>>>(m, n, k, A->data_ptr<float>(), B->data_ptr<unsigned char>(), 
                                                                                      absmax->data_ptr<float>(), datatype->data_ptr<float>(), 
                                                                                      out->data_ptr<float>(), lda, ldb, ldc, blocksize);
      break;
    case kFloat16:
      kgemm_4bit_inference_naive<half, 128, 16><<<num_blocks, 128, 0, cuda_stream>>>(m, n, k, A->data_ptr<half>(), B->data_ptr<unsigned char>(), 
                                                                                     absmax->data_ptr<float>(), datatype->data_ptr<float>(), 
                                                                                     out->data_ptr<half>(), lda, ldb, ldc, blocksize);
    break;                                                        
    case kBFloat16:
      kgemm_4bit_inference_naive<nv_bfloat16, 128, 16><<<num_blocks, 128, 0, cuda_stream>>>(m, n, k, A->data_ptr<nv_bfloat16>(), B->data_ptr<unsigned char>(), 
                                                                                            absmax->data_ptr<float>(), datatype->data_ptr<float>(), 
                                                                                            out->data_ptr<nv_bfloat16>(), lda, ldb, ldc, blocksize);
    break;
    default:
      HT_NOT_IMPLEMENTED << "Not support for this type:" << A->dtype();
  }
  NDArray::MarkUsedBy({A, B, absmax, datatype, out}, stream);
}

} // namespace impl
} // namespace hetu
#pragma once

#include <torch/extension.h>

#include <map>
#include <vector>

void swap_blocks(torch::Tensor& src, torch::Tensor& dst,
                 const torch::Tensor& block_mapping);

void copy_blocks(std::vector<torch::Tensor>& key_caches,
                 std::vector<torch::Tensor>& value_caches,
                 const torch::Tensor& block_mapping);

void reshape_and_cache(
  torch::Tensor& key, torch::Tensor& value,
  
  const std::string& key_cache_data_ptr_str,
  int key_cache_block_size,
  int key_cache_x,
  const std::string& value_cache_data_ptr_str,   // [num_blocks, num_heads, head_size, block_size]

  torch::Tensor& slot_mapping,
  const std::string& kv_cache_dtype, const float kv_scale
);

void reshape_and_cache_flash(
  torch::Tensor& key, torch::Tensor& value,
  
  const std::string& key_cache_data_ptr_str,
  int key_cache_size_1,
  int key_cache_stride_0,
  const std::string& value_cache_data_ptr_str,   // [num_blocks, num_heads, head_size, block_size]
  int value_cache_stride_0,

  torch::Tensor& slot_mapping,
  const std::string& kv_cache_dtype
);

// Just for unittest
void convert_fp8(torch::Tensor& dst_cache, torch::Tensor& src_cache,
                 const float scale, const std::string& kv_cache_dtype);

#pragma once

#include <torch/extension.h>

#include <map>
#include <vector>

void swap_blocks(
  torch::Tensor& src,
  torch::Tensor& dst,
  const std::map<int64_t, int64_t>& block_mapping);

void copy_blocks(
  std::vector<torch::Tensor>& key_caches,
  std::vector<torch::Tensor>& value_caches,
  const std::map<int64_t, std::vector<int64_t>>& block_mapping
);

void reshape_and_cache(
  torch::Tensor& key,           // [num_tokens, num_heads, head_size]
  torch::Tensor& value,         // [num_tokens, num_heads, head_size]
  // torch::Tensor& key_cache,     // [num_blocks, num_heads, head_size/x, block_size, x]
  const std::string& key_cache_data_ptr_str,
  int key_cache_block_size,
  int key_cache_x,
  const std::string& value_cache_data_ptr_str,   // [num_blocks, num_heads, head_size, block_size]
  torch::Tensor& slot_mapping,  // [num_tokens]
  const std::string& kv_cache_dtype
);

// Just for unittest
void convert_fp8_e5m2(
  torch::Tensor& src_cache,
  torch::Tensor& dst_cache
);
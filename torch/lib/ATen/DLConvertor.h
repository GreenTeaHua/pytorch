#pragma once

#include "ATen/Tensor.h"
#include "ATen/ATen.h"
#include "ATen/dlpack.h"

// this convertor will:
// 1) take a Tensor object and wrap it in the DLPack tensor object
// 2) take a dlpack tensor and convert it to the Tensor object

namespace at {

ATen_CLASS DLTensor* toDLPack(const Tensor& src, DLTensor* dlTensor);
ATen_CLASS fromDLPack(const DLTensor* src);

} //namespace at

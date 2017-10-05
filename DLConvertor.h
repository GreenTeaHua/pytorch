#pragma once

#include "ATen/Tensor.h"
#include "ATen/ATen.h"
#include "ATen/dlpack.h"

#include "THP.h"

// this convertor will:
// 1) take a Tensor object and wrap it in the DLPack tensor object
// 2) take a dlpack tensor and convert it to the Tensor object

namespace at {

THP_CLASS DLTensor* toDLPack(const Tensor& src, DLTensor* dlTensor);
THP_CLASS Tensor fromDLPack(const DLTensor* src);

} //namespace at

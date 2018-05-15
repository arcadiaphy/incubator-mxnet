/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * Copyright (c) 2016 by Contributors
 * \file multibox_prior.cc
 * \brief generate multibox prior boxes cpu implementation
 * \author Joshua Zhang
*/

#include "./multibox_prior-inl.h"

namespace mshadow {
template<typename DType>
inline void MultiBoxPriorForward(const Tensor<cpu, 2, DType> &out,
                            const std::vector<float> &sizes,
                            const std::vector<float> &ratios,
                            const std::vector<int> densities,
                            const int in_width, const int in_height,
                            const int img_width, const int img_height,
                            const std::vector<float> &steps,
                            const std::vector<float> &offsets) {
  const float step_x = steps[1] > 0 ? steps[1] / img_width : 1.f / in_width;
  const float step_y = steps[0] > 0 ? steps[0] / img_height : 1.f / in_height;
  const int density_x = densities[1];
  const int density_y = densities[0];
  const float dstep_x = 1.f / (2 * density_x);
  const float dstep_y = 1.f / (2 * density_y);
  const int num_sizes = static_cast<int>(sizes.size());
  const int num_ratios = static_cast<int>(ratios.size());
  int count = 0;

  for (int r = 0; r < in_height; ++r) {
    float center_y = (r + offsets[0]) * step_y;
    for (int c = 0; c < in_width; ++c) {
      float center_x = (c + offsets[1]) * step_x;
      // ratio = 1, various sizes
      for (int i = 0; i < num_sizes; ++i) {
        float size = sizes[i];
        float w = size / img_width / 2;
        float h = size / img_height / 2;
        for (int m = 0; m < density_x; ++m) {
          for (int n = 0; n < density_y; ++n) {
            out[count][0] = center_x + dstep_x * step_x * (1 - density_x + 2 * m) - w;
            out[count][1] = center_y + dstep_y * step_y * (1 - density_y + 2 * n) - h;
            out[count][2] = center_x + dstep_x * step_x * (1 - density_x + 2 * m) + w;
            out[count][3] = center_y + dstep_y * step_y * (1 - density_y + 2 * n) + h;
            ++count;
          }
        }
      }
      // various ratios, size = min_size = size[0]
      float size = sizes[0];
      for (int j = 1; j < num_ratios; ++j) {
        float ratio = sqrtf(ratios[j]);
        float w = size / img_width * ratio / 2;
        float h = size / img_height / ratio / 2;
        for (int m = 0; m < density_x; ++m) {
          for (int n = 0; n < density_y; ++n) {
            out[count][0] = center_x + dstep_x * step_x * (1 - density_x + 2 * m) - w;
            out[count][1] = center_y + dstep_y * step_y * (1 - density_y + 2 * n) - h;
            out[count][2] = center_x + dstep_x * step_x * (1 - density_x + 2 * m) + w;
            out[count][3] = center_y + dstep_y * step_y * (1 - density_y + 2 * n) + h;
            ++count;
          }
        }
      }
    }
  }
}
}  // namespace mshadow

namespace mxnet {
namespace op {
template<>
Operator* CreateOp<cpu>(MultiBoxPriorParam param, int dtype) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new MultiBoxPriorOp<cpu, DType>(param);
  });
  return op;
}

Operator* MultiBoxPriorProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                                       std::vector<int> *in_type) const {
  std::vector<TShape> out_shape, aux_shape;
  std::vector<int> out_type, aux_type;
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
  CHECK(InferType(in_type, &out_type, &aux_type));
  DO_BIND_DISPATCH(CreateOp, param_, in_type->at(0));
}

DMLC_REGISTER_PARAMETER(MultiBoxPriorParam);

MXNET_REGISTER_OP_PROPERTY(_contrib_MultiBoxPrior, MultiBoxPriorProp)
.add_argument("data", "NDArray-or-Symbol", "Input data.")
.add_argument("image", "NDArray-or-Symbol", "Input image data.")
.add_arguments(MultiBoxPriorParam::__FIELDS__())
.describe("Generate prior(anchor) boxes from data, sizes and ratios.");

}  // namespace op
}  // namespace mxnet

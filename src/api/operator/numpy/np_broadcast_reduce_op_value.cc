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
 * \file broadcast_reduce_op_value.cc
 * \brief Implementation of the API of functions in
 * src/operator/tensor/np_broadcast_reduce_op_value.cc
 */
#include <mxnet/api_registry.h>
#include <mxnet/runtime/packed_func.h>
#include "../utils.h"
#include "../../../operator/tensor/broadcast_reduce_op.h"

namespace mxnet {

MXNET_REGISTER_API("_npi.broadcast_to")
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
  using namespace runtime;
  const nnvm::Op* op = Op::Get("_npi_broadcast_to");
  nnvm::NodeAttrs attrs;
  op::BroadcastToParam param;
  if (args[1].type_code() == kDLInt) {
    param.shape = TShape(1, args[1].operator int64_t());
  } else {
    param.shape = TShape(args[1].operator ObjectRef());
  }
  attrs.parsed = std::move(param);
  attrs.op = op;
  SetAttrDict<op::BroadcastToParam>(&attrs);

  int num_outputs = 0;
  NDArray* inputs[] = {args[0].operator mxnet::NDArray*()};
  auto ndoutputs = Invoke(op, &attrs, 1, inputs, &num_outputs, nullptr);
  *ret = ndoutputs[0];
});

}  // namespace mxnet

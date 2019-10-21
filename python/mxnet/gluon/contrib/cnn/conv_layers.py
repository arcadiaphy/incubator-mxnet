# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# coding: utf-8
# pylint: disable= arguments-differ
"""Custom convolutional neural network layers in model_zoo."""

__all__ = ['DeformableConvolution']

from .... import symbol
from ...block import HybridBlock
from ....base import numeric_types
from ...nn import Activation

class DeformableConvolution(HybridBlock):
    r"""2-D Deformable Convolution v_1 (Dai, 2017).
    Normal Convolution uses sampling points in a regular grid, while the sampling
    points of Deformablem Convolution can be offset. The offset is learned with a
    separate convolution layer during the training. Both the convolution layer for
    generating the output features and the offsets are included in this gluon layer.

    Parameters
    ----------
    channels : int
        The dimensionality of the output space
        i.e. the number of output channels in the convolution.
    kernel_size : int or tuple/list of 2 ints, default (1,1)
        Specifies the dimensions of the convolution window.
    strides : int or tuple/list of 2 ints, default (1,1)
        Specifies the strides of the convolution.
    padding : int or tuple/list of 2 ints, default (0,0)
        If padding is non-zero, then the input is implicitly zero-padded
        on both sides for padding number of points.
    dilation : int or tuple/list of 2 ints, default (1,1)
        Specifies the dilation rate to use for dilated convolution.
    groups : int, default 1
        Controls the connections between inputs and outputs.
        At groups=1, all inputs are convolved to all outputs.
        At groups=2, the operation becomes equivalent to having two convolution
        layers side by side, each seeing half the input channels, and producing
        half the output channels, and both subsequently concatenated.
    num_deformable_group : int, default 1
        Number of deformable group partitions.
    layout : str, default NCHW
        Dimension ordering of data and weight. Can be 'NCW', 'NWC', 'NCHW',
        'NHWC', 'NCDHW', 'NDHWC', etc. 'N', 'C', 'H', 'W', 'D' stands for
        batch, channel, height, width and depth dimensions respectively.
        Convolution is performed over 'D', 'H', and 'W' dimensions.
    use_bias : bool, default True
        Whether the layer for generating the output features uses a bias vector.
    in_channels : int, default 0
        The number of input channels to this layer. If not specified,
        initialization will be deferred to the first time `forward` is called
        and input channels will be inferred from the shape of input data.
    activation : str, default None
        Activation function to use. See :func:`~mxnet.ndarray.Activation`.
        If you don't specify anything, no activation is applied
        (ie. "linear" activation: :math:`a(x) = x`).
    weight_initializer : str or Initializer, default None
        Initializer for the `weight` weights matrix for the convolution layer
        for generating the output features.
    bias_initializer : str or Initializer, default zeros
        Initializer for the bias vector for the convolution layer
        for generating the output features.
    offset_weight_initializer : str or Initializer, default zeros
        Initializer for the  weight matrix for the convolution layer
        for generating the offset.
    offset_bias_initializer : str or Initializer, default zeros
        Initializer for the bias vector for the convolution layer
        for generating the offset.
    offset_use_bias: bool, default  True
        Whether the layer for generating the offset uses a bias vector.


    Inputs:
        - :attr:`data` 4D tensor with shape :math:`(N, C_{in}, H_{in}, W_{in})`
          when layout is 'NCHW'. For other layouts shape is permuted accordingly.

    Outputs:
        - :attr:`out`: 4D tensor with shape :math:`(N, C_{out}, H_{out}, W_{out})`
          when layout is 'NCHW' in which :math:`H_{out}`and :math:`W_{out}` are calculated as:

          .. math::
              H_{out} = \left\lfloor\frac
              {H_{in} + 2 \times \text{padding}[0] - (\text{dilation}[0] \times (\text{kernel_size}[0] - 1) - 1)}
              {\text{stride}[0]}\right\rfloor + 1
          .. math::
              W_{out} = \left\lfloor\frac
              {W_{in} + 2 \times \text{padding}[1] - (\text{dilation}[1] \times (\text{kernel_size}[1] - 1) - 1)}
              {\text{stride}[1]}\right\rfloor + 1
    """

    def __init__(self, channels, kernel_size=(1, 1), strides=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1,
                 num_deformable_group=1, layout='NCHW', use_bias=True, in_channels=0, activation=None,
                 weight_initializer=None, bias_initializer='zeros',
                 offset_weight_initializer='zeros', offset_bias_initializer='zeros', offset_use_bias=True,
                 op_name='DeformableConvolution', adj=None, prefix=None, params=None):
        super(DeformableConvolution, self).__init__(prefix=prefix, params=params)
        with self.name_scope():
            self._channels = channels
            self._in_channels = in_channels

            assert layout in ('NCHW', 'NHWC'), "Only supports 'NCHW' and 'NHWC' layout for now"
            if isinstance(kernel_size, numeric_types):
                kernel_size = (kernel_size,) * 2
            if isinstance(strides, numeric_types):
                strides = (strides,) * len(kernel_size)
            if isinstance(padding, numeric_types):
                padding = (padding,) * len(kernel_size)
            if isinstance(dilation, numeric_types):
                dilation = (dilation,) * len(kernel_size)
            self._op_name = op_name

            offset_channels = 2 * kernel_size[0] * kernel_size[1] * num_deformable_group
            self._kwargs_offset = {
                'kernel': kernel_size, 'stride': strides, 'dilate': dilation,
                'pad': padding, 'num_filter': offset_channels, 'num_group': groups,
                'no_bias': not offset_use_bias, 'layout': layout}

            self._kwargs_deformable_conv = {
                'kernel': kernel_size, 'stride': strides, 'dilate': dilation,
                'pad': padding, 'num_filter': channels, 'num_group': groups,
                'num_deformable_group': num_deformable_group,
                'no_bias': not use_bias, 'layout': layout}

            if adj:
                self._kwargs_offset['adj'] = adj
                self._kwargs_deformable_conv['adj'] = adj

            dshape = [0] * (len(kernel_size) + 2)
            dshape[layout.find('N')] = 1
            dshape[layout.find('C')] = in_channels

            op = getattr(symbol, 'Convolution')
            offset = op(symbol.var('data', shape=dshape), **self._kwargs_offset)

            offsetshapes = offset.infer_shape_partial()[0]

            self.offset_weight = self.params.get('offset_weight', shape=offsetshapes[1],
                                                 init=offset_weight_initializer,
                                                 allow_deferred_init=True)

            if offset_use_bias:
                self.offset_bias = self.params.get('offset_bias', shape=offsetshapes[2],
                                                   init=offset_bias_initializer,
                                                   allow_deferred_init=True)
            else:
                self.offset_bias = None

            deformable_conv_weight_shape = [0] * (len(kernel_size) + 2)
            deformable_conv_weight_shape[0] = channels
            deformable_conv_weight_shape[2] = kernel_size[0]
            deformable_conv_weight_shape[3] = kernel_size[1]

            self.deformable_conv_weight = self.params.get('deformable_conv_weight',
                                                          shape=deformable_conv_weight_shape,
                                                          init=weight_initializer,
                                                          allow_deferred_init=True)

            if use_bias:
                self.deformable_conv_bias = self.params.get('deformable_conv_bias', shape=(channels,),
                                                            init=bias_initializer,
                                                            allow_deferred_init=True)
            else:
                self.deformable_conv_bias = None

            if activation:
                self.act = Activation(activation, prefix=activation + '_')
            else:
                self.act = None

    def hybrid_forward(self, F, x, offset_weight, deformable_conv_weight, offset_bias=None, deformable_conv_bias=None):
        if offset_bias is None:
            offset = F.Convolution(x, offset_weight, cudnn_off=True, **self._kwargs_offset)
        else:
            offset = F.Convolution(x, offset_weight, offset_bias, cudnn_off=True, **self._kwargs_offset)

        if deformable_conv_bias is None:
            act = F.contrib.DeformableConvolution(data=x, offset=offset, weight=deformable_conv_weight,
                                                  name='fwd', **self._kwargs_deformable_conv)
        else:
            act = F.contrib.DeformableConvolution(data=x, offset=offset, weight=deformable_conv_weight,
                                                  bias=deformable_conv_bias, name='fwd',
                                                  **self._kwargs_deformable_conv)

        if self.act:
            act = self.act(act)
        return act

    def _alias(self):
        return 'deformable_conv'

    def __repr__(self):
        s = '{name}({mapping}, kernel_size={kernel}, stride={stride}'
        len_kernel_size = len(self._kwargs_deformable_conv['kernel'])
        if self._kwargs_deformable_conv['pad'] != (0,) * len_kernel_size:
            s += ', padding={pad}'
        if self._kwargs_deformable_conv['dilate'] != (1,) * len_kernel_size:
            s += ', dilation={dilate}'
        if hasattr(self, 'out_pad') and self.out_pad != (0,) * len_kernel_size:
            s += ', output_padding={out_pad}'.format(out_pad=self.out_pad)
        if self._kwargs_deformable_conv['num_group'] != 1:
            s += ', groups={num_group}'
        if self.deformable_conv_bias is None:
            s += ', bias=False'
        if self.act:
            s += ', {}'.format(self.act)
        s += ')'
        shape = self.deformable_conv_weight.shape
        return s.format(name=self.__class__.__name__,
                        mapping='{0} -> {1}'.format(shape[1] if shape[1] else None, shape[0]),
                        **self._kwargs_deformable_conv)

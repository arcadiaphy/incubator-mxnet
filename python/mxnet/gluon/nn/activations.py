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
"""Basic neural network layers."""
__all__ = ['Activation', 'LeakyReLU', 'PReLU', 'ELU', 'SELU', 'Swish', 'GELU']

from ... import initializer
from ..block import HybridBlock
from ...util import is_np_array


class Activation(HybridBlock):
    r"""Applies an activation function to input.

    Parameters
    ----------
    activation : str
        Name of activation function to use.
        See :func:`~mxnet.ndarray.Activation` for available choices.


    Inputs:
        - :attr:`data` input tensor with arbitrary shape.

    Outputs:
        - :attr:`out` output tensor with the same shape as :attr:`data`.
    """
    def __init__(self, activation, **kwargs):
        self._act_type = activation
        super(Activation, self).__init__(**kwargs)

    def _alias(self):
        return self._act_type

    def hybrid_forward(self, F, x):
        act = F.npx.activation if is_np_array() else F.Activation
        return act(x, act_type=self._act_type, name='fwd')

    def __repr__(self):
        s = '{name}({_act_type})'
        return s.format(name=self.__class__.__name__,
                        **self.__dict__)


class LeakyReLU(HybridBlock):
    r"""Leaky version of a Rectified Linear Unit.

    It allows a small gradient when the unit is not active

    .. math::
        \text{LeakyReLU}(x) =
        \begin{cases}
        x, & \text{if} \ x > 0 \\
        \alpha * x, & \text{otherwise}
        \end{cases}

    Parameters
    ----------
    alpha : float
        The :math:`\alpha` parameter in the equation. Must be >= 0.


    Inputs:
        - :attr:`data` input tensor with arbitrary shape.

    Outputs:
        - :attr:`out` output tensor with the same shape as :attr:`data`.
    """
    def __init__(self, alpha, **kwargs):
        assert alpha >= 0, "Slope coefficient for LeakyReLU must be no less than 0."
        super(LeakyReLU, self).__init__(**kwargs)
        self._alpha = alpha

    def hybrid_forward(self, F, x):
        leaky_relu = F.npx.leaky_relu if is_np_array() else F.LeakyReLU
        return leaky_relu(x, act_type='leaky', slope=self._alpha, name='fwd')

    def __repr__(self):
        s = '{name}({alpha})'
        return s.format(name=self.__class__.__name__,
                        alpha=self._alpha)


class PReLU(HybridBlock):
    r"""Parametric leaky version of a Rectified Linear Unit.

    It learns a gradient when the unit is not active

    .. math::
        \text{PReLU}(x) =
        \begin{cases}
        x, & \text{if} \ x > 0 \\
        \alpha * x, & \text{otherwise}
        \end{cases}

    where :math:`\alpha` is a learnable parameter.

    Parameters
    ----------
    alpha_initializer : Initializer
        Initializer for the :math:`\alpha`.


    Inputs:
        - :attr:`data` input tensor with arbitrary shape.

    Outputs:
        - :attr:`out` output tensor with the same shape as :attr:`data`.

    References
    ----------
        `Parametric leaky ReLU <https://arxiv.org/abs/1502.01852>`_
    """
    def __init__(self, alpha_initializer=initializer.Constant(0.25), **kwargs):
        super(PReLU, self).__init__(**kwargs)
        with self.name_scope():
            self.alpha = self.params.get('alpha', shape=(1,), init=alpha_initializer)

    def hybrid_forward(self, F, x, alpha):
        return F.LeakyReLU(x, gamma=alpha, act_type='prelu', name='fwd')


class ELU(HybridBlock):
    r"""Exponential Linear Unit (ELU)

    Applies the element-wise function:

    .. math::
        \text{ELU}(x) =
        \begin{cases}
        x, & \text{if} \ x > 0 \\
        \alpha * (\exp(x) - 1), & \text{otherwise}
        \end{cases}

    Parameters
    ----------
    alpha : float
        The :math:`\alpha` parameter in the equation.


    Inputs:
        - :attr:`data` input tensor with arbitrary shape.

    Outputs:
        - :attr:`out` output tensor with the same shape as :attr:`data`.

    References
    ----------
        `Fast and Accurate Deep Network Learning by Exponential Linear Units
        <https://arxiv.org/abs/1511.07289>`_
    """

    def __init__(self, alpha=1.0, **kwargs):
        super(ELU, self).__init__(**kwargs)
        self._alpha = alpha

    def hybrid_forward(self, F, x):
        return F.LeakyReLU(x, act_type='elu', slope=self._alpha)


class SELU(HybridBlock):
    r"""Scaled Exponential Linear Unit (SELU)

    Applies the element-wise function:

    .. math::
        \text{SELU}(x) =
        \begin{cases}
        \text{scale} * x, & \text{if} \ x > 0 \\
        \text{scale} * \alpha (\exp(x) - 1), & \text{otherwise}
        \end{cases}

    where :math:`\text{scale} = 1.0507009873554804934193349852946`
    and :math:`\alpha = 1.6732632423543772848170429916717`.


    Inputs:
        - :attr:`data` input tensor with arbitrary shape.

    Outputs:
        - :attr:`out` output tensor with the same shape as :attr:`data`.

    References
    ----------
        `Self-Normalizing Neural Networks <https://arxiv.org/abs/1706.02515>`_
    """
    def __init__(self, **kwargs):
        super(SELU, self).__init__(**kwargs)

    def hybrid_forward(self, F, x):
        return F.LeakyReLU(x, act_type='selu', name='fwd')

class GELU(HybridBlock):
    r"""Gaussian Exponential Linear Unit (GELU)

    Applies the element-wise function:

    .. math:: \text{GELU}(x) = x * \Phi(x)

    where :math:`\Phi(x)` is the cumulative distribution function for Gaussian distribution.


    Inputs:
        - :attr:`data` input tensor with arbitrary shape.

    Outputs:
        - :attr:`out` output tensor with the same shape as :attr:`data`.

    References
    ----------
        `Gaussian Error Linear Units (GELUs) <https://arxiv.org/abs/1606.08415>`_
    """
    def __init__(self, **kwargs):
        super(GELU, self).__init__(**kwargs)

    def hybrid_forward(self, F, x):
        return F.LeakyReLU(x, act_type='gelu', name='fwd')


class Swish(HybridBlock):
    r"""Swish Activation function.

    Applies the element-wise function:

    .. math:: \text{swish}(x) = x * \text{sigmoid}(\beta * x)


    Parameters
    ----------
    beta : float
        The :math:`\beta` parameter in the equation.


    Inputs:
        - :attr:`data` input tensor with arbitrary shape.

    Outputs:
        - :attr:`out` output tensor with the same shape as :attr:`data`.

    References
    ----------
        `Searching for Activation Functions <https://arxiv.org/abs/1710.05941>`_
    """

    def __init__(self, beta=1.0, **kwargs):
        super(Swish, self).__init__(**kwargs)
        self._beta = beta

    def hybrid_forward(self, F, x):
        return x * F.sigmoid(self._beta * x, name='fwd')

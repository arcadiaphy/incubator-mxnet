.. Licensed to the Apache Software Foundation (ASF) under one
   or more contributor license agreements.  See the NOTICE file
   distributed with this work for additional information
   regarding copyright ownership.  The ASF licenses this file
   to you under the Apache License, Version 2.0 (the
   "License"); you may not use this file except in compliance
   with the License.  You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing,
   software distributed under the License is distributed on an
   "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
   KIND, either express or implied.  See the License for the
   specific language governing permissions and limitations
   under the License.

gluon.data
=============

Gluon provides a lot of data utilities in this module. The general data
APIs and the specific APIs for image data are listed according to their categories:


Dataset
--------------

.. currentmodule:: mxnet.gluon.data

.. autosummary::
    :nosignatures:

    Dataset
    SimpleDataset
    ArrayDataset
    RecordFileDataset


Sampler
----------------------------

.. autosummary::
    :nosignatures:

    Sampler
    BatchSampler
    FilterSampler
    SequentialSampler
    RandomSampler


Dataloader
------------------------

.. autosummary::
    :nosignatures:

    DataLoader


Image Dataset
--------------

.. currentmodule:: mxnet.gluon.data.vision

.. autosummary::
    :nosignatures:

   MNIST
   FashionMNIST
   CIFAR10
   CIFAR100
   ImageRecordDataset
   ImageFolderDataset


Image Data Transformation
----------------------------

.. currentmodule:: mxnet.gluon.data.vision.transforms

.. autosummary::
    :nosignatures:

    Compose
    Cast
    ToTensor
    Normalize
    RandomResizedCrop
    CenterCrop
    Resize
    RandomFlipLeftRight
    RandomFlipTopBottom
    RandomBrightness
    RandomContrast
    RandomSaturation
    RandomHue
    RandomColorJitter
    RandomLighting


API Reference
-------------


.. automodule:: mxnet.gluon.data
    :members:
    :imported-members:

.. automodule:: mxnet.gluon.data.vision
    :members:
    :imported-members:

.. automodule:: mxnet.gluon.data.vision.transforms
    :members:

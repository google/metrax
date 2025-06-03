# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import metrax
from metrax.nnx import nnx_wrapper

NnxWrapper = nnx_wrapper.NnxWrapper


class AUCPR(NnxWrapper):
  """An NNX class for the Metrax metric AUCPR."""

  def __init__(self):
    super().__init__(metrax.AUCPR)


class AUCROC(NnxWrapper):
  """An NNX class for the Metrax metric AUCROC."""

  def __init__(self):
    super().__init__(metrax.AUCROC)


class Accuracy(NnxWrapper):
  """An NNX class for the Metrax metric Accuracy."""

  def __init__(self):
    super().__init__(metrax.Accuracy)


class Average(NnxWrapper):
  """An NNX class for the Metrax metric Average."""

  def __init__(self):
    super().__init__(metrax.Average)


class AveragePrecisionAtK(NnxWrapper):
  """An NNX class for the Metrax metric AveragePrecisionAtK."""

  def __init__(self):
    super().__init__(metrax.AveragePrecisionAtK)


class BLEU(NnxWrapper):
  """An NNX class for the Metrax metric BLEU."""

  def __init__(self):
    super().__init__(metrax.BLEU)


class DCGAtK(NnxWrapper):
  """An NNX class for the Metrax metric DCGAtK."""

  def __init__(self):
    super().__init__(metrax.DCGAtK)


class Dice(NnxWrapper):
  """An NNX class for the Metrax metric Dice."""

  def __init__(self):
    super().__init__(metrax.Dice)

class FBetaScore(NnxWrapper):
  """An NNX class for the Metrax metric FBetaScore."""

  def __init__(self):
    super().__init__(metrax.FBetaScore)

class IoU(NnxWrapper):
  """An NNX class for the Metrax metric IoU."""

  def __init__(self):
    super().__init__(metrax.IoU)


class MAE(NnxWrapper):
  """An NNX class for the Metrax metric MAE."""

  def __init__(self):
    super().__init__(metrax.MAE)


class MRR(NnxWrapper):
  """An NNX class for the Metrax metric MRR."""

  def __init__(self):
    super().__init__(metrax.MRR)


class MSE(NnxWrapper):
  """An NNX class for the Metrax metric MSE."""

  def __init__(self):
    super().__init__(metrax.MSE)


class NDCGAtK(NnxWrapper):
  """An NNX class for the Metrax metric NDCGAtK."""

  def __init__(self):
    super().__init__(metrax.NDCGAtK)


class Perplexity(NnxWrapper):
  """An NNX class for the Metrax metric Perplexity."""

  def __init__(self):
    super().__init__(metrax.Perplexity)


class Precision(NnxWrapper):
  """An NNX class for the Metrax metric Precision."""

  def __init__(self):
    super().__init__(metrax.Precision)


class PrecisionAtK(NnxWrapper):
  """An NNX class for the Metrax metric PrecisionAtK."""

  def __init__(self):
    super().__init__(metrax.PrecisionAtK)


class PSNR(NnxWrapper):
  """An NNX class for the Metrax metric PSNR."""

  def __init__(self):
    super().__init__(metrax.PSNR)


class Recall(NnxWrapper):
  """An NNX class for the Metrax metric Recall."""

  def __init__(self):
    super().__init__(metrax.Recall)


class RecallAtK(NnxWrapper):
  """An NNX class for the Metrax metric RecallAtK."""

  def __init__(self):
    super().__init__(metrax.RecallAtK)


class RMSE(NnxWrapper):
  """An NNX class for the Metrax metric RMSE."""

  def __init__(self):
    super().__init__(metrax.RMSE)


class RougeL(NnxWrapper):
  """An NNX class for the Metrax metric RougeL."""

  def __init__(self):
    super().__init__(metrax.RougeL)


class RougeN(NnxWrapper):
  """An NNX class for the Metrax metric RougeN."""

  def __init__(self):
    super().__init__(metrax.RougeN)


class RSQUARED(NnxWrapper):
  """An NNX class for the Metrax metric RSQUARED."""

  def __init__(self):
    super().__init__(metrax.RSQUARED)


class SNR(NnxWrapper):
  """An NNX class for the Metrax metric SNR."""

  def __init__(self):
    super().__init__(metrax.SNR)


class SSIM(NnxWrapper):
  """An NNX class for the Metrax metric SSIM."""

  def __init__(self):
    super().__init__(metrax.SSIM)


class WER(NnxWrapper):
  """An NNX class for the Metrax metric WER."""

  def __init__(self):
    super().__init__(metrax.WER)

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

from metrax.nnx import nnx_metrics

AUCPR = nnx_metrics.AUCPR
AUCROC = nnx_metrics.AUCROC
Accuracy = nnx_metrics.Accuracy
Average = nnx_metrics.Average
AveragePrecisionAtK = nnx_metrics.AveragePrecisionAtK
BLEU = nnx_metrics.BLEU
DCGAtK = nnx_metrics.DCGAtK
Dice = nnx_metrics.Dice
FBetaScore = nnx_metrics.FBetaScore
IoU = nnx_metrics.IoU
MAE = nnx_metrics.MAE
MRR = nnx_metrics.MRR
MSE = nnx_metrics.MSE
NDCGAtK = nnx_metrics.NDCGAtK
Perplexity = nnx_metrics.Perplexity
Precision = nnx_metrics.Precision
PrecisionAtK = nnx_metrics.PrecisionAtK
PSNR = nnx_metrics.PSNR
RMSE = nnx_metrics.RMSE
RSQUARED = nnx_metrics.RSQUARED
Recall = nnx_metrics.Recall
RecallAtK = nnx_metrics.RecallAtK
RougeL = nnx_metrics.RougeL
RougeN = nnx_metrics.RougeN
SNR = nnx_metrics.SNR
SSIM = nnx_metrics.SSIM
WER = nnx_metrics.WER


__all__ = [
    "AUCPR",
    "AUCROC",
    "Average",
    "AveragePrecisionAtK",
    "BLEU",
    "DCGAtK",
    "Dice",
    "FBetaScore",
    "IoU",
    "MRR",
    "MAE",
    "MSE",
    "NDCGAtK",
    "Perplexity",
    "Precision",
    "PrecisionAtK",
    "PSNR",
    "RMSE",
    "RSQUARED",
    "Recall",
    "RecallAtK",
    "RougeL",
    "RougeN",
    "SNR",
    "SSIM",
    "WER",
]

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

from metrax import audio_metrics
from metrax import base
from metrax import classification_metrics
from metrax import image_metrics
from metrax import nlp_metrics
from metrax import ranking_metrics
from metrax import regression_metrics

AUCPR = classification_metrics.AUCPR
AUCROC = classification_metrics.AUCROC
Accuracy = classification_metrics.Accuracy
Average = base.Average
AveragePrecisionAtK = ranking_metrics.AveragePrecisionAtK
BLEU = nlp_metrics.BLEU
DCGAtK = ranking_metrics.DCGAtK
Dice = image_metrics.Dice
FBetaScore = classification_metrics.FBetaScore
IoU = image_metrics.IoU
MAE = regression_metrics.MAE
MRR = ranking_metrics.MRR
MSE = regression_metrics.MSE
NDCGAtK = ranking_metrics.NDCGAtK
Perplexity = nlp_metrics.Perplexity
Precision = classification_metrics.Precision
PrecisionAtK = ranking_metrics.PrecisionAtK
PSNR = image_metrics.PSNR
RMSE = regression_metrics.RMSE
RSQUARED = regression_metrics.RSQUARED
Recall = classification_metrics.Recall
RecallAtK = ranking_metrics.RecallAtK
RougeL = nlp_metrics.RougeL
RougeN = nlp_metrics.RougeN
SNR = audio_metrics.SNR
SSIM = image_metrics.SSIM
WER = nlp_metrics.WER


__all__ = [
    "AUCPR",
    "AUCROC",
    "Accuracy",
    "Average",
    "AveragePrecisionAtK",
    "BLEU",
    "DCGAtK",
    "Dice",
    "FBetaScore",
    "IoU",
    "MAE",
    "MRR",
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

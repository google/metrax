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

from metrax import base
from metrax import classification_metrics
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
MAE = regression_metrics.MAE
MRR = ranking_metrics.MRR
MSE = regression_metrics.MSE
NDCGAtK = ranking_metrics.NDCGAtK
Perplexity = nlp_metrics.Perplexity
Precision = classification_metrics.Precision
PrecisionAtK = ranking_metrics.PrecisionAtK
RMSE = regression_metrics.RMSE
RSQUARED = regression_metrics.RSQUARED
Recall = classification_metrics.Recall
RecallAtK = ranking_metrics.RecallAtK
RougeL = nlp_metrics.RougeL
RougeN = nlp_metrics.RougeN
WER = nlp_metrics.WER


__all__ = [
    "AUCPR",
    "AUCROC",
    "Accuracy",
    "Average",
    "AveragePrecisionAtK",
    "BLEU",
    "DCGAtK",
    "MAE",
    "MRR",
    "MSE",
    "NDCGAtK",
    "Perplexity",
    "Precision",
    "PrecisionAtK",
    "RMSE",
    "RSQUARED",
    "Recall",
    "RecallAtK",
    "RougeL",
    "RougeN",
    "WER",
]

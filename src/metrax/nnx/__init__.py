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
Average = nnx_metrics.Average
AveragePrecisionAtK = nnx_metrics.AveragePrecisionAtK
BLEU = nnx_metrics.BLEU
MSE = nnx_metrics.MSE
Perplexity = nnx_metrics.Perplexity
Precision = nnx_metrics.Precision
RMSE = nnx_metrics.RMSE
RSQUARED = nnx_metrics.RSQUARED
Recall = nnx_metrics.Recall
WER = nnx_metrics.WER


__all__ = [
    "AUCPR",
    "AUCROC",
    "Average",
    "AveragePrecisionAtK",
    "BLEU",
    "MSE",
    "Perplexity",
    "Precision",
    "RMSE",
    "RSQUARED",
    "Recall",
    "WER",
]

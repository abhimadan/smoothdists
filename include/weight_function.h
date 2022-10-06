#pragma once

#include "utility.h"

// weight_coeffs must have 10 entries
BOTH float triCubicWeight(const float* weight_coeffs, float s, float t);
BOTH float triCubicWeightGradS(const float* weight_coeffs, float s, float t);
BOTH float triCubicWeightGradT(const float* weight_coeffs, float s, float t);

BOTH float tri7Weight(const float* weight_coeffs, float s, float t);
BOTH float tri7WeightGradS(const float* weight_coeffs, float s, float t);
BOTH float tri7WeightGradT(const float* weight_coeffs, float s, float t);

// weight_coeffs must have 5 entries
BOTH float edgeQuarticWeight(const float* weight_coeffs, float t);
BOTH float edgeQuarticWeightGrad(const float* weight_coeffs, float t);

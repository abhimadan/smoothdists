#include "weight_function.h"

BOTH float triCubicWeight(const float* weight_coeffs, float s, float t) {
  return weight_coeffs[0] + weight_coeffs[1]*s + weight_coeffs[2]*t +
         weight_coeffs[3]*s*s + weight_coeffs[4]*s*t +
         weight_coeffs[5]*t*t + weight_coeffs[6]*s*s*s +
         weight_coeffs[7]*s*s*t + weight_coeffs[8]*s*t*t +
         weight_coeffs[9]*t*t*t;
}

BOTH float triCubicWeightGradS(const float* weight_coeffs, float s, float t) {
  return weight_coeffs[1] + 2.f*weight_coeffs[3]*s + weight_coeffs[4]*t +
         3.f*weight_coeffs[6]*s*s + 2.f*weight_coeffs[7]*s*t +
         weight_coeffs[8]*t*t;
}

BOTH float triCubicWeightGradT(const float* weight_coeffs, float s, float t) {
  return weight_coeffs[2] + weight_coeffs[4]*s + 2.f*weight_coeffs[5]*t +
         weight_coeffs[7]*s*s + 2.f*weight_coeffs[8]*s*t +
         3.f*weight_coeffs[9]*t*t;
}

BOTH float tri7Weight(const float* weight_coeffs, float s, float t) {
  float s2 = s*s;
  float s3 = s2*s;
  float s4 = s2*s2;
  float s5 = s2*s3;
  float s6 = s3*s3;
  float s7 = s4*s3;

  float t2 = t*t;
  float t3 = t2*t;
  float t4 = t2*t2;
  float t5 = t2*t3;
  float t6 = t3*t3;
  float t7 = t4*t3;

  return weight_coeffs[0] +
         weight_coeffs[1]*(s2 + t2) +
         weight_coeffs[2]*(s3 + t3) +
         weight_coeffs[3]*(s4 + t4) +
         weight_coeffs[4]*s2*t2 +
         weight_coeffs[5]*(s5 + t5) +
         weight_coeffs[6]*(s2*t3 + s3*t2) +
         weight_coeffs[7]*(s6 + t6) +
         weight_coeffs[8]*(s2*t4 + s4*t2) +
         weight_coeffs[9]*s3*t3 +
         weight_coeffs[10]*(s7 + t7) +
         weight_coeffs[11]*(s2*t5 + s5*t2) +
         weight_coeffs[12]*(s3*t4 + s4*t3);
}

BOTH float tri7WeightGradS(const float* weight_coeffs, float s, float t) {
  float s2 = s*s;
  float s3 = s2*s;
  float s4 = s2*s2;
  float s5 = s2*s3;
  float s6 = s3*s3;

  float t2 = t*t;
  float t3 = t2*t;
  float t4 = t2*t2;
  float t5 = t2*t3;
  float t6 = t3*t3;

  return weight_coeffs[1]*2*s +
         weight_coeffs[2]*3*s2 +
         weight_coeffs[3]*4*s3 +
         weight_coeffs[4]*2*s*t2 +
         weight_coeffs[5]*5*s4 +
         weight_coeffs[6]*(2*s*t3 + 3*s2*t2) +
         weight_coeffs[7]*6*s5 +
         weight_coeffs[8]*(2*s*t4 + 4*s3*t2) +
         weight_coeffs[9]*3*s2*t3 +
         weight_coeffs[10]*7*s6 +
         weight_coeffs[11]*(2*s*t5 + 5*s4*t2) +
         weight_coeffs[12]*(3*s2*t4 + 4*s3*t3);
}

BOTH float tri7WeightGradT(const float* weight_coeffs, float s, float t) {
  float s2 = s*s;
  float s3 = s2*s;
  float s4 = s2*s2;
  float s5 = s2*s3;
  float s6 = s3*s3;

  float t2 = t*t;
  float t3 = t2*t;
  float t4 = t2*t2;
  float t5 = t2*t3;
  float t6 = t3*t3;

  return weight_coeffs[1]*2*t +
         weight_coeffs[2]*3*t2 +
         weight_coeffs[3]*4*t3 +
         weight_coeffs[4]*s2*2*t +
         weight_coeffs[5]*5*t4 +
         weight_coeffs[6]*(s2*3*t2 + s3*2*t) +
         weight_coeffs[7]*6*t5 +
         weight_coeffs[8]*(s2*4*t3 + s4*2*t) +
         weight_coeffs[9]*s3*3*t2 +
         weight_coeffs[10]*7*t6 +
         weight_coeffs[11]*(s2*5*t4 + s5*2*t) +
         weight_coeffs[12]*(s3*4*t3 + s4*3*t2);
}

BOTH float edgeQuarticWeight(const float* weight_coeffs, float t) {
  return weight_coeffs[0] + weight_coeffs[1]*t + weight_coeffs[2]*t*t +
         weight_coeffs[3]*t*t*t + weight_coeffs[4]*t*t*t*t;
}

BOTH float edgeQuarticWeightGrad(const float* weight_coeffs, float t) {
  return weight_coeffs[1] + 2*weight_coeffs[2]*t +
         3*weight_coeffs[3]*t*t + 4*weight_coeffs[4]*t*t*t;
}

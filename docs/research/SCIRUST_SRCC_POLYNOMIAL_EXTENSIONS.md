# SRCC Polynomial Spectral Extensions

Status: deferred research track.

This document records polynomial extensions planned after validation of
the core SciRust Resonant Consensus Closure.

## 1. Chebyshev spectral filters

For a symmetric alignment operator Gamma whose spectrum is scaled into
[-1, 1], define:

T_0(Gamma) = I
T_1(Gamma) = Gamma
T_{n+1}(Gamma) = 2 Gamma T_n(Gamma) - T_{n-1}(Gamma)

A degree-d filter is:

P_d(Gamma) = sum_{k=0}^d c_k T_k(Gamma)

Purpose:

- sharp spectral transitions;
- controlled pass-band ripple;
- configurable rejection profiles;
- optimization over coefficients rather than matrix coordinates.

Evaluation should use deterministic matrix Clenshaw recurrence, without
forming successive powers of Gamma.

## 2. Bernstein spectral filters

For an operator Lambda with spectrum in [0, 1]:

B_{k,n}(Lambda)
=
binomial(n,k) Lambda^k (I - Lambda)^(n-k)

and:

F(Lambda)
=
sum_{k=0}^n beta_k B_{k,n}(Lambda)

with beta_k in [0, 1].

Monotone ordered coefficients may provide:

- smooth attenuation;
- bounded gains;
- reduced oscillation;
- stable interpolation between retained and rejected directions.

## 3. Ordered non-commutative operator words

A future extension may use ordered left and right actions:

P(X)
=
A_0
+ A_1 X
+ X A_2
+ A_3 X A_4
+ A_5 X^2
+ X^2 A_6
+ X A_7 X

These terms must be represented as explicit real-linear operators.
Evaluation order is part of the mathematical definition.

This track must not introduce hidden dependence on Cayley or Clifford.
Hypercomplex actions may be optional transport families, not SRCC
primitives.

## 4. Integration with SRCC

The planned architecture is:

1. SRCC generates a resonant closure.
2. The closure defines an alignment or inhibition operator.
3. A polynomial spectral layer shapes attenuation around that operator.
4. Train/dev optimization selects degree and coefficients.
5. Hard, soft, Chebyshev and Bernstein variants are compared under the
   same signal-distortion objective.

## 5. Required rigor

A fixed evaluated matrix polynomial is still a linear operator.
Therefore its kernel is a linear subspace.

Algebraic varieties may arise in:

- coefficient space;
- parameterized operator families;
- nonlinear constraints imposed on admissible filters.

Future implementations must also verify:

- spectral scaling;
- symmetry preservation;
- finite coefficients;
- deterministic accumulation;
- Clenshaw parity against direct recurrence;
- monotonicity for constrained Bernstein coefficients;
- projector or contraction bounds where claimed.

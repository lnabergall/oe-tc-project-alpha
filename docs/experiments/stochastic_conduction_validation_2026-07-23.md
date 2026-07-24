# Stochastic conduction validation (2026-07-23)

## Purpose

The original deterministic pair-averaging operator conserved internal energy,
but erased finite-reservoir fluctuations. Combined with exposure-dependent bath
coupling, it produced a persistent source-off circulation: in the mature
`n=64` control, exposed particles absorbed about `0.1985` energy per sweep while
fully shielded particles released about `0.1888`.

The replacement must:

1. conserve pair internal energy;
2. move heat preferentially, but not exclusively, from hot to cold particles;
3. let bonds alter transport speed without altering equilibrium;
4. preserve the same canonical law as direct bath exchange; and
5. remain a fixed-shape, parallel JAX kernel.

## Kernel

For an adjacent pair `(i, j)`, one sign is sampled uniformly and proposes a
fixed quantum `delta_c` in that direction. The candidate is

\[
(E_i',E_j')=(E_i-s\delta_c,E_j+s\delta_c).
\]

It is rejected below the energy floor. Otherwise its affinity is

\[
\mathcal A=C\log(E_i'/E_i)+C\log(E_j'/E_j),
\]

in the implementation's `k_B=1` units, and it is accepted with probability

\[
\lambda_{ij}\min(1,e^{\mathcal A}).
\]

Here `lambda_ij` is `conduction_contact` or `conduction_bond`. It is symmetric
under reversal, so it changes kinetics only. The implemented sampler derives
the sign and conditional acceptance uniform from one random variate per edge.
Four static edge matchings provide endpoint-disjoint parallel updates; their
order is randomly permuted each sweep.

The forward/reverse probability ratio is exactly `exp(A)`. Because pair energy
is conserved, the bath Boltzmann term cancels and this is the ratio required by

\[
\pi(\mathbf E)\propto\prod_i E_i^C e^{-E_i/T_b}.
\]

The direct bath kernel preserves the same law on each fixed-quantum energy
ladder. Unit tests verify the ratio in both directions, floor rejection,
contact/bond kinetics, matching independence, conservation, JIT equivalence,
and deterministic replay from a key.

## Defaults

The first calibrated defaults are:

| parameter | value |
|---|---:|
| `conduction_energy_quantum` | 0.25 |
| `conduction_contact` | 0.04 |
| `conduction_bond` | 0.20 |
| `bath_energy_quantum` | 0.25 |

Equal contact and bond frequencies, including both being zero, are now valid
for controlled ablations. The quantum matches direct bath exchange so the two
kernels share a convenient discrete energy ladder. The frequencies are doubled
relative to the former deterministic coefficients because a stochastic signed
proposal does not transfer energy on every active edge.

## Frozen-geometry control

A source-off `n=32`, `N=256` state was run for 5,000 sweeps, then frozen. It
contained one 256-particle molecule, 338 bonds, 215 exposed particles, and 41
fully shielded particles. The thermal control used 2,048 sweeps, eight
replicates, and 45 combinations of energy distribution, conduction, and bath
coupling. Canonical backgrounds were sampled from the exact discrete law of the
fixed-quantum bath kernel rather than from its continuous Gamma approximation.

For the canonical-background condition with the default bath:

| conduction | initial energy | final mean | late exposed bath flux | late shielded bath flux |
|---|---:|---:|---:|---:|
| default (`0.04`, `0.20`) | 174.50 | 177.81 | -0.0043 | -0.0047 |
| equal (`0.04`, `0.04`) | 174.50 | 178.47 | -0.0035 | -0.0042 |
| off | 174.50 | 178.16 | -0.0036 | -0.0060 |

The same sampled background microstate was reused across replicates, so its
small common relaxation is not an unbiased stationary-ensemble estimate. The
important checks are that the final energies do not shift systematically with
conduction strength and that exposure classes do not sustain opposing fluxes.
The late per-class fluxes are at least an order of magnitude below the former
`+0.1985/-0.1888` circulation and are comparable across the conduction-off and
conduction-on controls. The exact forward/reverse tests remain the primary
equilibrium guarantee.

Conduction still performs its intended transport function. For excess energy
localized on a fully shielded particle under the default bath:

| conduction | matched half-life | fraction dissipated by sweep 2,048 | mean throughput/sweep |
|---|---:|---:|---:|
| default | 754 | 100.0% | 10.40 |
| equal contact/bond | 1,901 | 54.0% | 2.12 |
| off | not reached | 23.2% | 0 |

Thus bonded paths substantially accelerate spreading to bath-accessible
particles without changing the source-off equilibrium law.

## Full-model sanity checks

Final-code `n=32`, `N=256`, 5,000-sweep runs used the same seed with the source
on and off. Over the final 1,000 sweeps:

| condition | mean internal energy | source flux | direct bath flux | structural bath flux | conduction exchanges/sweep | bath exchanges/sweep |
|---|---:|---:|---:|---:|---:|---:|
| source on | 449.08 | 2.354 | -1.345 | -1.038 | 46.92 | 38.43 |
| source off | 179.82 | 0 | -0.0045 | -0.0001 | 39.49 | 44.03 |

The source-on input and bath losses are closely balanced rather than producing
linear heating. The source-off mean is essentially the discrete canonical mean
(`0.7002` per particle, or `179.25` for 256 particles). Conduction and direct
bath exchange occur at comparable event counts under the defaults.

These are calibration checks, not evidence of open-endedness. Larger systems,
multiple seeds, and long structural controls remain necessary before drawing
claims about dissipation-enhancing self-organization.

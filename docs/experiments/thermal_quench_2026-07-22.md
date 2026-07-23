# Frozen-geometry thermal-transport controls (2026-07-22)

## Question

Does conduction accelerate dissipation because distributing excess energy over
more particles opens more independent fixed-quantum bath exchanges? Does the
bonded structure formed by the model improve that transport?

## Design

The experiment freezes particle positions and bonds, disables irradiation and
all structural transitions, and applies only one randomized conduction pass
followed by direct bath exchange. It uses the 64 by 64, 1,024-particle, seed-0
`run_004` state. Each condition has four common-random-number replicates and
runs for 4,096 thermal sweeps.

The direct bath kernel can propose only one signed quantum of 0.25 per particle
and sweep. Its maximum aggregate cooling capacity therefore grows with the
number of energetic bath-coupled particles. The experiment measures the
effective number of carriers using

\[
N_{\mathrm{eff}}
=
\frac{(\sum_i e_i)^2}{\sum_i e_i^2},
\qquad
e_i=\max\{E_i-\langle E\rangle_b,0\}.
\]

The bath reference is a reproducible canonical sample from

\[
p_b(E)\propto E^C e^{-E/T_b},
\]

conditioned on the energy floor. With the defaults, this is a Gamma law with
shape `C + 1 = 3.5`, scale `T_b = 0.2`, and mean 0.7. Its sampled total was
719.98. Every quench condition had total internal energy 874.97, giving the
same 154.98-unit excess, initialized as one of:

- the checkpoint's actual energy distribution;
- a uniform distribution;
- the canonical background plus all excess on one fully shielded particle;
- the same background plus all excess on one fully exposed particle.

The canonical background itself is run for every transport setting. Reported
excess relaxation is matched to that baseline:

\[
\Delta E(t)=E_{\mathrm{quench}}(t)-E_{\mathrm{background}}(t).
\]

This subtraction is necessary because deterministic conduction changes the
source-off stationary distribution, as discussed below.

The conduction settings were:

| name | ordinary contact | bonded contact |
|---|---:|---:|
| default | 0.02 | 0.10 |
| no bond advantage | 0.02 | 0.02 |
| off | 0 | 0 |

The bath settings were:

| name | coupling `kappa(epsilon)` |
|---|---|
| default | `0.10 + 0.40 epsilon` |
| transport limited | `0.01 + 0.49 epsilon` |
| exposed only | `0.50 epsilon` |

Here `epsilon` is the exposed-face fraction.

## Fixed-quantum transport result

On the mature sweep-10,000 geometry, the matched excess half-lives under the
default bath were:

| initial excess distribution | default conduction | no bond advantage | conduction off |
|---|---:|---:|---:|
| uniform | 12 | 9 | 8 |
| actual checkpoint | 55 | 58 | 47 |
| concentrated, shielded | 139 | 243 | not reached |
| concentrated, exposed and isolated | 1,797 | 1,797 | 1,797 |

The shielded hot spot retained 76.0% of its matched excess after 4,096 sweeps
with conduction off, while default conduction dissipated 99.8%. The bonded
conduction advantage reduced its half-life by 43% relative to equal contact and
bond conduction. This directly supports the proposed mechanism: conduction is
important when excess energy begins localized inside a connected structure,
because it increases the number of particles through which fixed-quantum bath
loss can occur.

Conduction did not improve an already uniform quench and barely changed the
actual checkpoint quench. Nor could it affect the fully exposed hot particle,
which was isolated and had no conductive edge. The effect is therefore not a
generic consequence of adding a conduction operator; it depends on energy
localization and network connectivity.

Removing direct bath access from shielded particles did not destroy the mature
network's effect:

| bath setting | shielded-hot-spot half-life with default conduction |
|---|---:|
| default | 139 |
| transport limited | 171 |
| exposed only | 178 |

By contrast, with conduction off, the same shielded hot spot dissipated 24.0%,
2.3%, and approximately 0% of its excess under those three bath settings. The
default `kappa_base=0.1` accelerates the quench, but it is not solely responsible
for it. Conduction can carry the excess to exposed sinks even when shielded
particles cannot exchange bath heat directly.

## Structure comparison

The same quench was applied to the earliest available structure at sweep 128
and the mature structure at sweep 10,000:

| property | sweep 128 | sweep 10,000 |
|---|---:|---:|
| bonds | 988 | 1,019 |
| molecules | 176 | 15 |
| largest molecule | 47 | 343 |
| exposed particles | 959 | 741 |
| fully shielded particles | 65 | 283 |
| mean exposed fraction | 0.478 | 0.272 |

For the shielded hot spot, default-conduction half-life fell from 395 to 139
sweeps, a 2.84-fold speedup. With the bond advantage removed it fell from 400
to 243 sweeps, a 1.65-fold speedup. Average absolute conduction throughput rose
from 9.78 to 11.66 energy units per sweep. The bonded advantage did essentially
nothing in the early fragmented state, but produced a 1.75-fold half-life
speedup in the mature state. These results are consistent with coarsening
creating a more effective internal transport network.

This is not a universal claim that mature structure dissipates everything more
quickly. A uniform quench relaxed in 7 sweeps on the exposed early geometry and
12 on the more shielded mature geometry; the actual-distribution half-lives
were 56 and 55. Organization changed *which spatial energy distributions* were
efficiently routed to the bath. That conditional control of energy flow is more
relevant to thermodynamic computation than maximizing total dissipation alone.

## Thermodynamic caveat

The matched background uncovered a significant model issue. Starting from the
canonical bath sample, the mature geometry's final mean totals were:

| conduction | initial background | final background |
|---|---:|---:|
| default | 719.98 | 607.67 |
| no bond advantage | 719.98 | 653.80 |
| off | 719.98 | 719.30 |

Thus deterministic pair averaging and the stochastic bath do not share the
same equilibrium invariant distribution. In the last 512 sweeps of the mature
default-background control, exposed particles absorbed 0.1985 energy per sweep
while fully shielded particles released 0.1888; net heat was near zero. This is
a persistent internal current in a source-off system coupled to one nominally
uniform bath.

The cause is that deterministic conduction removes finite-particle energy
fluctuations without a reverse fluctuating conduction channel. It conserves
energy, but it does not preserve the product canonical law implied by the
particle entropy and direct-bath kernel. Consequently, raw cooling curves
cannot be interpreted thermodynamically without the matched-background
subtraction, and source-off equilibrium depends on the conduction coefficient.

A thermodynamically consistent replacement should use stochastic,
energy-conserving pair exchange satisfying

\[
\frac{k((E_i,E_j)\to(E_i-q,E_j+q))}
{k((E_i-q,E_j+q)\to(E_i,E_j))}
=
\exp\!\left[
\frac{S(E_i-q)+S(E_j+q)-S(E_i)-S(E_j)}{k_B}
\right].
\]

A fixed-quantum Metropolis exchange is the simplest implementation. A
rejection-free alternative is to resample the conserved pair-energy split from
the corresponding Beta conditional distribution. Either kernel would retain
bond-dependent kinetic prefactors while preserving source-off equilibrium.

## Conclusion

The user's parallel-bath-channel hypothesis is supported, and the mature bond
network routes a localized shielded energy packet substantially better than the
early fragmented structure. The result is conditional rather than universal:
network transport matters for localized energy and can be counteracted by
geometric shielding when energy is already distributed.

Before using dissipation as evidence of thermodynamic computation or
open-endedness, the conduction kernel should be made equilibrium preserving.
After that correction, repeat this frozen-geometry test over several structural
checkpoints and independent simulation seeds, then relate transport gains to
specific molecular topology and later source-capture performance.

# Preliminary thermal-balance controls (2026-07-22)

## Question

Do the default conduction and bath couplings permit transport-dependent
dissipation, and is the falling total energy in the initial driven runs still a
long equilibration transient at 10,000 sweeps?

## Conditions

All comparisons use a 64 by 64 cylindrical lattice, 1,024 particles, seed 0,
10,000 sweeps, 128-sweep compiled chunks, and the default parameters unless
stated otherwise. `run_004` is the existing source-on trajectory. The two new
controls are ignored run artifacts under `data/runs/`:

- `control_source_off_n64_seed0`: irradiation disabled;
- `control_source_off_no_conduction_n64_seed0`: irradiation disabled,
  `conduction_contact=0`, and `conduction_bond=1e-7`.

The effectively-zero bonded coefficient preserves the model's strict
`conduction_contact < conduction_bond` invariant.

## Results

| condition | cumulative source | direct bath | structural bath | net external flux | final internal E | final configurational E | final bonds | final molecules |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| source on, default conduction | 51007.4 | -39791.5 | -12264.2 | -1048.3 | 875.0 | -1463.9 | 1461 | 15 |
| source off, default conduction | 0 | -1798.0 | 565.5 | -1232.5 | 612.2 | -1385.3 | 1385 | 8 |
| source off, conduction effectively off | 0 | -500.5 | -654.1 | -1154.6 | 723.8 | -1418.8 | 1417 | 10 |

Flux is signed positive into the particle system. Conduction is conservative
and therefore is not itself part of `net external flux`.

For the final 1,000 sweeps of the driven run, the mean per-sweep fluxes were:

- source: +5.1021;
- direct bath: -3.7325;
- structural bath: -1.3639;
- net: +0.00575.

The driven system is therefore close to a nonequilibrium steady flux balance by
10,000 sweeps. The larger 20,000-sweep `run_005` is stronger evidence: its
last-5,000-sweep total-energy slope was approximately 0.00014 per sweep and its
mean net flux was -0.0184 per sweep. The pronounced early decline is primarily
a coarsening transient in which configurational energy becomes more negative.
It is not evidence of indefinitely continuing cooling.

In the source-off control, most of the decline also occurred in the first 1,000
sweeps. During sweeps 9,001--10,000, the total-energy regression slope was
-0.00313 per sweep and mean net bath flux was -0.01975 per sweep. This is close
to equilibrium on the scale of the early transient, though finite-size noise
and slow bond turnover remain.

## Interpretation

Conduction is not irrelevant under the current defaults. Disabling it reduced
cumulative direct bath loss by a factor of 3.6 and left 18% more internal energy
at sweep 10,000. It changed total loss by only about 6.7%, however, because the
structural bath channel changed sign and compensated. The total-energy trace
alone therefore cannot diagnose whether energy was transported through the
particle network before dissipation.

The present `kappa_base=0.1` still gives a fully shielded particle direct bath
access. This weakens, but does not eliminate, the need for conduction. A useful
transport-limited candidate is `kappa_base=0.01` and
`kappa_exposure=0.49`, which retains the same maximum coupling of 0.5 while
making exposed surfaces much more important. This should be tested as an
ablation rather than adopted from this single seed.

The source-on/source-off comparison is an equilibrium control, not by itself a
test that self-organization improves dissipation. The driven run is both hotter
and more bonded, so its larger output flux is confounded by greater available
energy.

## Next evaluation

A stronger causal test should use ensembles and controlled initial states:

1. Run source-on and source-off ensembles over multiple seeds until late-window
   flux and structural observables are stationary.
2. Branch the same energized checkpoint into source-off quenches that preserve
   geometry or randomize geometry while preserving particle energies and bond
   count. Compare relaxation time and cumulative heat output.
3. Ablate conduction and vary `kappa_base` while holding peak exposed coupling
   fixed.
4. Record conservative conduction throughput, such as the sum of absolute edge
   flux per sweep, and bath heat resolved by particle exposure. These observables
   are needed to distinguish local bath short-circuiting from structure-mediated
   energy routing.
5. Relate lagged changes in structure, molecular diversity, and energy-current
   topology to later dissipation. Faster dissipation is neither necessary nor
   sufficient for open-endedness; persistent structure-dependent control of
   energy flow is the more relevant criterion.

These controls are preliminary because all three 64 by 64 comparisons use one
seed and dynamically diverge after the first differing update.

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from oe_tc.config import StaticConfig, default_params
from oe_tc.geometry import build_lattice
from oe_tc.molecules import (
    CONTACT_EFFECT_RECORDS_PER_PARTICLE,
    apply_selected_molecule_moves,
    bond_geometry_preserved,
    changed_contact_effects,
    effect_thermodynamics,
    lexicographic_pair_membership,
    minimum_image_displacement,
    molecule_metadata,
    propose_molecule_moves,
)
from oe_tc.state import State


def _state(
    positions: jax.Array,
    energy: jax.Array,
    bonds: jax.Array,
    root: jax.Array,
    n: int,
) -> State:
    return State(
        R=positions,
        E=energy,
        L=build_lattice(positions, n),
        bonds=bonds,
        root=root,
        sweep=jnp.asarray(0, dtype=jnp.uint32),
    )


def test_metadata_is_linear_root_segment_aggregation() -> None:
    metadata = molecule_metadata(jnp.asarray((0, 0, 2, 2, 2, 5), jnp.int32))
    np.testing.assert_array_equal(metadata.segment_size, (2, 0, 3, 0, 0, 1))
    np.testing.assert_array_equal(
        metadata.is_root, (True, False, True, False, False, True)
    )
    assert all(field.shape == (6,) for field in metadata)


def test_pair_membership_uses_unpacked_int32_pairs() -> None:
    # Packing first*radix+second would overflow int32 for these identifiers.
    first = jnp.asarray((2_000_000_000, 1, 2_000_000_000), jnp.int32)
    second = jnp.asarray((7, 9, 5), jnp.int32)
    query_first = jnp.asarray((1, 2_000_000_000, 2_000_000_000), jnp.int32)
    query_second = jnp.asarray((9, 5, 8), jnp.int32)
    expected = np.asarray((True, True, False))
    np.testing.assert_array_equal(
        lexicographic_pair_membership(first, second, query_first, query_second),
        expected,
    )
    compiled = jax.jit(lexicographic_pair_membership)(
        first, second, query_first, query_second
    )
    np.testing.assert_array_equal(compiled, expected)


def test_changed_contact_energy_is_symmetric_exact_and_floor_checked() -> None:
    positions = jnp.asarray(((1, 1), (3, 1)), jnp.int32)
    state = _state(
        positions,
        jnp.asarray((1.0, 2.0)),
        jnp.zeros(2, jnp.uint8),
        jnp.asarray((0, 1), jnp.int32),
        6,
    )
    # Root 0 translates next to root 1, forming exactly one weak contact.
    candidate = jnp.asarray(((2, 1), (3, 1)), jnp.int32)
    delta_u, member_delta, effects = changed_contact_effects(
        state,
        candidate,
        internal_channel=jnp.asarray((True, False)),
        active=jnp.asarray((True, False)),
        eta=0.1,
    )
    np.testing.assert_allclose(delta_u, (-0.1, 0.0), atol=1e-7)
    np.testing.assert_allclose(member_delta, (0.05, 0.0), atol=1e-7)
    valid_particle = np.asarray(effects.particle[effects.valid])
    valid_delta = np.asarray(effects.delta[effects.valid])
    np.testing.assert_array_equal(valid_particle, (0, 1))
    np.testing.assert_allclose(valid_delta, (0.05, 0.05), atol=1e-7)
    assert np.isclose(valid_delta.sum(), -float(delta_u[0]))

    entropy, floor_valid = effect_thermodynamics(effects, state.E, 10.0, 0.1)
    expected_entropy = 10.0 * (np.log(1.05 / 1.0) + np.log(2.05 / 2.0))
    np.testing.assert_allclose(entropy[0], expected_entropy, rtol=2e-6)
    assert bool(floor_valid[0])

    # Reversing the move removes the contact and costs eta/2 at each endpoint.
    contacted = state._replace(R=candidate, L=build_lattice(candidate, 6), E=jnp.asarray((0.12, 0.12)))
    delta_u, _, effects = changed_contact_effects(
        contacted,
        positions,
        internal_channel=jnp.asarray((True, False)),
        active=jnp.asarray((True, False)),
        eta=0.1,
    )
    np.testing.assert_allclose(delta_u[0], 0.1, atol=1e-7)
    _, floor_valid = effect_thermodynamics(effects, contacted.E, 10.0, 0.1)
    assert not bool(floor_valid[0])


def test_rotation_preserves_bond_endpoints_and_rejects_winding_lift() -> None:
    # Ordinary dimer: a CCW turn maps its +x/-x bond to +y/-y.
    positions = jnp.asarray(((2, 2), (3, 2)), jnp.int32)
    dimer = _state(
        positions,
        jnp.ones(2),
        jnp.asarray((1, 2), jnp.uint8),
        jnp.asarray((0, 0), jnp.int32),
        6,
    )
    candidate = jnp.asarray(((2, 2), (2, 3)), jnp.int32)
    translate = jnp.zeros(2, jnp.bool_)
    turns = jnp.ones(2, jnp.int32)
    assert bool(bond_geometry_preserved(dimer, candidate, translate, turns, 6)[0])
    compiled = jax.jit(
        lambda transformed: bond_geometry_preserved(
            dimer, transformed, translate, turns, 6
        )
    )(candidate)
    assert bool(compiled[0])

    # A ring winding around the cylinder has no globally consistent planar
    # quarter-turn lift.  The pointwise minimum-image branch breaks one seam
    # bond even though all resulting sites below are distinct and in bounds.
    ring_R = jnp.stack(
        (jnp.arange(6, dtype=jnp.int32), jnp.full((6,), 3, jnp.int32)), axis=-1
    )
    ring = _state(
        ring_R,
        jnp.ones(6),
        jnp.full((6,), 3, jnp.uint8),
        jnp.zeros(6, jnp.int32),
        6,
    )
    displacement, _ = minimum_image_displacement(ring_R, ring_R[0], 6)
    winding_candidate = ring_R[0] + jnp.stack(
        (-displacement[:, 1], displacement[:, 0]), axis=-1
    )
    assert not bool(
        bond_geometry_preserved(
            ring,
            winding_candidate,
            jnp.zeros(6, jnp.bool_),
            jnp.ones(6, jnp.int32),
            6,
        )[0]
    )
    rotation_only = default_params()._replace(
        translation_probability=0.0,
        rotation_frequency=1_000.0,
        bath_channel_probability=1.0,
    )
    proposal = propose_molecule_moves(
        jax.random.key(19), ring, rotation_only, StaticConfig(6, 6)
    )
    assert not bool(proposal.legal[0])


def test_monomer_proposal_is_fixed_shape_jittable_translation() -> None:
    positions = jnp.asarray(((2, 2),), jnp.int32)
    state = _state(
        positions,
        jnp.asarray((100.0,)),
        jnp.zeros(1, jnp.uint8),
        jnp.zeros(1, jnp.int32),
        6,
    )
    params = default_params()._replace(
        translation_probability=0.0,
        translation_frequency=1_000.0,
        bath_channel_probability=1.0,
    )
    static = StaticConfig(6, 1)
    proposal = jax.jit(
        lambda key: propose_molecule_moves(key, state, params, static)
    )(jax.random.key(5))
    assert bool(proposal.translate[0])
    assert bool(proposal.legal[0]) and bool(proposal.accepted[0])
    assert proposal.effects.root.shape == (
        CONTACT_EFFECT_RECORDS_PER_PARTICLE,
    )
    displacement = np.asarray(proposal.candidate_R[0] - state.R[0])
    # Account for a possible periodic x seam, although this particle is interior.
    assert np.abs(displacement).sum() == 1

def test_antipodal_rotation_legality_is_symmetric_and_jittable() -> None:
    # This non-winding chain crosses the periodic seam.  Rotating the vertical
    # embedding clockwise about particle 0 produces the horizontal embedding,
    # but the latter has a particle at horizontal offset n/2 from that pivot.
    # Checking only the source-state lift made this a one-way transition.
    n = 8
    horizontal_R = jnp.asarray(
        ((6, 3), (7, 3), (0, 3), (1, 3), (2, 3)), jnp.int32
    )
    vertical_R = jnp.asarray(
        ((6, 3), (6, 4), (6, 5), (6, 6), (6, 7)), jnp.int32
    )
    horizontal = _state(
        horizontal_R,
        jnp.full((5,), 100.0),
        jnp.asarray((1, 3, 3, 3, 2), jnp.uint8),
        jnp.zeros(5, jnp.int32),
        n,
    )
    vertical = _state(
        vertical_R,
        jnp.full((5,), 100.0),
        jnp.asarray((4, 12, 12, 12, 8), jnp.uint8),
        jnp.zeros(5, jnp.int32),
        n,
    )
    params = default_params()._replace(
        translation_probability=0.0,
        rotation_frequency=1_000.0,
        bath_channel_probability=1.0,
    )
    static = StaticConfig(n, 5)
    keys = jax.random.split(jax.random.key(0), 100)

    # These fixed keys choose pivot 0 and opposite quarter turns.  The vertical
    # proposal constructs the exact horizontal state, so its post-state
    # antipodal lift must make it illegal.  The nominal reverse is likewise
    # illegal because its source-state lift is ambiguous.
    reverse = jax.jit(
        lambda key: propose_molecule_moves(key, vertical, params, static)
    )(keys[2])
    forward = propose_molecule_moves(keys[10], horizontal, params, static)
    assert int(reverse.pivot[0]) == 0 and int(reverse.quarter_turns[0]) == -1
    np.testing.assert_array_equal(reverse.candidate_R, horizontal_R)
    assert int(forward.pivot[0]) == 0 and int(forward.quarter_turns[0]) == 1
    assert not bool(reverse.legal[0]) and not bool(reverse.accepted[0])
    assert not bool(forward.legal[0]) and not bool(forward.accepted[0])

    _, horizontal_unique = minimum_image_displacement(
        horizontal_R, jnp.broadcast_to(horizontal_R[0], horizontal_R.shape), n
    )
    _, vertical_unique = minimum_image_displacement(
        vertical_R, jnp.broadcast_to(vertical_R[0], vertical_R.shape), n
    )
    assert not bool(jnp.all(horizontal_unique))
    assert bool(jnp.all(vertical_unique))

def test_selected_move_application_is_sparse_exact_and_jittable() -> None:
    positions = jnp.asarray(((1, 2), (6, 5)), jnp.int32)
    state = _state(
        positions,
        jnp.asarray((100.0, 120.0)),
        jnp.zeros(2, jnp.uint8),
        jnp.arange(2, dtype=jnp.int32),
        8,
    )
    params = default_params()._replace(
        translation_probability=1.0,
        translation_frequency=1_000.0,
        bath_channel_probability=1.0,
    )
    static = StaticConfig(8, 2)
    proposals = propose_molecule_moves(jax.random.key(31), state, params, static)
    assert bool(jnp.all(proposals.legal))

    apply_compiled = jax.jit(
        lambda chosen: apply_selected_molecule_moves(
            state, proposals, chosen, static
        )
    )
    unchanged = apply_compiled(jnp.zeros(2, jnp.bool_))
    for before, after in zip(state, unchanged, strict=True):
        np.testing.assert_array_equal(after, before)

    moved = apply_compiled(jnp.ones(2, jnp.bool_))
    np.testing.assert_array_equal(moved.R, proposals.candidate_R)
    np.testing.assert_array_equal(moved.E, state.E)
    np.testing.assert_array_equal(
        moved.L, build_lattice(proposals.candidate_R, static.n, static.empty)
    )

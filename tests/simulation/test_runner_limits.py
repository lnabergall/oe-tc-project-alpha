import pytest

from oe_tc import runner


@pytest.mark.parametrize("steps", [-1, 2**32])
def test_sweep_target_must_fit_checkpoint_counter(steps, tmp_path):
    args = runner.build_parser().parse_args(
        ("--steps", str(steps), "--output", str(tmp_path / "run"))
    )
    with pytest.raises(ValueError, match="unsigned 32-bit sweep target"):
        runner.resolve_run_spec(args, None)

"""Phase 1 Verification Gate: headless physics sanity checks."""

import math
from src.config import MAX_POWER, TABLE_HEIGHT, TABLE_WIDTH
from src.physics import BilliardPhysics


def test_straight_pocket_shot() -> None:
    """Shoot cue ball straight into a target ball in front of the top-middle pocket."""
    sim = BilliardPhysics()
    # Cue ball at center-bottom, target ball right in front of top-middle pocket (400, 0)
    sim.reset(cue_pos=(400.0, 200.0), target_positions=[(400.0, 60.0)])

    # Shoot straight up (-pi/2)
    sim.apply_shot(angle=-math.pi / 2.0, power=600.0)
    res = sim.step_until_stopped()

    print(
        f"[straight_pocket] pocketed={res.pocketed_targets} "
        f"hit={res.cue_hit_target} scratch={res.cue_scratched} "
        f"steps={res.steps} timed_out={res.timed_out}"
    )
    assert res.cue_hit_target, "Cue ball should have hit target ball"
    assert res.pocketed_targets == [0], f"Expected target 0 pocketed, got {res.pocketed_targets}"
    assert not res.cue_scratched, "Cue ball should not have scratched"
    assert not res.timed_out, "Simulation should stop automatically before max_steps"
    assert len(sim.target_balls) == 0, "Pocketed target ball must be removed from space"


def test_cue_scratch() -> None:
    """Shoot cue ball directly into a corner pocket and verify scratch detection."""
    sim = BilliardPhysics()
    sim.reset(cue_pos=(100.0, 100.0), target_positions=[(600.0, 300.0)])

    # Shoot straight at top-left pocket (0, 0) -> angle = -3*pi/4
    sim.apply_shot(angle=-3.0 * math.pi / 4.0, power=500.0)
    res = sim.step_until_stopped()

    print(
        f"[cue_scratch]     pocketed={res.pocketed_targets} "
        f"hit={res.cue_hit_target} scratch={res.cue_scratched} "
        f"steps={res.steps} timed_out={res.timed_out}"
    )
    assert res.cue_scratched, "Cue ball should have scratched"
    assert sim.cue_body is None, "Scratched cue ball must be removed from space"
    assert not res.cue_hit_target, "Cue ball should not have hit target ball"
    assert not res.timed_out, "Simulation should stop cleanly"


def test_cushion_bounce_and_stop() -> None:
    """Shoot cue ball hard into cushion and verify it stays in bounds and stops."""
    sim = BilliardPhysics()
    sim.reset(cue_pos=(400.0, 200.0), target_positions=[(200.0, 300.0)])

    # Shoot right (0 rad) into the right rail midpoint (no pocket at (800, 200))
    sim.apply_shot(angle=0.0, power=MAX_POWER)
    res = sim.step_until_stopped()

    assert sim.cue_body is not None, "Cue ball should remain on table"
    cx, cy = sim.cue_body.position
    print(
        f"[cushion_bounce]  final_cue=({cx:.1f}, {cy:.1f}) "
        f"steps={res.steps} timed_out={res.timed_out}"
    )
    assert 0.0 < cx < TABLE_WIDTH and 0.0 < cy < TABLE_HEIGHT, "Cue ball out of bounds"
    assert not res.timed_out, "High-power shot should still stop before max_steps"


def test_rotated_cue_shot() -> None:
    """Verify apply_shot uses world-space direction even when cue_body is rotated."""
    sim = BilliardPhysics()
    sim.reset(cue_pos=(400.0, 200.0), target_positions=[(400.0, 60.0)])
    assert sim.cue_body is not None
    sim.cue_body.angle = 1.5  # Simulate spin from a prior collision
    sim.apply_shot(angle=-math.pi / 2.0, power=600.0)
    res = sim.step_until_stopped()
    assert res.pocketed_targets == [0], f"Rotated cue ball missed straight shot: {res}"


if __name__ == "__main__":
    test_straight_pocket_shot()
    test_cue_scratch()
    test_cushion_bounce_and_stop()
    test_rotated_cue_shot()
    print("ALL PHASE 1 PHYSICS TESTS PASSED.")

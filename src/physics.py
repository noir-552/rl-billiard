"""Headless 2D billiard physics engine built on Pymunk."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import pymunk

from src.config import (
    BALL_ELASTICITY,
    BALL_FRICTION,
    BALL_MASS,
    BALL_RADIUS,
    COLLISION_CUE,
    COLLISION_CUSHION,
    COLLISION_TARGET,
    CUSHION_ELASTICITY,
    CUSHION_RADIUS,
    DT,
    MAX_SIM_STEPS,
    POCKET_RADIUS,
    POCKETS,
    STOP_VELOCITY,
    TABLE_DAMPING,
    TABLE_HEIGHT,
    TABLE_WIDTH,
)


@dataclass
class ShotResult:
    pocketed_targets: list[int] = field(default_factory=list)
    cue_hit_target: bool = False
    first_target_hit: int | None = None
    cue_scratched: bool = False
    steps: int = 0
    timed_out: bool = False


class BilliardPhysics:
    """2D billiard table simulation with 6 pockets and cushion collisions."""

    def __init__(self) -> None:
        self.space: pymunk.Space = pymunk.Space()
        self.cue_body: pymunk.Body | None = None
        self.cue_shape: pymunk.Circle | None = None
        self.target_balls: dict[int, tuple[pymunk.Body, pymunk.Circle]] = {}
        self._first_target_hit: int | None = None
        self._build_space()

    def _build_space(self) -> None:
        self.space = pymunk.Space()
        self.space.gravity = (0.0, 0.0)
        self.space.damping = TABLE_DAMPING
        self._add_cushions()
        self.space.on_collision(
            COLLISION_CUE,
            COLLISION_TARGET,
            begin=self._on_cue_target_collision,
        )

    def _add_cushions(self) -> None:
        w, h = TABLE_WIDTH, TABLE_HEIGHT
        pr, cr = POCKET_RADIUS, CUSHION_RADIUS
        segments = [
            ((pr, -cr), (w / 2.0 - pr, -cr)),
            ((w / 2.0 + pr, -cr), (w - pr, -cr)),
            ((pr, h + cr), (w / 2.0 - pr, h + cr)),
            ((w / 2.0 + pr, h + cr), (w - pr, h + cr)),
            ((-cr, pr), (-cr, h - pr)),
            ((w + cr, pr), (w + cr, h - pr)),
        ]
        for a, b in segments:
            seg = pymunk.Segment(self.space.static_body, a, b, cr)
            seg.elasticity = CUSHION_ELASTICITY
            seg.friction = BALL_FRICTION
            seg.collision_type = COLLISION_CUSHION
            self.space.add(seg)

    def _on_cue_target_collision(
        self, arbiter: pymunk.Arbiter, space: pymunk.Space, data: dict
    ) -> None:
        if self._first_target_hit is None:
            for shape in arbiter.shapes:
                if shape.collision_type == COLLISION_TARGET:
                    self._first_target_hit = getattr(shape, "ball_id", None)
                    break

    def _create_ball(
        self, pos: tuple[float, float], collision_type: int, ball_id: int | None = None
    ) -> tuple[pymunk.Body, pymunk.Circle]:
        moment = pymunk.moment_for_circle(BALL_MASS, 0.0, BALL_RADIUS)
        body = pymunk.Body(BALL_MASS, moment)
        body.position = pos
        shape = pymunk.Circle(body, BALL_RADIUS)
        shape.elasticity = BALL_ELASTICITY
        shape.friction = BALL_FRICTION
        shape.collision_type = collision_type
        if ball_id is not None:
            shape.ball_id = ball_id  # type: ignore[attr-defined]
        self.space.add(body, shape)
        return body, shape

    def reset(
        self,
        cue_pos: tuple[float, float],
        target_positions: list[tuple[float, float]],
    ) -> None:
        """Reset table with a cue ball and target balls at given positions."""
        self._build_space()
        self.cue_body, self.cue_shape = self._create_ball(cue_pos, COLLISION_CUE)
        self.target_balls = {
            i: self._create_ball(pos, COLLISION_TARGET, ball_id=i)
            for i, pos in enumerate(target_positions)
        }
        self._first_target_hit = None

    def place_cue_ball(self, pos: tuple[float, float]) -> None:
        """Place or reposition the cue ball (e.g. after a scratch)."""
        if self.cue_body is not None and self.cue_shape is not None:
            self.space.remove(self.cue_body, self.cue_shape)
        self.cue_body, self.cue_shape = self._create_ball(pos, COLLISION_CUE)

    def place_target_ball(self, ball_id: int, pos: tuple[float, float]) -> None:
        """Place or respot a specific target ball (e.g. 9-ball respot on foul)."""
        if ball_id in self.target_balls:
            old_body, old_shape = self.target_balls[ball_id]
            self.space.remove(old_body, old_shape)
        self.target_balls[ball_id] = self._create_ball(pos, COLLISION_TARGET, ball_id=ball_id)

    def apply_shot(self, angle: float, power: float) -> None:
        """Apply an impulse to the cue ball in direction `angle` (radians) with `power`."""
        if self.cue_body is None:
            raise RuntimeError("Cue ball is not on the table.")
        self._first_target_hit = None
        impulse = (math.cos(angle) * power, math.sin(angle) * power)
        self.cue_body.apply_impulse_at_world_point(impulse, self.cue_body.position)

    @staticmethod
    def _is_pocketed(pos: pymunk.Vec2d) -> bool:
        x, y = pos.x, pos.y
        if x < 0.0 or x > TABLE_WIDTH or y < 0.0 or y > TABLE_HEIGHT:
            return True
        pr_sq = POCKET_RADIUS * POCKET_RADIUS
        for px, py in POCKETS:
            dx = x - px
            dy = y - py
            if dx * dx + dy * dy < pr_sq:
                return True
        return False

    def _check_pockets(self) -> tuple[list[int], bool]:
        pocketed_targets: list[int] = []
        cue_scratched = False

        if self.cue_body is not None and self._is_pocketed(self.cue_body.position):
            assert self.cue_shape is not None
            self.space.remove(self.cue_body, self.cue_shape)
            self.cue_body = None
            self.cue_shape = None
            cue_scratched = True

        for ball_id in list(self.target_balls):
            body, shape = self.target_balls[ball_id]
            if self._is_pocketed(body.position):
                self.space.remove(body, shape)
                del self.target_balls[ball_id]
                pocketed_targets.append(ball_id)

        return pocketed_targets, cue_scratched

    def _active_bodies(self) -> list[pymunk.Body]:
        bodies = [body for body, _ in self.target_balls.values()]
        if self.cue_body is not None:
            bodies.append(self.cue_body)
        return bodies

    def _all_stopped(self) -> bool:
        thresh_sq = STOP_VELOCITY * STOP_VELOCITY
        return all(b.velocity.length_squared < thresh_sq for b in self._active_bodies())

    def _zero_velocities(self) -> None:
        for body in self._active_bodies():
            body.velocity = (0.0, 0.0)
            body.angular_velocity = 0.0

    def step_until_stopped(
        self,
        max_steps: int = MAX_SIM_STEPS,
        on_step: Callable[[], None] | None = None,
    ) -> ShotResult:
        """Advance physics until all balls stop or `max_steps` is reached."""
        pocketed: list[int] = []
        scratched = False

        for step in range(1, max_steps + 1):
            self.space.step(DT)
            newly_pocketed, cue_in = self._check_pockets()
            if newly_pocketed:
                pocketed.extend(newly_pocketed)
            if cue_in:
                scratched = True
            if on_step is not None:
                on_step()
            if self._all_stopped():
                self._zero_velocities()
                return ShotResult(
                    pocketed_targets=pocketed,
                    cue_hit_target=self._first_target_hit is not None,
                    first_target_hit=self._first_target_hit,
                    cue_scratched=scratched,
                    steps=step,
                    timed_out=False,
                )

        self._zero_velocities()
        return ShotResult(
            pocketed_targets=pocketed,
            cue_hit_target=self._first_target_hit is not None,
            first_target_hit=self._first_target_hit,
            cue_scratched=scratched,
            steps=max_steps,
            timed_out=True,
        )

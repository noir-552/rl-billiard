"""Pygame visualizer for BilliardPhysics (lazy-imported only in human render mode)."""

from __future__ import annotations

import math

import pygame

from src.config import (
    BALL_RADIUS,
    MAX_POWER,
    MIN_POWER,
    POCKET_RADIUS,
    POCKETS,
    TABLE_HEIGHT,
    TABLE_WIDTH,
)
from src.physics import BilliardPhysics

MARGIN = 40
COLOR_FRAME = (92, 54, 28)
COLOR_FELT = (20, 110, 56)
COLOR_RAIL = (14, 82, 42)
COLOR_POCKET = (15, 15, 15)
COLOR_CUE = (245, 245, 245)
COLOR_TEXT = (235, 235, 235)
TARGET_COLORS = (
    (235, 195, 30),  # 1: yellow
    (40, 110, 220),  # 2: blue
    (220, 50, 47),   # 3: red
    (140, 60, 180),  # 4: purple
    (235, 125, 35),  # 5: orange
    (35, 155, 85),   # 6: green
    (145, 35, 45),   # 7: maroon
    (30, 30, 30),    # 8: black
    (245, 205, 45),  # 9: yellow stripe
)


class BilliardRenderer:
    """Renders the 2D billiard table, balls, aim vector, and power indicator."""

    def __init__(self, fps: int = 60, title: str = "RL Billiard") -> None:
        pygame.init()
        self.fps = fps
        self.width = int(TABLE_WIDTH) + 2 * MARGIN
        self.height = int(TABLE_HEIGHT) + 2 * MARGIN
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption(title)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 22)
        self.ball_font = pygame.font.SysFont(None, 16)
        self.closed = False

    def _to_screen(self, x: float, y: float) -> tuple[int, int]:
        return int(round(x + MARGIN)), int(round(y + MARGIN))

    def _pump_events(self) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.closed = True
                return False
        return not self.closed

    def render(
        self,
        physics: BilliardPhysics,
        aim: tuple[float, float] | None = None,
        hud_text: str = "",
    ) -> bool:
        """Draw one frame. Returns False if the user closed the window."""
        if not self._pump_events():
            return False

        # Wood frame & green felt
        self.screen.fill(COLOR_FRAME)
        felt_rect = pygame.Rect(MARGIN, MARGIN, int(TABLE_WIDTH), int(TABLE_HEIGHT))
        pygame.draw.rect(self.screen, COLOR_FELT, felt_rect)
        pygame.draw.rect(self.screen, COLOR_RAIL, felt_rect, 3)

        # 6 black pockets
        for px, py in POCKETS:
            pygame.draw.circle(
                self.screen, COLOR_POCKET, self._to_screen(px, py), int(POCKET_RADIUS)
            )

        # Target balls (numbered 1..9; 9-ball has white stripe ring)
        for bid, (body, _) in physics.target_balls.items():
            pos = self._to_screen(body.position.x, body.position.y)
            color = TARGET_COLORS[bid % len(TARGET_COLORS)]
            if bid == 8:
                pygame.draw.circle(self.screen, COLOR_CUE, pos, int(BALL_RADIUS))
                pygame.draw.circle(self.screen, color, pos, int(BALL_RADIUS * 0.72))
            else:
                pygame.draw.circle(self.screen, color, pos, int(BALL_RADIUS))
            pygame.draw.circle(self.screen, (25, 25, 25), pos, int(BALL_RADIUS), 1)
            label_color = (245, 245, 245) if bid in (1, 3, 6, 7) else (20, 20, 20)
            num_surf = self.ball_font.render(str(bid + 1), True, label_color)
            self.screen.blit(num_surf, num_surf.get_rect(center=pos))

        # Cue ball & optional aim vector + power indicator
        if physics.cue_body is not None:
            cx, cy = physics.cue_body.position.x, physics.cue_body.position.y
            cue_screen = self._to_screen(cx, cy)

            if aim is not None:
                angle, power = aim
                ratio = max(0.0, min(1.0, (power - MIN_POWER) / (MAX_POWER - MIN_POWER)))
                line_len = 40.0 + ratio * 160.0
                end_screen = self._to_screen(
                    cx + math.cos(angle) * line_len,
                    cy + math.sin(angle) * line_len,
                )
                # Green (low power) -> Yellow -> Red (max power)
                aim_color = (int(255 * ratio), int(255 * (1.0 - ratio * 0.7)), 60)
                pygame.draw.line(self.screen, aim_color, cue_screen, end_screen, 2)

                # Power bar in top margin
                bar_x, bar_y, bar_w, bar_h = MARGIN, 12, 160, 14
                pygame.draw.rect(self.screen, (40, 40, 40), (bar_x, bar_y, bar_w, bar_h))
                pygame.draw.rect(
                    self.screen, aim_color, (bar_x, bar_y, int(bar_w * ratio), bar_h)
                )
                pygame.draw.rect(self.screen, COLOR_TEXT, (bar_x, bar_y, bar_w, bar_h), 1)

            pygame.draw.circle(self.screen, COLOR_CUE, cue_screen, int(BALL_RADIUS))
            pygame.draw.circle(self.screen, (40, 40, 40), cue_screen, int(BALL_RADIUS), 1)

        if hud_text:
            surf = self.font.render(hud_text, True, COLOR_TEXT)
            self.screen.blit(surf, (MARGIN + 180, 11))

        pygame.display.flip()
        self.clock.tick(self.fps)
        return True

    def close(self) -> None:
        if not self.closed:
            self.closed = True
        pygame.quit()

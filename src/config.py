"""Constants for the 2D billiard physics engine and RL environment."""

# Table dimensions (2:1 standard ratio)
TABLE_WIDTH = 800.0
TABLE_HEIGHT = 400.0

# Ball properties
BALL_RADIUS = 12.0
BALL_MASS = 1.0
BALL_ELASTICITY = 0.95
BALL_FRICTION = 0.05

# Cushion properties
CUSHION_ELASTICITY = 0.85
CUSHION_RADIUS = 4.0

# Pockets (6 standard positions: 4 corners + 2 side midpoints)
POCKET_RADIUS = 24.0
POCKETS = (
    (0.0, 0.0),
    (TABLE_WIDTH / 2.0, 0.0),
    (TABLE_WIDTH, 0.0),
    (0.0, TABLE_HEIGHT),
    (TABLE_WIDTH / 2.0, TABLE_HEIGHT),
    (TABLE_WIDTH, TABLE_HEIGHT),
)

# Physics simulation
DT = 1.0 / 60.0
TABLE_DAMPING = 0.40  # Velocity fraction retained per second
STOP_VELOCITY = 1.0   # Speed threshold (px/s) below which a ball is considered stopped
MAX_SIM_STEPS = 1000

# Shot power bounds (impulse magnitude; with mass=1.0, equals initial speed in px/s)
MIN_POWER = 200.0
MAX_POWER = 1000.0

# Pymunk collision types
COLLISION_CUE = 1
COLLISION_TARGET = 2
COLLISION_CUSHION = 3

# Reward weights (used in Phase 2+)
REWARD_POCKET = 20.0
REWARD_SCRATCH = -3.0
REWARD_FOUL = -2.0      # Stage 3: wrong first ball hit or miss
REWARD_WIN_RACK = 15.0  # Stage 3: legal 9-ball win bonus
REWARD_POSITION = 1.5   # Stage 3: cue ball alignment bonus for next target after legal pocket
REWARD_FIRST_HIT = 0.1  # Kept well below |REWARD_STEP| to prevent hit-farming local optimum
REWARD_PROGRESS = 0.3   # Symmetric (potential-based) progress shaping
REWARD_STEP = -0.5      # Strong enough that stalling > scratch risk (avoids soft-tap local minimum)

# Curriculum stages (stage 1: 1 ball, stage 2: 3 balls, stage 3: 9-ball rules)
STAGE_TARGET_COUNTS = {1: 1, 2: 3, 3: 9}
MAX_TARGET_BALLS = max(STAGE_TARGET_COUNTS.values())
MAX_SHOTS_PER_TARGET = 15
STAGE3_MAX_SHOTS = 40



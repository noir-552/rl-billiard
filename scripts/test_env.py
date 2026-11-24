"""Phase 2 & 3 Verification Gate: SB3 check_env, headless FPS benchmark, and Pygame render check."""

import argparse
import time
import warnings

from stable_baselines3.common.env_checker import check_env

from src.env import BilliardEnv


def verify_env(stage: int) -> None:
    env = BilliardEnv(stage=stage)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        check_env(env, warn=True)
    if caught:
        for w in caught:
            print(f"[WARNING stage={stage}] {w.category.__name__}: {w.message}")
        raise AssertionError(f"check_env(stage={stage}) emitted {len(caught)} warning(s)")
    print(f"[check_env] stage={stage} PASSED (0 warnings, obs_shape={env.observation_space.shape})")
    env.close()


def benchmark_random_agent(stage: int = 1, n_shots: int = 100) -> None:
    env = BilliardEnv(stage=stage)
    env.reset(seed=42)
    total_reward = 0.0
    pockets = 0
    scratches = 0
    episodes = 0

    t0 = time.perf_counter()
    for _ in range(n_shots):
        action = env.action_space.sample()
        _, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        pockets += len(info["pocketed"])
        scratches += int(info["scratched"])
        if terminated or truncated:
            episodes += 1
            env.reset()
    elapsed = time.perf_counter() - t0
    sps = n_shots / elapsed

    print(
        f"[benchmark] stage={stage} shots={n_shots} time={elapsed:.3f}s "
        f"speed={sps:.1f} shots/s episodes={episodes} pockets={pockets} "
        f"scratches={scratches} total_reward={total_reward:.2f}"
    )
    env.close()


def run_render_demo(stage: int = 2, n_shots: int = 10) -> None:
    env = BilliardEnv(stage=stage, render_mode="human")
    env.reset(seed=7)
    env.render()
    for shot in range(1, n_shots + 1):
        action = env.action_space.sample()
        _, reward, terminated, truncated, info = env.step(action)
        print(
            f"[render shot={shot}] reward={reward:+.2f} "
            f"hit={info['cue_hit_target']} pocketed={info['pocketed']} "
            f"scratch={info['scratched']} left={info['remaining_targets']}"
        )
        if env._renderer is not None and env._renderer.closed:
            print("[render] Window closed by user.")
            break
        if terminated or truncated:
            env.reset()
            env.render()
    env.close()
    print("[render] Pygame visual check completed cleanly.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify BilliardEnv (headless or Pygame render).")
    parser.add_argument("--render", action="store_true", help="Watch random agent in Pygame.")
    parser.add_argument("--stage", type=int, default=3, choices=[1, 2, 3], help="Stage for --render.")
    parser.add_argument("--shots", type=int, default=8, help="Number of random shots for --render.")
    args = parser.parse_args()

    if args.render:
        run_render_demo(stage=args.stage, n_shots=args.shots)
    else:
        verify_env(stage=1)
        verify_env(stage=2)
        verify_env(stage=3)
        benchmark_random_agent(stage=1, n_shots=100)
        benchmark_random_agent(stage=2, n_shots=100)
        benchmark_random_agent(stage=3, n_shots=100)
        print("ALL ENV CHECKS (STAGES 1-3) PASSED.")

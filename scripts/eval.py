"""Phase 5: Evaluate trained SAC model(s), compare checkpoints, or watch live Pygame replay."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3 import SAC

from src.env import BilliardEnv


def evaluate(
    model_path: str,
    stage: int = 1,
    episodes: int = 50,
    render: bool = False,
    seed: int = 100,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run deterministic evaluation and return metrics dict."""
    model = SAC.load(model_path)
    env = BilliardEnv(stage=stage, render_mode="human" if render else None)

    total_targets = episodes * env.num_targets
    total_pocketed = 0
    total_shots = 0
    total_scratches = 0
    cleared_shots: list[int] = []
    episode_rewards: list[float] = []

    try:
        for ep in range(episodes):
            obs, _ = env.reset(seed=seed + ep)
            if render:
                env.render()
            ep_reward = 0.0
            done = False
            info: dict[str, Any] = {}
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                ep_reward += reward
                total_shots += 1
                total_pocketed += len(info["pocketed"])
                total_scratches += int(info["scratched"])
                done = terminated or truncated

                if render and env._renderer is not None and env._renderer.closed:
                    if verbose:
                        print("[eval] Window closed by user.")
                    break
            else:
                episode_rewards.append(ep_reward)
                if info.get("cleared"):
                    cleared_shots.append(info["shots"])
                continue
            break
    finally:
        env.close()

    n_done = len(episode_rewards)
    denom_targets = n_done * env.num_targets
    metrics = {
        "model": str(model_path),
        "stage": stage,
        "episodes": n_done,
        "pocket_rate": round((total_pocketed / denom_targets) * 100.0, 2) if denom_targets else 0.0,
        "clear_rate": round((len(cleared_shots) / n_done) * 100.0, 2) if n_done else 0.0,
        "avg_shots_to_clear": round(float(np.mean(cleared_shots)), 2) if cleared_shots else None,
        "scratch_rate": round((total_scratches / total_shots) * 100.0, 2) if total_shots else 0.0,
        "mean_reward": round(float(np.mean(episode_rewards)), 2) if episode_rewards else 0.0,
        "pocketed": total_pocketed,
        "total_targets": denom_targets,
        "cleared": len(cleared_shots),
        "scratches": total_scratches,
        "total_shots": total_shots,
    }

    if verbose:
        avg_str = f"{metrics['avg_shots_to_clear']:.2f}" if metrics["avg_shots_to_clear"] is not None else "N/A"
        print(f"=== Evaluation ({model_path} | stage={stage} | episodes={n_done}) ===")
        print(f"Pocket Success Rate : {metrics['pocket_rate']:.1f}% ({total_pocketed}/{denom_targets} balls)")
        print(f"Table Clear Rate    : {metrics['clear_rate']:.1f}% ({len(cleared_shots)}/{n_done} episodes)")
        print(f"Avg Shots to Clear  : {avg_str}")
        print(f"Scratch Rate        : {metrics['scratch_rate']:.1f}% ({total_scratches}/{total_shots} shots)")
        print(f"Mean Episode Reward : {metrics['mean_reward']:+.2f}")

    return metrics


def print_comparison_table(results: list[tuple[str, dict[str, Any]]]) -> None:
    """Print a clean side-by-side comparison table for multiple evaluated models."""
    header = f"{'Model':<16} | {'Pocket%':>8} | {'Clear%':>7} | {'AvgShots':>8} | {'Scratch%':>8} | {'MeanRew':>8}"
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for label, m in results:
        avg_s = f"{m['avg_shots_to_clear']:.2f}" if m["avg_shots_to_clear"] is not None else "N/A"
        print(
            f"{label:<16} | {m['pocket_rate']:>7.1f}% | {m['clear_rate']:>6.1f}% | "
            f"{avg_s:>8} | {m['scratch_rate']:>7.1f}% | {m['mean_reward']:>+8.2f}"
        )
    print(sep)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate or compare SAC billiard agents.")
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--model", type=str, default=None, help="Path to model .zip (default: logs/stage{stage}/best_model.zip)")
    parser.add_argument("--compare", action="store_true", help="Compare best_model.zip vs final_model.zip on identical seeds.")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--render", action="store_true", help="Watch live gameplay in Pygame.")
    parser.add_argument("--seed", type=int, default=100)
    args = parser.parse_args()

    if args.compare:
        stage_dir = Path(f"logs/stage{args.stage}")
        candidates = [
            ("best_model", stage_dir / "best_model.zip"),
            ("final_model", stage_dir / "final_model.zip"),
        ]
        results: list[tuple[str, dict[str, Any]]] = []
        for label, path in candidates:
            if path.exists():
                m = evaluate(str(path), stage=args.stage, episodes=args.episodes, seed=args.seed, verbose=False)
                results.append((label, m))
            else:
                print(f"[compare] Skipping missing {path}")
        if results:
            print(f"=== Head-to-Head Comparison (stage={args.stage}, episodes={args.episodes}, seed={args.seed}) ===")
            print_comparison_table(results)
    else:
        model_path = args.model or f"logs/stage{args.stage}/best_model.zip"
        evaluate(
            model_path=model_path,
            stage=args.stage,
            episodes=args.episodes,
            render=args.render,
            seed=args.seed,
        )

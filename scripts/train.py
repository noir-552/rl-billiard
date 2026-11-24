"""Phase 4: SAC training pipeline with SubprocVecEnv, EvalCallback, and automatic winner selection."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from datetime import datetime
from pathlib import Path
from typing import Callable

import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from scripts.eval import evaluate, print_comparison_table
from src.env import BilliardEnv


def make_env(stage: int, seed: int, rank: int) -> Callable[[], Monitor]:
    def _init() -> Monitor:
        env = BilliardEnv(stage=stage)
        env.reset(seed=seed + rank)
        return Monitor(env)

    return _init


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SAC agent on BilliardEnv.")
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--timesteps", type=int, default=20_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--load-model", type=str, default=None, help="Path to .zip model for transfer learning.")
    parser.add_argument("--log-dir", type=str, default="./logs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-episodes", type=int, default=50, help="Final head-to-head eval episodes.")
    args = parser.parse_args()

    log_root = Path(args.log_dir)
    tb_dir = log_root / "tb"
    save_dir = log_root / f"stage{args.stage}"
    save_dir.mkdir(parents=True, exist_ok=True)

    train_env = SubprocVecEnv([make_env(args.stage, args.seed, i) for i in range(args.n_envs)])
    eval_env = DummyVecEnv([make_env(args.stage, args.seed + 1000, 0)])

    eval_freq = max(1000, args.timesteps // (5 * args.n_envs))
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(save_dir),
        log_path=str(save_dir),
        eval_freq=eval_freq,
        n_eval_episodes=30,
        deterministic=True,
        verbose=1,
    )

    lr_schedule = lambda progress: 1e-4 + (3e-4 - 1e-4) * progress

    if args.load_model:
        print(f"[train] Loading model from {args.load_model} for Stage {args.stage}...")
        model = SAC.load(
            args.load_model,
            env=train_env,
            tensorboard_log=str(tb_dir),
            verbose=1,
            custom_objects={
                "learning_rate": lr_schedule,
                "lr_schedule": lr_schedule,
            },
        )
        # Re-open exploration on stage transfer so SAC doesn't stay trapped in prior stage's local minimum
        model.learning_starts = 2_000
        model.gradient_steps = args.n_envs
        if model.log_ent_coef is not None:
            with torch.no_grad():
                model.log_ent_coef.fill_(math.log(0.15))
            if model.ent_coef_optimizer is not None:
                model.ent_coef_optimizer.state.clear()
    else:
        model = SAC(
            "MlpPolicy",
            train_env,
            learning_rate=lr_schedule,
            buffer_size=200_000,
            learning_starts=2_000,
            batch_size=256,
            gradient_steps=args.n_envs,
            gamma=0.95,
            ent_coef="auto_0.2",
            verbose=1,
            tensorboard_log=str(tb_dir),
            seed=args.seed,
        )

    final_zip = save_dir / "final_model.zip"
    best_zip = save_dir / "best_model.zip"

    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=eval_callback,
            tb_log_name=f"SAC_stage{args.stage}",
            reset_num_timesteps=True,
        )
        model.save(str(final_zip))
        print(f"[train] Saved final model to {final_zip}")
    finally:
        train_env.close()
        eval_env.close()

    # Fair head-to-head 50-episode evaluation on identical seeds -> promote true winner to best_model.zip
    eval_seed = args.seed + 5000
    final_metrics = evaluate(
        str(final_zip), stage=args.stage, episodes=args.eval_episodes, seed=eval_seed, verbose=False
    )
    rows: list[tuple[str, dict]] = []
    best_metrics = None
    if best_zip.exists():
        best_metrics = evaluate(
            str(best_zip), stage=args.stage, episodes=args.eval_episodes, seed=eval_seed, verbose=False
        )
        rows.append(("best_checkpoint", best_metrics))
    rows.append(("final_model", final_metrics))

    print(f"\n[train] Head-to-head evaluation ({args.eval_episodes} episodes, seed={eval_seed}):")
    print_comparison_table(rows)

    if best_metrics is None or final_metrics["mean_reward"] > best_metrics["mean_reward"]:
        shutil.copy2(final_zip, best_zip)
        winner = "final_model"
        print(f"[train] Winner: final_model (promoted -> {best_zip})")
    else:
        winner = "best_checkpoint"
        print(f"[train] Winner: best_checkpoint (kept at {best_zip})")

    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "stage": args.stage,
        "timesteps": args.timesteps,
        "n_envs": args.n_envs,
        "seed": args.seed,
        "loaded_from": args.load_model,
        "eval_episodes": args.eval_episodes,
        "winner": winner,
        "best_checkpoint_metrics": best_metrics,
        "final_model_metrics": final_metrics,
    }
    summary_path = save_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[train] Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()

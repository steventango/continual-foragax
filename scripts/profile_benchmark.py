import argparse
import glob
import gzip
import json
import os
import time

import jax
import pandas as pd

from foragax import registry
from foragax.env import Actions


def profile_environment(
    env_name, num_envs=128, steps=1000, trace_dir="/tmp/jax_trace", fast=True
):
    print(f"Structuring benchmark for {env_name}...")
    env = registry.make(env_name)
    params = env.default_params

    # Init
    key = jax.random.key(0)
    key, reset_key = jax.random.split(key)
    reset_keys = jax.random.split(reset_key, num_envs)
    states = jax.vmap(env.reset, in_axes=(0, None))(reset_keys, params)[1]

    if fast:
        print("Using jax.lax.scan (fast, less granular logic trace)")

        @jax.jit
        def _run(states, key):
            def f(carry, _):
                states, key = carry
                key, step_key = jax.random.split(key, 2)
                step_keys = jax.random.split(step_key, num_envs)
                _, new_states, _, _, _ = jax.vmap(env.step, in_axes=(0, 0, None, None))(
                    step_keys, states, Actions.DOWN, params
                )
                return (new_states, key), None

            (final_state, _), _ = jax.lax.scan(f, (states, key), None, length=steps)
            return final_state

        # Warmup
        print("Warming up...")
        key, run_key = jax.random.split(key)
        _run(states, run_key).pos.block_until_ready()
        print("Warmup done.")
    else:
        print("Using Python loop (slower, detailed logic trace)")

        # Define single step (no outer jit)
        def _step_fn(states, key):
            key, step_key = jax.random.split(key, 2)
            step_keys = jax.random.split(step_key, num_envs)
            _, new_states, _, _, _ = jax.vmap(env.step_env, in_axes=(0, 0, None, None))(
                step_keys, states, Actions.DOWN, params
            )
            return new_states, key

        # Warmup
        print("Warming up...")
        key, run_key = jax.random.split(key)
        _step_fn(states, run_key).pos.block_until_ready()
        print("Warmup done.")

        def _run(states, key):
            # Manual loop
            for _ in range(steps):
                states, key = _step_fn(states, key)
            return states

    # Profiling
    print(f"Starting JAX trace. Output dir: {trace_dir}")
    try:
        with jax.profiler.trace(trace_dir):
            start_time = time.time()
            key, run_key = jax.random.split(key)
            _run(states, run_key).pos.block_until_ready()
            end_time = time.time()
    except Exception as e:
        print(f"Profiling failed: {e}")
        end_time = time.time()

    print(f"Trace stopped. Total time: {end_time - start_time:.4f}s")

    # Debug: Check for files
    if os.path.exists(trace_dir):
        print(f"Files in {trace_dir}: {os.listdir(trace_dir)}")
    else:
        print(f"Directory {trace_dir} does not exist.")

    total_steps = num_envs * steps
    fps = total_steps / (end_time - start_time)
    print(f"FPS: {fps:.2f}")

    return trace_dir


def parse_trace(trace_dir):
    # Find latest trace file
    files = glob.glob(os.path.join(trace_dir, "**", "*.json.gz"), recursive=True)
    if not files:
        print("No trace files found.")
        return

    latest_file = max(files, key=os.path.getctime)
    print(f"Parsing latest trace: {latest_file}")

    with gzip.open(latest_file, "rb") as f:
        trace_data = json.load(f)

    events = trace_data.get("traceEvents", [])

    # Sort events by timestamp to ensure B comes before E
    # Events might not be sorted in the file
    events.sort(key=lambda x: x.get("ts", 0))

    scope_stats = {}
    thread_stacks = {}  # (pid, tid) -> dict of open scopes

    for event in events:
        name = event.get("name", "unknown")
        ph = event.get("ph")
        pid = event.get("pid")
        tid = event.get("tid")
        ts = event.get("ts", 0)

        # Complete events (X) have duration
        if ph == "X":
            dur = event.get("dur", 0)
            dur_ms = dur / 1000.0

            if name not in scope_stats:
                scope_stats[name] = {"total_ms": 0.0, "count": 0}
            scope_stats[name]["total_ms"] += dur_ms
            scope_stats[name]["count"] += 1

        # Begin events (B) push to stack
        elif ph == "B":
            key = (pid, tid)
            if key not in thread_stacks:
                thread_stacks[key] = []
            thread_stacks[key].append((name, ts))

        # End events (E) pop from stack
        elif ph == "E":
            key = (pid, tid)
            if key in thread_stacks and thread_stacks[key]:
                # We assume strict nesting (LIFO)
                # JAX traces should be well-formed
                start_name, start_ts = thread_stacks[key].pop()

                # If names don't match, it might be an issue, but usually E events don't carry name
                # or carry the same name. We rely on stack.
                # Use start_name as the source of truth for the scope

                dur_ms = (ts - start_ts) / 1000.0
                if start_name not in scope_stats:
                    scope_stats[start_name] = {"total_ms": 0.0, "count": 0}
                scope_stats[start_name]["total_ms"] += dur_ms
                scope_stats[start_name]["count"] += 1

    # Convert to DataFrame
    df_data = []
    for name, stats in scope_stats.items():
        df_data.append(
            {
                "Scope Name": name,
                "Total (ms)": stats["total_ms"],
                "Count": stats["count"],
                "Mean (ms)": stats["total_ms"] / stats["count"]
                if stats["count"] > 0
                else 0,
            }
        )

    df = pd.DataFrame(df_data)
    if not df.empty:
        df = df.sort_values(by="Total (ms)", ascending=False)
        print("\n=== Profiling Results (Top 20 Scopes) ===")
        print(df.head(20).to_string(index=False))

        # Check for our specific named scopes
        # Since JAX names might be mangled/nested, we search for partial matches
        target_scopes = [
            "move_agent",
            "compute_reward",
            "respawn_logic",
            "expire_objects",
            "dynamic_biomes",
            "observation",
            "update_state",
            "reward_grid",
        ]
        print("\n=== Target Environment Scopes ===")

        # Create a more readable view for targets
        found_targets = []
        for target in target_scopes:
            mask = df["Scope Name"].str.contains(target)
            if mask.any():
                subset = df[mask]
                # Summing might be double counting if there are multiple events per call (e.g. across batch)
                # or separate XLA ops. But usually named_scope wraps the whole block.
                # Let's show the matched rows.
                found_targets.append(subset)

        if found_targets:
            target_df = (
                pd.concat(found_targets)
                .drop_duplicates()
                .sort_values(by="Total (ms)", ascending=False)
            )
            print(target_df.to_string(index=False))
        else:
            print("No target scopes found in trace.")
            print("DEBUG: All unique scope names found:")
            print(df["Scope Name"].unique()[:50])

    else:
        print("No duration events found in trace.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="ForagaxDiwali-v5")
    parser.add_argument("--num_envs", type=int, default=128)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--trace_dir", type=str, default="/tmp/jax_trace")
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use scan (faster) but less detailed trace",
    )
    args = parser.parse_args()

    if not os.path.exists(args.trace_dir):
        os.makedirs(args.trace_dir, exist_ok=True)

    os.environ["TF_PROFILER_TRACE_VIEWER_MAX_EVENTS"] = "10000000"

    profile_environment(
        args.env, args.num_envs, args.steps, args.trace_dir, fast=args.fast
    )
    parse_trace(args.trace_dir)

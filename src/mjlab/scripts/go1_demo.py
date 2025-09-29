"""Script to run a tracking demo with a pretrained policy.

This demo downloads a pretrained checkpoint and motion file from cloud storage
and launches an interactive viewer with a humanoid robot performing a cartwheel.
"""

from functools import partial

import tyro

from mjlab.scripts.gcs import ensure_asset_downloaded
from mjlab.scripts.play import run_play


def main() -> None:
  """Run demo with pretrained velocity policy."""
  print("🎮 Setting up MJLab demo with pretrained velocity policy...")

  try:
    path = ensure_asset_downloaded("go1_demo_ckpt.pt")
    checkpoint_path = str(path.resolve())
  except RuntimeError as e:
    print(f"❌ Failed to download demo assets: {e}")
    print("Please check your internet connection and try again.")
    return

  tyro.cli(
    partial(
      run_play,
      task="Mjlab-Velocity-Flat-Unitree-Go1-Play",
      checkpoint_file=checkpoint_path,
      num_envs=32,
      render_all_envs=True,
      viewer="viser",
    )
  )


if __name__ == "__main__":
  main()

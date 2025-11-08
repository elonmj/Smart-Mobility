"""
Custom Callbacks for RL Training with Checkpoint Rotation

Implements intelligent checkpoint management with automatic cleanup
to prevent disk space issues on Kaggle.
"""

import os
import glob
from pathlib import Path
from typing import Optional
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback


class RotatingCheckpointCallback(CheckpointCallback):
    """
    Checkpoint callback with automatic rotation to save disk space.
    
    Keeps only the N most recent checkpoints, automatically deleting older ones.
    This is critical for Kaggle environments with limited storage (20GB).
    
    Args:
        save_freq: Frequency (in timesteps) to save checkpoints
        save_path: Directory to save checkpoints
        name_prefix: Prefix for checkpoint filenames
        max_checkpoints: Maximum number of checkpoints to keep (default: 3)
        save_replay_buffer: Whether to save the replay buffer (important for DQN/SAC)
        verbose: Verbosity level
    
    Example:
        >>> callback = RotatingCheckpointCallback(
        ...     save_freq=1000,
        ...     save_path="./checkpoints",
        ...     name_prefix="rl_model",
        ...     max_checkpoints=2  # Keep only 2 most recent
        ... )
    """
    
    def __init__(
        self,
        save_freq: int,
        save_path: str,
        name_prefix: str = "rl_model",
        max_checkpoints: int = 3,
        save_replay_buffer: bool = True,
        save_vecnormalize: bool = True,
        verbose: int = 1,
    ):
        super().__init__(
            save_freq=save_freq,
            save_path=save_path,
            name_prefix=name_prefix,
            save_replay_buffer=save_replay_buffer,
            save_vecnormalize=save_vecnormalize,
            verbose=verbose,
        )
        self.max_checkpoints = max_checkpoints
        
    def _on_step(self) -> bool:
        """
        Called at each environment step.
        Saves checkpoint and performs rotation if needed.
        """
        # Call parent to handle checkpoint saving
        continue_training = super()._on_step()
        
        # After saving, perform rotation
        if self.n_calls % self.save_freq == 0:
            self._rotate_checkpoints()
        
        return continue_training
    
    def _rotate_checkpoints(self):
        """
        Remove old checkpoints, keeping only max_checkpoints most recent ones.
        
        Handles both .zip files and associated replay buffer files.
        """
        # Find all checkpoint files
        checkpoint_pattern = os.path.join(self.save_path, f"{self.name_prefix}_*_steps.zip")
        checkpoint_files = glob.glob(checkpoint_pattern)
        
        if len(checkpoint_files) <= self.max_checkpoints:
            return  # Nothing to delete
        
        # Sort by modification time (oldest first)
        checkpoint_files.sort(key=os.path.getmtime)
        
        # Calculate how many to delete
        num_to_delete = len(checkpoint_files) - self.max_checkpoints
        files_to_delete = checkpoint_files[:num_to_delete]
        
        # Delete old checkpoints
        for filepath in files_to_delete:
            try:
                os.remove(filepath)
                if self.verbose > 0:
                    print(f"🗑️  Deleted old checkpoint: {os.path.basename(filepath)}")
                
                # Also delete associated replay buffer if it exists
                replay_buffer_path = filepath.replace('.zip', '_replay_buffer.pkl')
                if os.path.exists(replay_buffer_path):
                    os.remove(replay_buffer_path)
                    if self.verbose > 0:
                        print(f"🗑️  Deleted replay buffer: {os.path.basename(replay_buffer_path)}")
                        
            except OSError as e:
                if self.verbose > 0:
                    print(f"⚠️  Warning: Could not delete {filepath}: {e}")


class TrainingProgressCallback(BaseCallback):
    """
    Callback to log detailed training progress with time estimates.
    
    Useful for monitoring long training runs on Kaggle.
    """
    
    def __init__(self, total_timesteps: int, log_freq: int = 1000, verbose: int = 1):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.log_freq = log_freq
        self.start_time = None
        
    def _on_training_start(self):
        """Called at the beginning of training."""
        import time
        self.start_time = time.time()
        if self.verbose > 0:
            print(f"\n Training started: {self.total_timesteps:,} total timesteps")
    
    def _on_step(self) -> bool:
        """Called at each step."""
        if self.n_calls % self.log_freq == 0:
            import time
            elapsed = time.time() - self.start_time
            steps_done = self.n_calls
            steps_remaining = self.total_timesteps - steps_done
            
            if steps_done > 0:
                time_per_step = elapsed / steps_done
                eta_seconds = time_per_step * steps_remaining
                eta_minutes = eta_seconds / 60
                
                progress_pct = (steps_done / self.total_timesteps) * 100
                
                if self.verbose > 0:
                    print(f"📊 Progress: {steps_done:,}/{self.total_timesteps:,} ({progress_pct:.1f}%) | "
                          f"ETA: {eta_minutes:.1f} min | "
                          f"Speed: {1/time_per_step:.1f} steps/s")
        
        return True  # Continue training
    
    def _on_training_end(self):
        """Called at the end of training."""
        import time
        total_time = time.time() - self.start_time
        if self.verbose > 0:
            print(f"\n✅ Training completed in {total_time/60:.1f} minutes ({total_time:.1f}s)")


class EarlyStoppingCallback(BaseCallback):
    """
    Stop training when performance plateaus or degrades.
    
    Useful for preventing overfitting and saving GPU time on Kaggle.
    """
    
    def __init__(
        self,
        reward_threshold: Optional[float] = None,
        patience: int = 10,
        min_delta: float = 0.0,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.reward_threshold = reward_threshold
        self.patience = patience
        self.min_delta = min_delta
        self.best_mean_reward = -float('inf')
        self.no_improvement_count = 0
        
    def _on_step(self) -> bool:
        """Check if training should stop."""
        # This would need to be integrated with evaluation results
        # For now, this is a placeholder for the pattern
        return True

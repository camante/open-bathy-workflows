#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
checkpoints.py - Pipeline Checkpoint and Resume Capability

This module enables the pipeline to save progress at major stages and resume
from the last successful checkpoint after a failure.

Features:
- Automatic checkpoint saving at configurable stages
- Config hash validation (invalidates checkpoints if config changes)
- Artifact tracking (paths to outputs from each stage)
- Optional checkpoint compression
- Thread-safe checkpoint updates

Usage:
    from checkpoints import PipelineCheckpoint, CheckpointStage
    
    # Initialize checkpoint manager
    checkpoint = PipelineCheckpoint(
        output_dir=Path("./output"),
        config={"aoi": "...", "start_date": "..."}
    )
    
    # Check if stage can be skipped
    if checkpoint.should_skip(CheckpointStage.S2_FETCH):
        s2_result = checkpoint.get_artifact(CheckpointStage.S2_FETCH, "rgb_path")
    else:
        s2_result = fetch_sentinel2(cfg)
        checkpoint.mark_complete(
            CheckpointStage.S2_FETCH,
            artifacts={"rgb_path": str(s2_result)}
        )
"""

import hashlib
import json
import logging
import os
import shutil
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

log = logging.getLogger(__name__)


class CheckpointStage(Enum):
    """Pipeline stages that support checkpointing."""
    
    # SDB Pipeline Stages
    S2_FETCH = "s2_fetch"
    ATL_FETCH = "atl_fetch"
    FUSION = "fusion"
    TRAIN = "train"
    PREDICT = "predict"
    ALIGN = "align"
    
    # River Pipeline Stages
    RIVER_NETWORK = "river_network"
    XS_BUILD = "xs_build"
    XS_INFER = "xs_infer"
    RIVER_RASTER = "river_raster"
    
    # Combined Pipeline Stages
    BATHY_FUSION = "bathy_fusion"
    FINAL_OUTPUT = "final_output"
    
    @classmethod
    def from_string(cls, s: str) -> "CheckpointStage":
        """Convert string to CheckpointStage."""
        s = s.lower().strip()
        for stage in cls:
            if stage.value == s:
                return stage
        raise ValueError(f"Unknown checkpoint stage: {s}")
    
    @classmethod
    def all_stages(cls) -> List["CheckpointStage"]:
        """Return all stages in execution order."""
        return [
            cls.S2_FETCH, cls.ATL_FETCH, cls.FUSION, cls.TRAIN,
            cls.PREDICT, cls.ALIGN, cls.RIVER_NETWORK, cls.XS_BUILD,
            cls.XS_INFER, cls.RIVER_RASTER, cls.BATHY_FUSION, cls.FINAL_OUTPUT
        ]


@dataclass
class StageCheckpoint:
    """Checkpoint data for a single stage."""
    stage: str
    completed: bool = False
    timestamp: Optional[str] = None
    duration_seconds: Optional[float] = None
    artifacts: Dict[str, str] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass 
class PipelineState:
    """Complete pipeline checkpoint state."""
    config_hash: str
    pipeline_version: str
    created: str
    last_updated: str
    stages: Dict[str, StageCheckpoint] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "config_hash": self.config_hash,
            "pipeline_version": self.pipeline_version,
            "created": self.created,
            "last_updated": self.last_updated,
            "stages": {k: asdict(v) for k, v in self.stages.items()},
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PipelineState":
        """Create from dictionary."""
        stages = {}
        for k, v in d.get("stages", {}).items():
            stages[k] = StageCheckpoint(**v)
        
        return cls(
            config_hash=d["config_hash"],
            pipeline_version=d.get("pipeline_version", "unknown"),
            created=d["created"],
            last_updated=d["last_updated"],
            stages=stages,
            metadata=d.get("metadata", {})
        )


class PipelineCheckpoint:
    """
    Manages pipeline checkpoints for resume capability.
    
    Checkpoints are saved to a .checkpoints directory within the output
    directory. Each checkpoint includes:
    - Config hash (to invalidate on config changes)
    - Completed stages
    - Artifact paths from each stage
    - Timing information
    
    Thread-safe for concurrent access.
    """
    
    CHECKPOINT_DIR_NAME = ".checkpoints"
    STATE_FILE_NAME = "pipeline_state.json"
    
    # Try to import pipeline version
    try:
        from constants import PIPELINE_VERSION
    except ImportError:
        PIPELINE_VERSION = "sdb_river_unified_v0.7.6"
    
    def __init__(
        self,
        output_dir: Path,
        config: Dict[str, Any],
        enabled: bool = True,
        auto_save: bool = True
    ):
        """
        Initialize checkpoint manager.
        
        Args:
            output_dir: Pipeline output directory
            config: Pipeline configuration dictionary
            enabled: Whether checkpointing is enabled
            auto_save: Automatically save state after each update
        """
        self.output_dir = Path(output_dir)
        self.config = config
        self.enabled = enabled
        self.auto_save = auto_save
        self._lock = threading.RLock()
        
        # Compute config hash for invalidation
        self.config_hash = self._compute_config_hash(config)
        
        # Setup checkpoint directory
        self.checkpoint_dir = self.output_dir / self.CHECKPOINT_DIR_NAME
        self.state_file = self.checkpoint_dir / self.STATE_FILE_NAME
        
        # Load or create state
        self._state: Optional[PipelineState] = None
        if enabled:
            self._ensure_checkpoint_dir()
            self._load_or_create_state()
    
    def _compute_config_hash(self, config: Dict[str, Any]) -> str:
        """
        Compute a stable hash of the configuration.
        
        Only includes keys that affect pipeline outputs.
        """
        # Keys that should trigger checkpoint invalidation if changed
        hash_keys = [
            "aoi", "start_date", "end_date", "start", "end",
            "max_depth_sdb", "methods", "priority",
            "xs_spacing_m", "xs_length_m",
            "fusion_strategy", "align_mode"
        ]
        
        # Build hashable config subset
        hash_config = {}
        for key in hash_keys:
            if key in config:
                val = config[key]
                # Convert lists/dicts to sorted tuples for stable hashing
                if isinstance(val, list):
                    val = tuple(sorted(str(v) for v in val))
                elif isinstance(val, dict):
                    val = tuple(sorted((str(k), str(v)) for k, v in val.items()))
                hash_config[key] = str(val)
        
        # Compute hash
        config_str = json.dumps(hash_config, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]
    
    def _ensure_checkpoint_dir(self) -> None:
        """Create checkpoint directory if it doesn't exist."""
        try:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            log.warning(f"[CHECKPOINT] Could not create checkpoint directory: {e}")
            self.enabled = False
    
    def _load_or_create_state(self) -> None:
        """Load existing state or create new one."""
        with self._lock:
            if self.state_file.exists():
                try:
                    with open(self.state_file, "r") as f:
                        data = json.load(f)
                    
                    loaded_state = PipelineState.from_dict(data)
                    
                    # Check if config hash matches
                    if loaded_state.config_hash == self.config_hash:
                        self._state = loaded_state
                        n_completed = sum(
                            1 for s in self._state.stages.values() if s.completed
                        )
                        log.info(
                            f"[CHECKPOINT] Loaded existing checkpoint: "
                            f"{n_completed} stages completed"
                        )
                    else:
                        log.info(
                            "[CHECKPOINT] Config changed, invalidating existing checkpoints"
                        )
                        self._state = self._create_new_state()
                        self._save_state()
                        
                except (json.JSONDecodeError, KeyError, TypeError) as e:
                    log.warning(f"[CHECKPOINT] Could not load state, creating new: {e}")
                    self._state = self._create_new_state()
            else:
                self._state = self._create_new_state()
                self._save_state()
    
    def _create_new_state(self) -> PipelineState:
        """Create a fresh pipeline state."""
        now = datetime.now().isoformat()
        return PipelineState(
            config_hash=self.config_hash,
            pipeline_version=self.PIPELINE_VERSION,
            created=now,
            last_updated=now,
            stages={},
            metadata={}
        )
    
    def _save_state(self) -> None:
        """Save current state to disk."""
        if not self.enabled or self._state is None:
            return
        
        with self._lock:
            try:
                self._state.last_updated = datetime.now().isoformat()
                
                # Write to temp file first, then rename (atomic on most filesystems)
                temp_file = self.state_file.with_suffix(".tmp")
                with open(temp_file, "w") as f:
                    json.dump(self._state.to_dict(), f, indent=2)
                
                # Atomic rename
                temp_file.rename(self.state_file)
                
            except Exception as e:
                log.warning(f"[CHECKPOINT] Could not save state: {e}")
    
    def should_skip(self, stage: CheckpointStage) -> bool:
        """
        Check if a stage can be skipped (already completed).
        
        Args:
            stage: Stage to check
            
        Returns:
            True if stage is complete and can be skipped
        """
        if not self.enabled or self._state is None:
            return False
        
        with self._lock:
            stage_key = stage.value
            if stage_key not in self._state.stages:
                return False
            
            stage_data = self._state.stages[stage_key]
            if not stage_data.completed:
                return False
            
            # Verify artifacts still exist
            for name, path in stage_data.artifacts.items():
                if path and not Path(path).exists():
                    log.info(
                        f"[CHECKPOINT] Stage {stage_key} artifact missing: {name}={path}"
                    )
                    return False
            
            log.info(f"[CHECKPOINT] Skipping completed stage: {stage_key}")
            return True
    
    def mark_complete(
        self,
        stage: CheckpointStage,
        artifacts: Dict[str, str] = None,
        metrics: Dict[str, Any] = None,
        duration_seconds: float = None
    ) -> None:
        """
        Mark a stage as complete.
        
        Args:
            stage: Stage that completed
            artifacts: Dictionary of artifact name -> path
            metrics: Optional metrics from the stage
            duration_seconds: Time taken for the stage
        """
        if not self.enabled or self._state is None:
            return
        
        with self._lock:
            stage_key = stage.value
            
            checkpoint = StageCheckpoint(
                stage=stage_key,
                completed=True,
                timestamp=datetime.now().isoformat(),
                duration_seconds=duration_seconds,
                artifacts=artifacts or {},
                metrics=metrics or {},
                error=None
            )
            
            self._state.stages[stage_key] = checkpoint
            
            log.info(
                f"[CHECKPOINT] Stage complete: {stage_key} "
                f"(artifacts: {list(checkpoint.artifacts.keys())})"
            )
            
            if self.auto_save:
                self._save_state()
    
    def mark_failed(
        self,
        stage: CheckpointStage,
        error: str,
        duration_seconds: float = None
    ) -> None:
        """
        Mark a stage as failed.
        
        Args:
            stage: Stage that failed
            error: Error message
            duration_seconds: Time before failure
        """
        if not self.enabled or self._state is None:
            return
        
        with self._lock:
            stage_key = stage.value
            
            checkpoint = StageCheckpoint(
                stage=stage_key,
                completed=False,
                timestamp=datetime.now().isoformat(),
                duration_seconds=duration_seconds,
                artifacts={},
                metrics={},
                error=str(error)[:1000]  # Truncate long errors
            )
            
            self._state.stages[stage_key] = checkpoint
            
            log.warning(f"[CHECKPOINT] Stage failed: {stage_key} - {error[:100]}")
            
            if self.auto_save:
                self._save_state()
    
    def get_artifact(
        self,
        stage: CheckpointStage,
        artifact_name: str
    ) -> Optional[str]:
        """
        Get an artifact path from a completed stage.
        
        Args:
            stage: Stage to get artifact from
            artifact_name: Name of the artifact
            
        Returns:
            Path to artifact or None if not found
        """
        if not self.enabled or self._state is None:
            return None
        
        with self._lock:
            stage_key = stage.value
            if stage_key not in self._state.stages:
                return None
            
            return self._state.stages[stage_key].artifacts.get(artifact_name)
    
    def get_all_artifacts(self, stage: CheckpointStage) -> Dict[str, str]:
        """Get all artifacts from a stage."""
        if not self.enabled or self._state is None:
            return {}
        
        with self._lock:
            stage_key = stage.value
            if stage_key not in self._state.stages:
                return {}
            return dict(self._state.stages[stage_key].artifacts)
    
    def get_completed_stages(self) -> List[str]:
        """Get list of completed stage names."""
        if not self.enabled or self._state is None:
            return []
        
        with self._lock:
            return [
                k for k, v in self._state.stages.items()
                if v.completed
            ]
    
    def get_progress_summary(self) -> Dict[str, Any]:
        """Get a summary of pipeline progress."""
        if not self.enabled or self._state is None:
            return {"enabled": False}
        
        with self._lock:
            completed = [k for k, v in self._state.stages.items() if v.completed]
            failed = [k for k, v in self._state.stages.items() if v.error]
            
            total_duration = sum(
                s.duration_seconds or 0
                for s in self._state.stages.values()
                if s.completed
            )
            
            return {
                "enabled": True,
                "config_hash": self.config_hash,
                "completed_stages": completed,
                "failed_stages": failed,
                "total_completed": len(completed),
                "total_duration_seconds": total_duration,
                "last_updated": self._state.last_updated
            }
    
    def invalidate(self, stages: List[CheckpointStage] = None) -> None:
        """
        Invalidate checkpoints.
        
        Args:
            stages: Specific stages to invalidate, or None for all
        """
        if not self.enabled or self._state is None:
            return
        
        with self._lock:
            if stages is None:
                # Invalidate all
                self._state = self._create_new_state()
                log.info("[CHECKPOINT] All checkpoints invalidated")
            else:
                # Invalidate specific stages
                for stage in stages:
                    if stage.value in self._state.stages:
                        del self._state.stages[stage.value]
                        log.info(f"[CHECKPOINT] Invalidated: {stage.value}")
            
            self._save_state()
    
    def clean(self) -> None:
        """Remove all checkpoint data."""
        if self.checkpoint_dir.exists():
            try:
                shutil.rmtree(self.checkpoint_dir)
                log.info("[CHECKPOINT] Cleaned checkpoint directory")
            except Exception as e:
                log.warning(f"[CHECKPOINT] Could not clean: {e}")
        
        self._state = None


class CheckpointContext:
    """
    Context manager for automatic checkpoint management.
    
    Usage:
        checkpoint = PipelineCheckpoint(output_dir, config)
        
        with CheckpointContext(checkpoint, CheckpointStage.TRAIN) as ctx:
            # Do training
            model = train_model(data)
            ctx.add_artifact("model_path", str(model_path))
            ctx.add_metric("rmse", rmse_value)
    """
    
    def __init__(
        self,
        checkpoint: PipelineCheckpoint,
        stage: CheckpointStage,
        skip_if_complete: bool = True
    ):
        self.checkpoint = checkpoint
        self.stage = stage
        self.skip_if_complete = skip_if_complete
        self.skipped = False
        self.artifacts: Dict[str, str] = {}
        self.metrics: Dict[str, Any] = {}
        self._start_time: Optional[float] = None
    
    def __enter__(self) -> "CheckpointContext":
        import time
        self._start_time = time.time()
        
        if self.skip_if_complete and self.checkpoint.should_skip(self.stage):
            self.skipped = True
            self.artifacts = self.checkpoint.get_all_artifacts(self.stage)
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        import time
        duration = time.time() - self._start_time if self._start_time else None
        
        if self.skipped:
            return False  # Don't suppress exceptions
        
        if exc_type is None:
            # Success
            self.checkpoint.mark_complete(
                self.stage,
                artifacts=self.artifacts,
                metrics=self.metrics,
                duration_seconds=duration
            )
        else:
            # Failure
            self.checkpoint.mark_failed(
                self.stage,
                error=str(exc_val),
                duration_seconds=duration
            )
        
        return False  # Don't suppress exceptions
    
    def add_artifact(self, name: str, path: str) -> None:
        """Add an artifact to this checkpoint."""
        self.artifacts[name] = path
    
    def add_metric(self, name: str, value: Any) -> None:
        """Add a metric to this checkpoint."""
        self.metrics[name] = value


# =============================================================================
# Helper Functions
# =============================================================================

def resume_or_run(
    checkpoint: PipelineCheckpoint,
    stage: CheckpointStage,
    func: Callable,
    *args,
    artifact_keys: List[str] = None,
    **kwargs
) -> Any:
    """
    Resume from checkpoint or run function.
    
    This is a convenience wrapper that checks for existing checkpoint,
    runs the function if needed, and saves the checkpoint.
    
    Args:
        checkpoint: PipelineCheckpoint instance
        stage: Stage being executed
        func: Function to run
        *args: Positional arguments for func
        artifact_keys: Keys in function result dict to save as artifacts
        **kwargs: Keyword arguments for func
        
    Returns:
        Function result or cached artifacts
    """
    import time
    
    # Check if we can skip
    if checkpoint.should_skip(stage):
        artifacts = checkpoint.get_all_artifacts(stage)
        log.info(f"[CHECKPOINT] Resuming from checkpoint: {stage.value}")
        return artifacts
    
    # Run the function
    start = time.time()
    try:
        result = func(*args, **kwargs)
        duration = time.time() - start
        
        # Extract artifacts from result
        artifacts = {}
        if artifact_keys and isinstance(result, dict):
            for key in artifact_keys:
                if key in result:
                    artifacts[key] = str(result[key])
        elif isinstance(result, (str, Path)):
            artifacts["output"] = str(result)
        
        # Mark complete
        checkpoint.mark_complete(
            stage,
            artifacts=artifacts,
            duration_seconds=duration
        )
        
        return result
        
    except Exception as e:
        duration = time.time() - start
        checkpoint.mark_failed(stage, str(e), duration)
        raise


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == "__main__":
    try:
        from logging_config import setup_logging
        setup_logging()
    except Exception:
        import logging
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    import sys
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    
    if len(sys.argv) < 2:
        log.info("Usage: python checkpoints.py <output_dir> [--status|--clean|--invalidate <stage>]")
        sys.exit(1)
    
    output_dir = Path(sys.argv[1])
    
    # Create dummy config for inspection
    checkpoint = PipelineCheckpoint(
        output_dir=output_dir,
        config={},  # Empty config - just for inspection
        enabled=True
    )
    
    if len(sys.argv) > 2:
        cmd = sys.argv[2]
        
        if cmd == "--status":
            summary = checkpoint.get_progress_summary()
            log.info(json.dumps(summary, indent=2))
            
        elif cmd == "--clean":
            checkpoint.clean()
            log.info("Checkpoints cleaned")
            
        elif cmd == "--invalidate" and len(sys.argv) > 3:
            stage_name = sys.argv[3]
            try:
                stage = CheckpointStage.from_string(stage_name)
                checkpoint.invalidate([stage])
                log.info(f"Invalidated: {stage_name}")
            except ValueError as e:
                log.info(f"Error: {e}")
                sys.exit(1)
        else:
            log.info(f"Unknown command: {cmd}")
            sys.exit(1)
    else:
        # Default: show status
        summary = checkpoint.get_progress_summary()
        log.info(json.dumps(summary, indent=2))

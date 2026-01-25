"""
Prediction Output Validation

Critical validation to catch zero-output and constant-output bugs.
Add this at the end of predict_scene() before writing output.
"""

import numpy as np
import logging

log = logging.getLogger(__name__)


def validate_prediction_output(
    prediction_array: np.ndarray,
    training_depth_range: tuple,
    min_finite_pixels: int = 100
) -> dict:
    """
    Validate prediction output for common failure modes.
    
    Args:
        prediction_array: The predicted depth raster
        training_depth_range: (min_depth, max_depth) from training
        min_finite_pixels: Minimum expected finite predictions
    
    Returns:
        dict with validation results and warnings
    """
    validation = {
        "status": "ok",
        "warnings": [],
        "errors": [],
        "stats": {}
    }
    
    # Get finite predictions
    finite_mask = np.isfinite(prediction_array)
    finite_preds = prediction_array[finite_mask]
    
    n_finite = len(finite_preds)
    validation["stats"]["n_finite"] = n_finite
    validation["stats"]["n_total"] = prediction_array.size
    
    # Check 1: No finite predictions
    if n_finite == 0:
        validation["status"] = "error"
        validation["errors"].append("No finite predictions produced")
        log.error("[PREDICT] ❌ CRITICAL: No finite predictions!")
        return validation
    
    # Check 2: Too few finite predictions
    if n_finite < min_finite_pixels:
        validation["warnings"].append(f"Very few predictions: {n_finite} < {min_finite_pixels}")
        log.warning(f"[PREDICT] ⚠️  Only {n_finite} finite predictions (expected >{min_finite_pixels})")
    
    # Check 3: All zeros
    n_zero = (finite_preds == 0).sum()
    pct_zero = 100.0 * n_zero / n_finite
    validation["stats"]["pct_zero"] = pct_zero
    
    if pct_zero > 99.0:
        validation["status"] = "error"
        validation["errors"].append(f"Output is {pct_zero:.1f}% zeros")
        log.error("[PREDICT] ❌ CRITICAL: Prediction output is all/mostly zeros!")
        log.error("[PREDICT]    Possible causes:")
        log.error("[PREDICT]      1. Model didn't learn (check training RMSE)")
        log.error("[PREDICT]      2. Feature values out of training distribution")
        log.error("[PREDICT]      3. Depth sign convention mismatch")
        log.error("[PREDICT]      4. Linf estimation failed")
    elif pct_zero > 50.0:
        validation["warnings"].append(f"{pct_zero:.1f}% of predictions are zero")
        log.warning(f"[PREDICT] ⚠️  {pct_zero:.1f}% of predictions are zero")
    
    # Check 4: Nearly constant output
    pred_std = finite_preds.std()
    pred_mean = finite_preds.mean()
    validation["stats"]["std"] = float(pred_std)
    validation["stats"]["mean"] = float(pred_mean)
    
    if pred_std < 0.01 and n_finite > 1000:
        validation["warnings"].append(f"Output nearly constant (std={pred_std:.4f})")
        log.warning(f"[PREDICT] ⚠️  Output nearly constant (std={pred_std:.4f})")
        log.warning(f"[PREDICT]    Model may not be generalizing properly")
    
    # Check 5: Prediction range vs training range
    pred_min = float(finite_preds.min())
    pred_max = float(finite_preds.max())
    pred_range = pred_max - pred_min
    
    validation["stats"]["min"] = pred_min
    validation["stats"]["max"] = pred_max
    validation["stats"]["range"] = pred_range
    
    train_min, train_max = training_depth_range
    train_range = train_max - train_min
    
    # Extrapolation check
    if pred_min < train_min - 5.0 or pred_max > train_max + 5.0:
        validation["warnings"].append(
            f"Predictions outside training range: "
            f"pred=[{pred_min:.1f}, {pred_max:.1f}], "
            f"train=[{train_min:.1f}, {train_max:.1f}]"
        )
        log.warning(f"[PREDICT] ⚠️  Predictions extrapolating beyond training:")
        log.warning(f"[PREDICT]    Prediction range: [{pred_min:.2f}, {pred_max:.2f}] m")
        log.warning(f"[PREDICT]    Training range:   [{train_min:.2f}, {train_max:.2f}] m")
    
    # Check 6: Sign consistency
    pct_negative = (finite_preds < 0).mean() * 100
    pct_positive = (finite_preds > 0).mean() * 100
    validation["stats"]["pct_negative"] = pct_negative
    validation["stats"]["pct_positive"] = pct_positive
    
    if train_min < 0 and train_max < 0:
        # Training was all negative, predictions should be mostly negative
        if pct_positive > 20:
            validation["warnings"].append(
                f"Training was negative, but {pct_positive:.1f}% predictions are positive"
            )
            log.warning(f"[PREDICT] ⚠️  Sign mismatch: training all negative, {pct_positive:.1f}% preds positive")
    
    # Summary log
    log.info("[PREDICT] Output validation:")
    log.info(f"  Finite pixels: {n_finite:,} ({100*n_finite/validation['stats']['n_total']:.1f}%)")
    log.info(f"  Range: [{pred_min:.2f}, {pred_max:.2f}] m")
    log.info(f"  Mean: {pred_mean:.2f} m, Std: {pred_std:.2f} m")
    log.info(f"  Sign: {pct_negative:.1f}% negative, {pct_positive:.1f}% positive, {pct_zero:.1f}% zero")
    
    if validation["status"] == "error":
        log.error(f"[PREDICT] ❌ Validation FAILED: {len(validation['errors'])} error(s)")
    elif validation["warnings"]:
        log.warning(f"[PREDICT] ⚠️  Validation passed with {len(validation['warnings'])} warning(s)")
    else:
        log.info("[PREDICT] ✅ Output validation passed")
    
    return validation


# Usage in predict.py:
# After computing predict_arr, before writing to file:
#
# validation = validate_prediction_output(
#     predict_arr,
#     training_depth_range=(model_meta.get('depth_min', -50), model_meta.get('depth_max', 0)),
#     min_finite_pixels=1000
# )
#
# if validation["status"] == "error":
#     # Save diagnostic outputs
#     np.save(str(out_path).replace('.tif', '_FAILED.npy'), predict_arr)
#     log.error(f"[PREDICT] Diagnostic array saved: {out_path}_FAILED.npy")

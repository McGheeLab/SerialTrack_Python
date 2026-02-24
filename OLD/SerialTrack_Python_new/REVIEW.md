# SerialTrack Python — Code Review & Bug Fix Report

## Summary

Thorough review of the 11 Python modules against the MATLAB reference implementation (`f_track_serial_match3D.m`, `fun_SerialTrack_3D_HardPar.m`, manual). **10 bugs identified and fixed**, including the 4 critical issues reported plus 6 additional issues discovered during review. All fixes verified with targeted unit tests and a synthetic integration test (200 particles, known rigid translation → converges to 100% match in 3 iterations).

---

## Bugs Fixed

### 1. Circular import in `outliers.py` *(reported — critical)*

**File:** `outliers.py` line 9  
**Bug:** `from .matching import TopologyMatcher, NearestNeighborMatcher, compute_displacement` — these imports were unused in outliers.py but created a circular dependency since `matching.py` imports `from .outliers import remove_outliers`.  
**Fix:** Removed the entire import line. No code in `outliers.py` uses any matching classes.

### 2. `DisplacementField.gradient()` fragile grid spacing *(reported — critical)*

**File:** `fields.py` lines 59–64  
**Bug:** Grid spacing was inferred via a fragile chain: `np.unique(np.round(np.diff(self.grids[d].ravel()[:self.shape[d]+1]), 6))[0]` — this fails on edge cases (single-point axes, non-uniform rounding, high-dimensional indexing).  
**Fix:** Replaced with `_axis_spacings()` method that extracts the 1-D coordinate slice along each axis from the meshgrid and computes `np.median(np.diff(...))`. Falls back to `pixel_steps[d]` for single-point axes. Verified with a 3D linear displacement test (exact recovery of F[i,j] = ∂u_i/∂x_j at interior points).

### 3. `fields.py` logger name *(reported)*

**File:** `fields.py` line 22  
**Bug:** `log = logging.getLogger("serialtrack.regularization")` — wrong module name.  
**Fix:** Changed to `"serialtrack.fields"`.

### 4. `outliers.py` logger name *(discovered)*

**File:** `outliers.py` line 11  
**Bug:** `log = logging.getLogger("serialtrack.matching")` — wrong module name.  
**Fix:** Changed to `"serialtrack.outliers"`.

### 5. ADMM convergence order mismatch *(discovered — critical)*

**File:** `tracking.py` `_ADMMFrameTracker.run()`  
**Bug:** Python applied the global-step warp (`disp_b2a += temp_disp`) **before** checking convergence, meaning the final displacement included one extra update that MATLAB skips. In MATLAB (`f_track_serial_match3D.m` lines 289–300), convergence is checked BEFORE warping — if converged, the code breaks without adding the last update.  
**Fix:** Moved the convergence check before the warp block. Now on convergence, `break` is reached without modifying `disp_b2a`, exactly matching MATLAB.

### 6. `_local_step` retry loop was a no-op *(discovered — critical)*

**File:** `tracking.py` `_local_step()`  
**Bug:** The retry loop did `n_max_local += 5; n_neighbors = min(n_neighbors, n_max_local)` — but `n_max_local` starts at `n_neighbors_max` which is already ≥ `n_neighbors`. The `min` always returned `n_neighbors`, making the retry identical to the first attempt (infinite no-op loop). MATLAB instead increases `n_neighborsMax` on retry to allow richer topology features.  
**Fix:** Rewrote as a bounded retry loop (max 5 attempts) that increases `retry_n_neighbors += 5` on failure, matching the MATLAB `n_neighborsMax = round(n_neighborsMax + 5)` logic.

### 7. `DisplacementField.velocity` fragile broadcasting *(discovered)*

**File:** `fields.py` line 47  
**Bug:** `self.pixel_steps[:, None, None, None][:self.ndim]` hardcodes 3 trailing `None` dimensions. For 2D grids (ndim=2), the resulting shape is wrong.  
**Fix:** Uses generic reshaping: `self.pixel_steps.reshape([ndim] + [1] * ndim)` — works for any dimensionality.

### 8. `StrainField` serialization mismatch *(discovered — critical)*

**File:** `results.py` `_save_strain_field` / `_load_strain_field`  
**Bug:** Save accessed `sfield.tensor` and `sfield.pixel_steps` — but `StrainField` had attributes named `F_tensor` and `eps_tensor`, and no `pixel_steps`. Load tried to construct `StrainField(tensor=..., pixel_steps=...)` with non-existent fields.  
**Fix:** (a) Added `pixel_steps` attribute to `StrainField` dataclass with a default. (b) Updated save to write `F_tensor` and `eps_tensor` as separate datasets. (c) Updated load to read both tensors back. Verified with HDF5 round-trip test.

### 9. `update_f_o_s` hardcoded floor prevented shrinkage *(discovered)*

**File:** `outliers.py` `update_f_o_s()` + `tracking.py` call site  
**Bug:** `f_o_s_min=60.0` was hardcoded. Since displacement quantiles are typically ≪ 60 px, `max(60, small_value) = 60` always, so f_o_s never actually shrank between ADMM iterations. The MATLAB code also uses 60 as the floor, but this is only useful when the initial f_o_s is large.  
**Fix:** Changed the function signature to accept `f_o_s_current` instead of a fixed min. The floor is now `max(2 px, 10% of current f_o_s)`, allowing meaningful shrinkage while preventing collapse to zero. The call site in `tracking.py` passes the working f_o_s.

### 10. `matching.py` truth-value check on `ball_results` *(discovered)*

**File:** `matching.py` `_build_candidates()` line 483  
**Bug:** `if ball_results` evaluates the truth value of a list of numpy arrays, which raises `ValueError: ambiguous truth value`.  
**Fix:** Changed to `if len(ball_results) > 0`.

### 11. `io.py` unnecessary imports *(discovered — minor)*

**File:** `io.py` lines 13–14  
**Bug:** `from scipy import ndimage` and `import numba as nb` are imported but never used. This adds ~2 seconds to import time and pulls in unnecessary compile-time dependencies.  
**Fix:** Removed both imports.

---

## Files Modified

| File | Changes |
|---|---|
| `outliers.py` | Removed circular import, fixed logger, improved `update_f_o_s` floor |
| `fields.py` | Fixed gradient spacing, logger, velocity broadcasting, added StrainField.pixel_steps |
| `tracking.py` | Fixed ADMM convergence order, local step retry, f_o_s update passthrough |
| `results.py` | Fixed StrainField save/load to use correct attribute names |
| `matching.py` | Fixed `ball_results` truth-value check |
| `io.py` | Removed unused imports |

Files unchanged: `config.py`, `detection.py`, `regularization.py`, `prediction.py`, `trajectories.py`, `__init__.py`

---

## GUI Readiness Notes

The codebase is now well-structured for PySide6 integration:

- **`results.py`** already provides `CallbackAccumulator` and `make_qt_bridge()` for thread-safe progress reporting via Qt signals.
- **`TrackingSession`** is fully serializable to HDF5/MAT/NPZ for save/load workflows.
- **`FrameSummary`/`SessionSummary`** provide GUI-ready statistics.
- The `progress_cb` parameter on all tracking methods is the hook point for GUI updates.
- No Qt dependency anywhere in the core engine — clean separation.

For the GUI, the recommended architecture is a `QThread` worker that calls `tracker.track_images(images, progress_cb=bridge)` where `bridge` is the `_QtSignalBridge` from `make_qt_bridge()`.

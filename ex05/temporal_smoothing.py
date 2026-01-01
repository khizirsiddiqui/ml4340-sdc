import numpy as np
from scipy.signal import butter, filtfilt
from scipy.interpolate import splev, splprep

class SplineRefiner:
    @staticmethod
    def smooth_boundary(new_tck, old_tck, alpha=0.8, resolution=10, max_deviation=50.0):
        """
        Temporally smooths a spline tuple (tck) with outlier rejection.

        Args:
            new_tck: The raw (knots, coeffs, degree) tuple from the current frame.
            old_tck: The smoothed (knots, coeffs, degree) tuple from the previous frame.
            alpha: Smoothing factor (0.0 to 1.0).
                   1.0 = use only new data. 0.1 = heavy lag.
            resolution: Number of points to use for evaluation.
            max_deviation: Threshold (in pixels). If the average distance between
                           new and old splines exceeds this, the new spline is
                           considered noise and rejected.

        Returns:
            A (knots, coeffs, degree) tuple.
        """
        # 1. Initialization Check
        if old_tck is None:
            return new_tck

        try:
            u_eval = np.linspace(0, 1, resolution)
            pts_new = np.array(splev(u_eval, new_tck))
            pts_old = np.array(splev(u_eval, old_tck))

            distances = np.linalg.norm(pts_new - pts_old, axis=0)
            avg_deviation = np.mean(distances)

            if avg_deviation > max_deviation:
                return old_tck

            pts_smooth = (alpha * pts_new) + ((1 - alpha) * pts_old)
            smoothed_tck, _ = splprep(pts_smooth, s=0, k=3)

            return smoothed_tck

        except Exception:
            # Fallback for math errors or shape mismatches
            return new_tck

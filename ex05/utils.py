from scipy.interpolate import splprep, splev
import numpy as np
import json

def remove_duplicates(points, tol=1e-6):
    diffs = np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1))
    keep = np.r_[True, diffs > tol]
    return points[keep]

def fit_line_from_points(points, smoothing=0.1):
    smoothing = smoothing * points.shape[0]
    points = remove_duplicates(points)
    if points.shape[0] < 2:
        return np.empty((2, 0))
    tck, u = splprep(points.T, s=smoothing)
    u_fine = np.linspace(0, 1, num=100)
    x_fine, y_fine = splev(u_fine, tck)
    return np.vstack((x_fine, y_fine))

def distance_point_to_line(point, list_of_points_on_line, smoothing=0.1):
    if list_of_points_on_line.shape[1] == 0:
        return np.inf
    if list_of_points_on_line.shape[1] == 1:
        min_dist = np.linalg.norm(point - list_of_points_on_line[:, 0])
        min_dist = np.sign(np.cross(list_of_points_on_line[:, 0] - point, np.array([1, 0]))) * min_dist
        return min_dist
    points_on_line = fit_line_from_points(list_of_points_on_line, smoothing)
    sign = 1.0
    min_dist = np.inf
    for i in range(points_on_line.shape[1] - 1):
        a = points_on_line[:, i]
        b = points_on_line[:, i + 1]
        ab = b - a
        ap = point - a
        proj = np.dot(ap, ab) / np.dot(ab, ab)
        proj = np.clip(proj, 0, 1)
        closest = a + proj * ab
        dist = np.linalg.norm(point - closest)
        if dist < min_dist:
            min_dist = dist
            cross = np.cross(ab, ap)
            sign = np.sign(cross)
    min_dist *= sign
    return min_dist

def get_config(path='config.json'):
    with open(path, 'r') as f:
        config = json.load(f)
    return config

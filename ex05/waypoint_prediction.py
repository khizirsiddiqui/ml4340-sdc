import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.interpolate import splprep, splev
from scipy.optimize import minimize
import time

old_waypoint_predictions = None
old_target_speed = None

curvature_history = []

def normalize(v):
    norm = np.linalg.norm(v,axis=0) + 1e-5
    return v / norm.reshape(1, v.shape[1])

def curvature(waypoints):
    '''
    Curvature as  the sum of the normalized dot product between the way elements
    Implement second term of the smoothin objective.

    args:
        waypoints [2, num_waypoints] !!!!!
    '''
    dir_vectors = waypoints[:,1:] - waypoints[:,:-1]
    dir_vectors_normalized = normalize(dir_vectors)
    dot_products = np.sum(dir_vectors_normalized[:,1:] * dir_vectors_normalized[:,:-1], axis=0)
    curvature = np.sum(dot_products)
    return curvature

def curvature2(waypoints):
    x = waypoints[0, :]
    y = waypoints[1, :]
    dx = np.gradient(x)
    dy = np.gradient(y)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    # Curvature k = (x'y'' - y'x'') / (x'^2 + y'^2)^(3/2)
    num = np.abs(dx * ddy - dy * ddx)
    den = np.power(dx**2 + dy**2, 1.5) + 1e-8
    return num / den

def smoothing_objective(waypoints, waypoints_center, weight_curvature=40):
    '''
    Objective for path smoothing

    args:
        waypoints [2 * num_waypoints] !!!!!
        waypoints_center [2 * num_waypoints] !!!!!
        weight_curvature (default=40)
    '''
    # mean least square error between waypoint and way point center
    ls_tocenter = np.mean((waypoints_center - waypoints)**2)

    # derive curvature
    curv = curvature(waypoints.reshape(2,-1))

    return ls_tocenter - weight_curvature * curv


def waypoint_prediction(roadside1_spline, roadside2_spline, time_step=0, num_waypoints=10, way_type = "smooth2"):
    '''
    Predict waypoint via two different methods:
    - center
    - smooth

    args:
        roadside1_spline
        roadside2_spline
        time_step
        num_waypoints (default=6)
        parameter_bound_waypoints (default=1)
        waytype (default="smoothed")
    '''
    global old_waypoint_predictions
    global curvature_history

    if time_step < 50:
        waypoints = np.zeros((2, num_waypoints))
        waypoints[0, :] = 48
        waypoints[1, :] = np.linspace(0, num_waypoints - 1, num_waypoints)
        return waypoints

    def get_center_waypoints():
        # create spline arguments
        spline_args = np.linspace(time_step, time_step + 1, num_waypoints)
        # derive roadside points from spline
        roadside1_points = np.array(splev(spline_args, roadside1_spline))
        roadside2_points = np.array(splev(spline_args, roadside2_spline))
        # derive center between corresponding roadside points
        # output way_points with shape(2 x Num_waypoints)
        way_points = (roadside1_points + roadside2_points) / 2
        return way_points

    if way_type == "center":
        way_points = get_center_waypoints()
        return way_points

    elif way_type == "smooth":
        way_points_center = get_center_waypoints().flatten()
        # optimization
        way_points = minimize(smoothing_objective,
                      (way_points_center),
                      args=way_points_center)["x"]

        waypoints = way_points.reshape(2,-1)

    elif way_type == "smooth2":
        t_vals = np.linspace(0, 1, num_waypoints)
        left_pts = np.array(splev(t_vals, roadside1_spline))
        right_pts = np.array(splev(t_vals, roadside2_spline))
        w_initial = np.full(num_waypoints, 0.5)
        bounds = [(0.05, 0.95) for _ in range(num_waypoints)]
        img_size = 96 * 65
        def objective(w, v=0):
            path = left_pts * w + right_pts * (1 - w)
            diffs = path[:, 2:] - 2 * path[:, 1:-1] + path[:, :-2]
            smooth = np.mean(diffs**2) / (img_size)
            centering = np.mean((w-0.5)**2) / (0.25)
            if old_waypoint_predictions is None:
                temporal = 0
            else:
                temporal = np.mean((path - old_waypoint_predictions)**2) / (img_size**2)
                temporal = np.sqrt((temporal))
            cost = smooth + 80 * centering + 0.2 * temporal
            if v==1:
                print("cost: ", cost, " smooth:", smooth, " centering:", centering, " temporal:", temporal)
            return cost
        optimal_w = minimize(objective, w_initial, bounds=bounds)["x"]
        waypoints = left_pts * optimal_w + right_pts * (1 - optimal_w)

    if old_waypoint_predictions is None:
        old_waypoint_predictions = waypoints

    curvature_history.append(curvature2(waypoints))
    return waypoints

def plot_curvature_history():
    global curvature_history
    # curvature_history is a list of curvature values over time
    plt.figure()
    plt.plot(curvature_history)
    plt.title("Curvature History")
    plt.xlabel("Time Step")
    plt.ylabel("Curvature")
    plt.yscale("log")
    plt.grid()
    plt.savefig("curvature_history.png")


def target_speed_prediction_old(waypoints, time_step=0, num_waypoints_used=5, max_speed=60, exp_constant=4.5, offset_speed=30):
    '''
    Predict target speed given waypoints
    Implement the function using curvature()

    args:
        waypoints [2,num_waypoints]
        time_step(int)
        num_waypoints_used (default=5)
        max_speed (default=60)
        exp_constant (default=4.5)
        offset_speed (default=30)

    output:
        target_speed (float)
    '''
    # v_target = ( max_speed - min_speed ) * exp( -K_v * | N - 2 - curvature | ) + min_speed
    curv = curvature(waypoints[:,:num_waypoints_used])
    target_speed = (max_speed - offset_speed) * np.exp(-exp_constant * np.abs(num_waypoints_used - 2 - curv)) + offset_speed
    return target_speed

def target_speed_prediction(waypoints, time_step=0, num_waypoints_used=5, max_speed=60, offset_speed=15):
    '''
    Predict target speed given waypoints
    Implement the function using curvature()

    args:
        waypoints [2,num_waypoints]
        time_step(int)
        num_waypoints_used (default=5)
        max_speed (default=60)
        offset_speed (default=30)

    output:
        target_speed (float)
    '''
    global old_target_speed
    if time_step < 50:
        old_target_speed = max_speed
        return max_speed

    num_waypoints_used = int(num_waypoints_used + old_target_speed / 5)
    future_waypoints = waypoints[:, :num_waypoints_used]

    max_curv = 0
    for i in range(len(future_waypoints[0]) - 2):
        p_segment = future_waypoints[:, i:i+3]
        curv_segment = curvature2(p_segment)
        current_max = np.max(curv_segment)
        if current_max > max_curv:
            max_curv = current_max

    with np.errstate(divide='ignore'):
        radius = 1.0 / (max_curv + 1e-8)

    safe_v = 2 * np.sqrt(radius)
    target_v = np.clip(safe_v, offset_speed, max_speed)
    if old_target_speed is not None:
        if target_v > old_target_speed:
            # accelerate slowly
            target_v = 0.9 * old_target_speed + 0.1 * target_v
        else:
            # brake quickly
            target_v = 0.3 * old_target_speed + 0.7 * target_v
    else:
        old_target_speed = target_v

    # if waypoints are close to each other, reduce speed because they are noisy
    dists = np.linalg.norm(waypoints[:,1:] - waypoints[:,:-1], axis=0)
    if np.mean(dists) < 2.0:
        target_v = 0.7 * target_v
    return target_v

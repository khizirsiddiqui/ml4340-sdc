import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.interpolate import splprep, splev
from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter1d
import time
from PIL import Image

from utils import remove_duplicates, get_config
from temporal_smoothing import SplineRefiner


class LaneDetectionBase:
    '''
    Base class for lane detection modules
    '''

    def lane_detection(self, state_image_full, time_step=0):
        '''
        This function should perform the road detection

        args:
            state_image_full shape: 96x96x3
            time_step (int) [optional, default=0]

        out:
            lane_boundary1 spline
            lane_boundary2 spline
        '''
        raise NotImplementedError

    def plot_state_lane(self, state_image_full, steps, fig, waypoints=[]):
        '''
        Plot lanes and way points
        '''
        if steps < 50:
            return False
        # evaluate spline for 6 different spline parameters.
        t = np.linspace(0, 1, 6)
        lane_boundary1_points_points = np.array(splev(t, self.lane_boundary1_old))
        lane_boundary2_points_points = np.array(splev(t, self.lane_boundary2_old))

        plt.gcf().clear()
        plt.imshow(state_image_full[::-1])
        plt.plot(lane_boundary1_points_points[0], lane_boundary1_points_points[1]+96-self.cut_size, linewidth=5, color='blue')
        plt.plot(lane_boundary2_points_points[0], lane_boundary2_points_points[1]+96-self.cut_size, linewidth=5, color='orange')
        if len(waypoints):
            plt.scatter(waypoints[0], waypoints[1]+96-self.cut_size, color='white')

        plt.axis('off')
        plt.xlim((-0.5,95.5))
        plt.ylim((-0.5,95.5))
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.gca().axes.get_yaxis().set_visible(False)

        try:
            fig.canvas.flush_events()
            return False
        except:
            return True

class LaneFindingVanilla:
    def find_maxima_gradient_rowwise(self, gradient_sum, base):
        '''
        This function should output arguments of local maxima for each row of the gradient image.
        You can use scipy.signal.find_peaks to detect maxima.
        Hint: Use distance argument for a better robustness.

        input:
            gradient_sum 65x96x1

        output:
            maxima () 2x Number_maxima

        '''
        argmaxima = []
        for row in range(base.cut_size):
            peaks, _ = find_peaks(gradient_sum[row], distance=base.distance_maxima_gradient)
            for peak in peaks:
                argmaxima.append([peak, row])
        return argmaxima

    def find_first_lane_point(self, gradient_sum, base):
        '''
        Find the first lane_boundary points above the car.
        Special cases like just detecting one lane_boundary or more than two are considered.
        Even though there is space for improvement ;)

        input:
            gradient_sum 65x96x1

        output:
            lane_boundary1_startpoint
            lane_boundary2_startpoint
            lanes_found  true if lane_boundaries were found
        '''

        # Variable if lanes were found or not
        lanes_found = False
        row = 0

        # loop through the rows
        while not lanes_found:

            # Find peaks with min distance of at least 3 pixel
            argmaxima = find_peaks(gradient_sum[row],distance=3)[0]

            # if one lane_boundary is found
            if argmaxima.shape[0] == 1:
                lane_boundary1_startpoint = np.array([[argmaxima[0],  row]])

                if argmaxima[0] < 48:
                    lane_boundary2_startpoint = np.array([[0,  row]])
                else:
                    lane_boundary2_startpoint = np.array([[96,  row]])

                lanes_found = True

            # if 2 lane_boundaries are found
            elif argmaxima.shape[0] == 2:
                lane_boundary1_startpoint = np.array([[argmaxima[0],  row]])
                lane_boundary2_startpoint = np.array([[argmaxima[1],  row]])
                lanes_found = True

            # if more than 2 lane_boundaries are found
            elif argmaxima.shape[0] > 2:
                # if more than two maxima then take the two lanes next to the car, regarding least square
                A = np.argsort((argmaxima - base.car_position[0])**2)
                lane_boundary1_startpoint = np.array([[argmaxima[A[0]],  0]])
                lane_boundary2_startpoint = np.array([[argmaxima[A[1]],  0]])
                lanes_found = True

            row += 1

            # if no lane_boundaries are found
            if row == base.cut_size:
                lane_boundary1_startpoint = np.array([[0,  0]])
                lane_boundary2_startpoint = np.array([[0,  0]])
                break

        return lane_boundary1_startpoint, lane_boundary2_startpoint, lanes_found

    def find_lanes_from_maxima(self, gradient_sum, base):
        '''
        This function should output arguments of local maxima for each row of the gradient image.
        You can use scipy.signal.find_peaks to detect maxima.
        Hint: Use distance argument for a better robustness.

        input:
            gradient_sum 65x96

        output:
            maxima (List) 2x Number_maxima

        '''
        cut_size = base.cut_size
        maxima = self.find_maxima_gradient_rowwise(gradient_sum, base)

        # first lane_boundary points
        lane_boundary1_points, lane_boundary2_points, lane_found = self.find_first_lane_point(gradient_sum, base)

        LOOKAHEAD_ROWS = 5
        MAX_X_SHIFT = 15
        def get_next_point(current_points, available_maxima):
            if len(available_maxima) == 0:
                return None, available_maxima
            last_point = current_points[-1]
            last_x, last_y = last_point[0], last_point[1]

            candidates_idx = [i for i, m in enumerate(available_maxima) if 0 < m[1] - last_y <= LOOKAHEAD_ROWS]
            if not candidates_idx:
                return None, available_maxima

            best_idx = -1
            min_x_distance = MAX_X_SHIFT
            for i in candidates_idx:
                cur_x = available_maxima[i][0]
                x_distance = abs(cur_x - last_x) + abs(available_maxima[i][1] - last_y) * 0.5
                if x_distance < min_x_distance:
                    min_x_distance = x_distance
                    best_idx = i
            if best_idx == -1:
                return None, available_maxima
            next_point = available_maxima[best_idx]
            available_maxima.pop(best_idx)
            return next_point, available_maxima

        keep_searching_1 = True
        keep_searching_2 = True
        while keep_searching_1 or keep_searching_2:
            if keep_searching_1:
                next_point1, maxima = get_next_point(lane_boundary1_points.tolist(), maxima)
            if keep_searching_2:
                next_point2, maxima = get_next_point(lane_boundary2_points.tolist(), maxima)

            if keep_searching_2 and keep_searching_1:
                if next_point1 is not None and next_point2 is not None:
                    if next_point1[0] > next_point2[0] and abs(next_point1[1] - next_point2[1]) <= 1:
                        next_point1, next_point2 = next_point2, next_point1

            if keep_searching_2:
                if next_point2 is not None:
                    lane_boundary2_points = np.vstack([lane_boundary2_points, next_point2])
                else:
                    keep_searching_2 = False
            if keep_searching_1:
                if next_point1 is not None:
                    lane_boundary1_points = np.vstack([lane_boundary1_points, next_point1])
                else:
                    keep_searching_1 = False

        lane_boundary1_points = remove_duplicates(lane_boundary1_points)
        lane_boundary2_points = remove_duplicates(lane_boundary2_points)
        return lane_boundary1_points, lane_boundary2_points, lane_found

class LaneFindingSlidingWindow:
    def find_lanes_from_maxima(self, gradient, base):
        min_threshold = base.kwargs.get('sliding_window_min_threshold', 5)
        window_margin = base.kwargs.get('sliding_window_margin', 30)

        hieght, width = gradient.shape
        bottom_half = gradient[hieght//2:,:]
        histogram = np.sum(bottom_half, axis=0)
        midpoint = histogram.shape[0]//2

        left_peaks, _ = find_peaks(histogram[:midpoint], distance=10)
        right_peaks, _ = find_peaks(histogram[midpoint:], distance=10)

        if len(left_peaks) > 0:
            leftx_base = left_peaks[np.argmax(histogram[left_peaks])]
        else:
            leftx_base = np.argmax(histogram[:midpoint])

        if len(right_peaks) > 0:
            rightx_base = midpoint + right_peaks[np.argmax(histogram[midpoint + right_peaks])]
        else:
            rightx_base = midpoint + np.argmax(histogram[midpoint:])

        # Ensure minimum separation between lanes
        min_lane_separation = width // 4
        if rightx_base - leftx_base < min_lane_separation:
            # Adjust if too close
            center = (leftx_base + rightx_base) // 2
            leftx_base = max(0, center - min_lane_separation // 2)
            rightx_base = min(width - 1, center + min_lane_separation // 2)

        nwindows = base.kwargs.get('sliding_window_nwindows', 9)
        window_height = int(hieght / nwindows)
        nonzero = gradient.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        left_lane_indices = []
        right_lane_indices = []
        leftx_current = leftx_base
        rightx_current = rightx_base

        for window in range(nwindows):
            win_y_low = hieght - (window + 1) * window_height
            win_y_high = hieght - window * window_height
            win_xleft_low = leftx_current - window_margin
            win_xleft_high = leftx_current + window_margin
            win_xright_low = rightx_current - window_margin
            win_xright_high = rightx_current + window_margin

            good_left_indices = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                                 (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_indices = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                                  (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

            left_lane_indices.append(good_left_indices)
            right_lane_indices.append(good_right_indices)

            if len(good_left_indices) > min_threshold:
                leftx_current = int(np.mean(nonzerox[good_left_indices]))
                max_shift = max(0, rightx_current - leftx_current - min_lane_separation)
                leftx_current = min(leftx_current, rightx_current - min_lane_separation)
            if len(good_right_indices) > min_threshold:
                rightx_current = int(np.mean(nonzerox[good_right_indices]))
                min_shift = min(0, leftx_current + min_lane_separation - rightx_current)
                rightx_current = max(rightx_current, leftx_current + min_lane_separation)
        left_lane_indices = np.concatenate(left_lane_indices)
        right_lane_indices = np.concatenate(right_lane_indices)

        leftx = nonzerox[left_lane_indices]
        lefty = nonzeroy[left_lane_indices]
        rightx = nonzerox[right_lane_indices]
        righty = nonzeroy[right_lane_indices]

        lane_boundary1_points = np.array(list(zip(leftx, lefty)))
        lane_boundary2_points = np.array(list(zip(rightx, righty)))
        lane_found = lane_boundary1_points.shape[0] > 0 and lane_boundary2_points.shape[0] > 0
        return lane_boundary1_points, lane_boundary2_points, lane_found

class LaneDetectionVanilla(LaneDetectionBase):
    '''
    Lane detection module using edge detection and b-spline fitting

    args:
        cut_size (cut_size=65) cut the image at the front of the car
        spline_smoothness (default=0.1) multiplied by number of lane points
        gradient_threshold (default=14)
        distance_maxima_gradient (default=3)

    '''

    def __init__(self, cut_size=65, spline_smoothness=0.1, gradient_threshold=14, distance_maxima_gradient=3, **kwargs):
        self.car_position = np.array([48,0])
        self.spline_smoothness = spline_smoothness
        self.cut_size = cut_size
        self.gradient_threshold = gradient_threshold
        self.distance_maxima_gradient = distance_maxima_gradient
        self.lane_boundary1_old = (0, None)
        self.lane_boundary2_old = (0, None)

        self.temporal_smoothing_method = kwargs.get('temporal_smoothing_method', 'none')
        # self.smoother_lane1 = TemporalSmoother(method=self.temporal_smoothing_method, **kwargs)
        # self.smoother_lane2 = TemporalSmoother(method=self.temporal_smoothing_method, **kwargs)

        self.smoother_lane1 = SplineRefiner()
        self.smoother_lane2 = SplineRefiner()
        self.kwargs = kwargs

        self.gradient_frames = []

    def cut_gray(self, state_image_full):
        '''
        This function should cut the image at the front end of the car (e.g. pixel row 65)
        and translate to grey scale

        input:
            state_image_full 96x96x3

        output:
            gray_state_image 65x96x1

        '''
        # crops out the area above self.cut_size
        cut_image = state_image_full[:self.cut_size, :, :]
        gray_state_image = cut_image.mean(axis=2) / 255.0
        return gray_state_image[::-1]


    def edge_detection(self, gray_image):
        '''
        In order to find edges in the gray state image,
        this function should derive the absolute gradients of the gray state image.
        Derive the absolute gradients using numpy for each pixel.
        To ignore small gradients, set all gradients below a threshold (self.gradient_threshold) to zero.

        input:
            gray_state_image 65x96x1

        output:
            gradient_sum 65x96x1

        '''
        gradient_x = np.abs(np.gradient(gray_image, axis=1))
        gradient_y = np.abs(np.gradient(gray_image, axis=0))
        gradient_sum = gradient_x + gradient_y
        gradient_sum[gradient_sum < self.gradient_threshold / 255.0 ] = 0
        gradient_sum[gradient_sum >= self.gradient_threshold / 255.0 ] = 1
        gradient_sum = gaussian_filter1d(gradient_sum, sigma=2)
        return gradient_sum

    def save_gradient_frames_as_gif(self, filename):
        '''
        Save the gradient frames as a gif for visualization

        args:
            filename (str)
        '''
        pil_images = []
        for frame in self.gradient_frames:
            img = (frame * 255).astype(np.uint8)
            pil_image = Image.fromarray(img)
            pil_images.append(pil_image.convert("P"))
        pil_images[0].save(
            filename,
            save_all=True,
            append_images=pil_images[1:],
            duration=100,
            loop=0
        )

    def lane_detection(self, state_image_full, time_step=0):
        '''
        This function should perform the road detection

        args:
            state_image_full [96, 96, 3]
            time_step (int)

        out:
            lane_boundary1 spline
            lane_boundary2 spline
        '''
        if time_step < 50:
            return self.lane_boundary1_old, self.lane_boundary2_old
        # to gray
        gray_state = self.cut_gray(state_image_full)
        # debug:
        # plt.imshow(gray_state[::-1].squeeze(), cmap='gray')
        # edge detection via gradient sum and thresholding

        gradient_sum = self.edge_detection(gray_state)
        self.gradient_frames.append(gradient_sum)
        # debug:
        # plt.imshow(gradient_sum[::-1].squeeze(), cmap='gray')
        if self.kwargs.get('sliding_window', False) and time_step > 50:
            lane_boundary1_points, lane_boundary2_points, lane_found = LaneFindingSlidingWindow().find_lanes_from_maxima(gradient_sum, self)
        else:
            lane_boundary1_points, lane_boundary2_points, lane_found = LaneFindingVanilla().find_lanes_from_maxima(gradient_sum, self)

        if lane_found:
            # spline fitting using scipy.interpolate.splprep
            # and the arguments self.spline_smoothness
            #
            # if there are more lane_boundary points points than spline parameters
            # else use perceding spline
            if lane_boundary1_points.shape[0] > 4 and lane_boundary2_points.shape[0] > 4:
                #### KHIZIR: Ideas for improvements
                #### 1. temporal smoothing
                lane_boundary1, _ = splprep(
                        [lane_boundary1_points[:, 0], lane_boundary1_points[:, 1]],
                        s=self.spline_smoothness * lane_boundary1_points.shape[0]
                )
                lane_boundary2, _ = splprep(
                        [lane_boundary2_points[:, 0], lane_boundary2_points[:, 1]],
                        s=self.spline_smoothness * lane_boundary2_points.shape[0]
                )
            else:
                lane_boundary1 = self.lane_boundary1_old
                lane_boundary2 = self.lane_boundary2_old
            ################

        else:
            lane_boundary1 = self.lane_boundary1_old
            lane_boundary2 = self.lane_boundary2_old

        # account for curves in smoothing param
        def compute_smoothing_param(num_points):
            if num_points > 15:
                return self.spline_smoothness * num_points * 0.4
            else:
                return self.spline_smoothness * num_points

        # temporal smoothing
        resolution = compute_smoothing_param(lane_boundary1_points.shape[0])
        lane_boundary1 = self.smoother_lane1.smooth_boundary(lane_boundary1, self.lane_boundary1_old, resolution=resolution)
        resolution = compute_smoothing_param(lane_boundary2_points.shape[0])
        lane_boundary2 = self.smoother_lane2.smooth_boundary(lane_boundary2, self.lane_boundary2_old, resolution=resolution)

        self.lane_boundary1_old = lane_boundary1
        self.lane_boundary2_old = lane_boundary2

        # output the spline
        return lane_boundary1, lane_boundary2


# read the config file and set the LaneDetection class accordingly
config = get_config()
if config['lane_detection']['module'] == 'vanilla':
    lane_detection_params = config['lane_detection']['params']
    LaneDetection = lambda: LaneDetectionVanilla(**lane_detection_params)
else:
    raise ValueError('Unknown lane detection module')

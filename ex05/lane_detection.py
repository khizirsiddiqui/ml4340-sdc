import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.interpolate import splprep, splev
from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter1d
from scipy import ndimage
import time
from PIL import Image
import PIL.ImageDraw as ImageDraw

from utils import remove_duplicates, get_config
from temporal_smoothing import SplineRefiner

def get_smooth_points(x, y, hieght):
    if len(x) < 5: return None
    fit = np.polyfit(y, x, 2)
    ploty = np.linspace(0, hieght - 1, hieght)
    fit_x = fit[0]*ploty**2 + fit[1]*ploty + fit[2]
    return np.array(list(zip(fit_x, ploty)))

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
        if len(gradient_sum.shape) == 3:
            gradient_sum = gradient_sum.squeeze(-1)
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

        if lane_found:
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

class LaneFindingSlidingWindow(LaneFindingVanilla):
    def __init__(self):
        super().__init__()
        self.window_history = []

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
        lane_boundary1_points, lane_boundary2_points, lane_found = self.find_first_lane_point(gradient_sum, base)

        lane_1_windows = []
        lane_2_windows = []
        window_height = 10
        window_width = 20
        # lane 1
        while True:
            last_point = lane_boundary1_points[-1]
            window_y_min = last_point[1] + 1
            window_y_max = min(last_point[1] + window_height, cut_size - 1)
            window_x_min = max(last_point[0] - window_width // 2, 0)
            window_x_max = min(last_point[0] + window_width // 2, 95)
            lane_1_windows.append((window_x_min, window_x_max, window_y_min, window_y_max))

            # find maxima in the window
            window_maxima = [maximum for maximum in maxima if
                             window_x_min <= maximum[0] <= window_x_max and
                             window_y_min <= maximum[1] <= window_y_max]

            if len(window_maxima) == 0:
                break

            # choose the closest maxima to the center of the window
            center_x = (window_x_min + window_x_max) / 2
            distances = [abs(maximum[0] - center_x) for maximum in window_maxima]
            min_distance_index = np.argmin(distances)
            lane_boundary1_points = np.vstack((lane_boundary1_points, window_maxima[min_distance_index]))
            maxima.remove(window_maxima[min_distance_index])
        # lane 2
        while True:
            last_point = lane_boundary2_points[-1]
            window_y_min = last_point[1] + 1
            window_y_max = min(last_point[1] + window_height, cut_size - 1)
            window_x_min = max(last_point[0] - window_width // 2, 0)
            window_x_max = min(last_point[0] + window_width // 2, 95)
            lane_2_windows.append((window_x_min, window_x_max, window_y_min, window_y_max))

            # find maxima in the window
            window_maxima = [maximum for maximum in maxima if
                             window_x_min <= maximum[0] <= window_x_max and
                             window_y_min <= maximum[1] <= window_y_max]

            if len(window_maxima) == 0:
                break

            # choose the closest maxima to the center of the window
            center_x = (window_x_min + window_x_max) / 2
            distances = [abs(maximum[0] - center_x) for maximum in window_maxima]
            min_distance_index = np.argmin(distances)
            lane_boundary2_points = np.vstack((lane_boundary2_points, window_maxima[min_distance_index]))
            maxima.remove(window_maxima[min_distance_index])

        self.window_history.append((gradient_sum, lane_1_windows, lane_2_windows))
        lane_boundary1_points = remove_duplicates(lane_boundary1_points)
        lane_boundary2_points = remove_duplicates(lane_boundary2_points)
        return lane_boundary1_points, lane_boundary2_points, lane_found

    def plot_window_gift(self, filename):
        pil_images = []
        for gradient_sum, lane_1_windows, lane_2_windows in self.window_history:
            img = ((1-gradient_sum) * 255).astype(np.uint8)
            pil_image = Image.fromarray(img).convert("RGB")
            draw = ImageDraw.Draw(pil_image)

            for window in lane_1_windows:
                draw.rectangle([window[0], window[2], window[1], window[3]], outline="red")
            #
            # for window in lane_2_windows:
            #     draw.rectangle([window[0], window[2], window[1], window[3]], outline="blue")

            pil_images.append(pil_image.convert("P"))

        pil_images[0].save(
            filename,
            save_all=True,
            append_images=pil_images[1:],
            duration=200,
            loop=0
        )

class LaneFindingContourMethod:
    def __init__(self):
        self.color_dist = {'Lane': {'Lower': np.array([0, 0, 90]), 'Upper': np.array([10, 10, 120])}}
        self.min_area = 20
        self.lane_area_min = 600
        self.lane_area_max = 5000
        self.img = None
        self.img_HSV = None
        self.lane_mask = None
        self.img_idx = 0

    def to_hsv(self, img):
        pil_img = Image.fromarray(img)
        pil_img_hsv = pil_img.convert("HSV")
        img_hsv = np.array(pil_img_hsv)
        return img_hsv

    def find_lane_mask(self, img):
        self.img = img
        self.img_HSV = self.to_hsv(self.img)
        lower = self.color_dist['Lane']['Lower']
        upper = self.color_dist['Lane']['Upper']
        lane_mask_bool = (
                (self.img_HSV[..., 0] >= lower[0]) & (self.img_HSV[..., 0] <= upper[0]) &
                (self.img_HSV[..., 1] >= lower[1]) & (self.img_HSV[..., 1] <= upper[1]) &
                (self.img_HSV[..., 2] >= lower[2]) & (self.img_HSV[..., 2] <= upper[2])
        )
        lane_mask = lane_mask_bool.astype(np.uint8)
        self.lane_mask = lane_mask
        # flip the mask for correct orientation - upside down
        for row in range(lane_mask.shape[0] // 2):
            temp = np.copy(lane_mask[row, :])
            lane_mask[row, :] = lane_mask[lane_mask.shape[0] - 1 - row, :]
            lane_mask[lane_mask.shape[0] - 1 - row, :] = temp
        # self.save_lane_mask()
        return lane_mask, None, None

    def save_lane_mask(self):
        plt.imshow(self.lane_mask, cmap='gray')
        plt.axis('off')
        plt.xlim((-0.5,95.5))
        plt.ylim((-0.5,64.5))
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.gca().axes.get_yaxis().set_visible(False)
        plt.savefig(f'lane_mask_{self.img_idx:04d}.png', bbox_inches='tight', pad_inches=0)
        self.img_idx += 1

    def find_lanes(self, img):
        # Label connected components (equivalent to RETR_EXTERNAL contours)
        structure = np.ones((3,3), dtype=np.uint8)  # 8-connectivity
        labels, num_labels = ndimage.label(lane_mask, structure=structure)

        # Process each labeled region
        for i in range(1, num_labels + 1):  # Skip background (label 0)
            # Create mask for this specific region
            region_mask = (labels == i)
            area = np.sum(region_mask)
            if area < self.min_area:
                # Fill with black (0) - remove small contours
                lane_mask[region_mask] = 0
            elif self.lane_area_min <= area <= self.lane_area_max:
                # Fill with white (255) - keep valid lane contours
                lane_mask[region_mask] = 1
            # else: leave unchanged (contours outside both ranges remain as-is)

        self.lane_mask = lane_mask
        return lane_mask, labels, num_labels
        plt.gcf().clear()
        lane_mask, labels, num_labels = self.find_lane_mask(img)
        plt.imshow(self.lane_mask, cmap='gray')
        for i in range(1, num_labels + 1):
            region_mask = (labels == i)
            area = np.sum(region_mask)
            if self.lane_area_min <= area <= self.lane_area_max:
                coords = np.column_stack(np.where(region_mask))
                plt.scatter(coords[:, 1], coords[:, 0], s=1, color='red' if i == 1 else 'blue')
        plt.axis('off')
        plt.xlim((-0.5,95.5))
        plt.ylim((-0.5,64.5))
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.gca().axes.get_yaxis().set_visible(False)
        plt.savefig('contour_debug.png', bbox_inches='tight', pad_inches=0)

class LaneDetectionBase:
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

        self.smoother_lane1 = SplineRefiner()
        self.smoother_lane2 = SplineRefiner()
        self.kwargs = kwargs

        self.gradient_frames = []
        if self.kwargs.get('sliding_window', False):
            self.FindLanes = LaneFindingSlidingWindow()
        else:
            self.FindLanes = LaneFindingVanilla()
        self.testMethod = LaneFindingContourMethod()

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
        out, _, _ = self.testMethod.find_lane_mask(cut_image)
        return out



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
        gradient_sum = ( gradient_x + gradient_y ) * 255.0
        gradient_sum[gradient_sum < self.gradient_threshold / 255] = 0
        gradient_sum[gradient_sum >= self.gradient_threshold / 255] = int(1)
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
        # self.testMethod.find_lanes(state_image_full)
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
            lane_boundary1_points, lane_boundary2_points, lane_found = self.FindLanes.find_lanes_from_maxima(gradient_sum, self)
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

        # lane1 and lane2 swap check
        if self.lane_boundary1_old[1] is not None and self.lane_boundary2_old[1] is not None:
            t_test = np.linspace(0, 1, 6)
            lane1_points_test = np.array(splev(t_test, lane_boundary1))
            lane2_points_test = np.array(splev(t_test, lane_boundary2))
            old_lane1_points_test = np.array(splev(t_test, self.lane_boundary1_old))
            old_lane2_points_test = np.array(splev(t_test, self.lane_boundary2_old))

            d1_p1 = np.linalg.norm(lane1_points_test - old_lane1_points_test)
            d1_p2 = np.linalg.norm(lane1_points_test - old_lane2_points_test)
            d2_p1 = np.linalg.norm(lane2_points_test - old_lane1_points_test)
            d2_p2 = np.linalg.norm(lane2_points_test - old_lane2_points_test)
            if d1_p2 + d2_p1 < d1_p1 + d2_p2:
                lane_boundary1, lane_boundary2 = lane_boundary2, lane_boundary1

        self.lane_boundary1_old = lane_boundary1
        self.lane_boundary2_old = lane_boundary2

        # output the spline
        return lane_boundary1, lane_boundary2

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

    def plot_window_gift(self, filename):
        if self.kwargs.get('sliding_window', False):
            self.FindLanes.plot_window_gift(filename)

# read the config file and set the LaneDetection class accordingly
config = get_config()
if config['lane_detection']['module'] == 'vanilla':
    lane_detection_params = config['lane_detection']['params']
    LaneDetection = lambda: LaneDetectionBase(**lane_detection_params)
else:
    raise ValueError('Unknown lane detection module')

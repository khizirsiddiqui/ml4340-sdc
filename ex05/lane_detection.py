import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.interpolate import splprep, splev
from scipy.optimize import minimize
import time
from PIL import Image

from utils import remove_duplicates

class LaneDetection:
    '''
    Lane detection module using edge detection and b-spline fitting

    args: 
        cut_size (cut_size=65) cut the image at the front of the car
        spline_smoothness (default=0.1) multiplied by number of lane points
        gradient_threshold (default=14)
        distance_maxima_gradient (default=3)

    '''

    def __init__(self, cut_size=65, spline_smoothness=0.1, gradient_threshold=14, distance_maxima_gradient=3):
        self.car_position = np.array([48,0])
        self.spline_smoothness = spline_smoothness
        self.cut_size = cut_size
        self.gradient_threshold = gradient_threshold
        self.distance_maxima_gradient = distance_maxima_gradient
        self.lane_boundary1_old = 0
        self.lane_boundary2_old = 0


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
        return gradient_sum


    def find_maxima_gradient_rowwise(self, gradient_sum):
        '''
        This function should output arguments of local maxima for each row of the gradient image.
        You can use scipy.signal.find_peaks to detect maxima. 
        Hint: Use distance argument for a better robustness.

        input:
            gradient_sum 65x96x1

        output:
            maxima (np.array) 2x Number_maxima

        '''
        argmaxima = []
        for row in range(self.cut_size):
            peaks, _ = find_peaks(gradient_sum[row], distance=self.distance_maxima_gradient)
            for peak in peaks:
                argmaxima.append([peak, row])
        return argmaxima


    def find_first_lane_point(self, gradient_sum):
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
                A = np.argsort((argmaxima - self.car_position[0])**2)
                lane_boundary1_startpoint = np.array([[argmaxima[A[0]],  0]])
                lane_boundary2_startpoint = np.array([[argmaxima[A[1]],  0]])
                lanes_found = True

            row += 1
            
            # if no lane_boundaries are found
            if row == self.cut_size:
                lane_boundary1_startpoint = np.array([[0,  0]])
                lane_boundary2_startpoint = np.array([[0,  0]])
                break

        return lane_boundary1_startpoint, lane_boundary2_startpoint, lanes_found


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

        # to gray
        gray_state = self.cut_gray(state_image_full)
        # debug:
        # plt.imshow(gray_state[::-1].squeeze(), cmap='gray')
        # edge detection via gradient sum and thresholding
        gradient_sum = self.edge_detection(gray_state)
        # debug:
        # plt.imshow(gradient_sum[::-1].squeeze(), cmap='gray')
        maxima = self.find_maxima_gradient_rowwise(gradient_sum)

        # first lane_boundary points
        lane_boundary1_points, lane_boundary2_points, lane_found = self.find_first_lane_point(gradient_sum)

        # if no lane was found,use lane_boundaries of the preceding step
        if lane_found:
            
            #  in every iteration: 
            # 1- find maximum/edge with the lowest distance to the last lane boundary point 
            # 2- append maxium to lane_boundary1_points or lane_boundary2_points
            # 3- delete maximum from maxima
            # 4- stop loop if there is no maximum left 
            #    or if the distance to the next one is too big (>=100)
            too_far = 0
            while len(maxima) or too_far == 2:
                # lane_boundary 1
                last_point = lane_boundary1_points[-1]
                distances = [np.linalg.norm(np.array(maximum) - last_point) for maximum in maxima]
                min_distance_index = np.argmin(distances)
                if distances[min_distance_index] < 100:
                    lane_boundary1_points = np.vstack((lane_boundary1_points, maxima[min_distance_index]))
                    maxima.pop(min_distance_index)
                else:
                    too_far += 1

                if not len(maxima):
                    break

                # lane_boundary 2
                last_point = lane_boundary2_points[-1]
                distances = [np.linalg.norm(np.array(maximum) - last_point) for maximum in maxima]
                min_distance_index = np.argmin(distances)
                if distances[min_distance_index] < 100:
                    lane_boundary2_points = np.vstack((lane_boundary2_points, maxima[min_distance_index]))
                    maxima.pop(min_distance_index)
                else:
                    too_far += 1
                if too_far == 2: break
                too_far = 0
            lane_boundary1_points = remove_duplicates(lane_boundary1_points)
            lane_boundary2_points = remove_duplicates(lane_boundary2_points)

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

        self.lane_boundary1_old = lane_boundary1
        self.lane_boundary2_old = lane_boundary2

        # output the spline
        return lane_boundary1, lane_boundary2


    def plot_state_lane(self, state_image_full, steps, fig, waypoints=[]):
        '''
        Plot lanes and way points
        '''
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

import sys

sys.path.append('..')
from math import atan2, acos, sqrt, sin, cos

# TODO change geometric libs to shapely
from StateReader.cv_utils import Rectangle
from Utils.point2D import Point2D


class SimpleTrajectoryPlanner:
    """a simple trajectory planner, reimplementation of the java code"""

    def __init__(self):

        # for calculating reference point
        self.X_OFFSET = 0.45
        self.Y_OFFSET = 0.35
        self.scale_factor = 2.7
        # STRETCH factor for shooting a bird relative to the sling size
        self.STRETCH = 0.4
        self.X_MAX = 640

        # unit velocity
        self._velocity = 9.5 / self.scale_factor

        # conversion between the trajectory time and actual time in milliseconds
        self._time_unit = 815

        # boolean flag on set trajectory
        self._traj_set = False

        # parameters of the set trajectory
        self._release = None
        self._theta = None
        self._ux = None
        self._uy = None
        self._a = None
        self._b = None

        # the trajectory points
        self._trajectory = []

        # reference point and current scale
        self._ref = None
        self._scale = None

    def get_y_coordinate(self, sling, release_point, x):
        """Calculate the y-coordinate of a point on the set trajectory
        *
        * @param   sling - bounding rectangle of the slingshot
        *          release_point - point the mouse click is released from
        *          x - x-coordinate (on screen) of the requested point
        * @return  y-coordinate (on screen) of the requested point
        """
        self.set_trajectory(self, sling, release_point)

        # find the normalised coordinates
        xn = (x - self._ref.X) / self._scale

        return self._ref.Y - (int)((self._a * xn * xn + self._b * xn) * self._scale)

    def estimate_launch_point(self, slingshot, targetPoint):
        # calculate relative position of the target (normalised)
        scale = self.get_scene_scale(slingshot)
        # print ('scale ', scale)
        # System.out.println("scale " + scale)
        ref = self.get_reference_point(slingshot)
        x = (targetPoint.X - ref.X)
        y = -(targetPoint.Y - ref.Y)

        # gravity
        g = 0.48 * 9.81 / self.scale_factor * scale

        # launch speed
        v = self._velocity * scale
        #        print ('launch speed ', v)
        pts = []

        solution_existence_factor = v ** 4 - g ** 2 * x ** 2 - 2 * y * g * v ** 2

        # the target point cannot be reached
        if solution_existence_factor < 0:
            return None

        # solve cos theta from projectile equation

        cos_theta_1 = sqrt(
            (x ** 2 * v ** 2 - x ** 2 * y * g + x ** 2 * sqrt(v ** 4 - g ** 2 * x ** 2 - 2 * y * g * v ** 2)) / (
                        2 * v ** 2 * (x ** 2 + y ** 2)))
        cos_theta_2 = sqrt(
            (x ** 2 * v ** 2 - x ** 2 * y * g - x ** 2 * sqrt(v ** 4 - g ** 2 * x ** 2 - 2 * y * g * v ** 2)) / (
                        2 * v ** 2 * (x ** 2 + y ** 2)))
        #        print ('cos_theta_1 ', cos_theta_1, ' cos_theta_2 ', cos_theta_2)

        distance_between = sqrt(x ** 2 + y ** 2)  # ad-hoc patch

        theta_1 = acos(cos_theta_1) + distance_between * 0.0001  # compensate the rounding error
        theta_2 = acos(cos_theta_2) + distance_between * 0.00005  # compensate the rounding error
        pts.append(self.find_release_point(slingshot, theta_1))
        pts.append(self.find_release_point(slingshot, theta_2))

        return pts

    def get_release_angle(self, sling, release_point):
        """get release angle"""

        ref = self.get_reference_point(sling)
        return -atan2(ref.Y - release_point.Y, ref.X - release_point.X)

    def get_time_by_distance(self, sling, release, tap_point):
        """* the estimated tap time given the tap point
         *
         * @param   sling - bounding box of the slingshot
         *          release - point the mouse clicked was released from
         *          tap_point - point the tap should be made
         * @return  tap time (relative to the release time) in milli-seconds
         *
        """
        # update trajectory parameters
        self.set_trajectory(sling, release)

        pullback = self._scale * self.STRETCH * cos(self._theta)
        distance = (tap_point.X - self._ref.X + pullback) / self._scale

        # print("distance " , distance)
        # print("velocity " , self._ux)

        return (int)(distance / self._ux * self._time_unit)

    def set_trajectory(self, sling, release_point):
        """ Choose a trajectory by specifying the sling location and release point
         * Derive all related parameters (angle, velocity, equation of the parabola, etc)
         *
         * @param   sling - bounding rectangle of the slingshot
         *          release_point - point where the mouse click was released from
         *
        """

        # don't update parameters if the ref point and release point are the same
        if self._traj_set and self._ref != None and self._ref == self.get_reference_point(sling) and \
                self._release != None and self._release == release_point:
            return

        # set the scene parameters
        self._scale = sling.height + sling.width
        self._ref = self.get_reference_point(sling)

        # set parameters for the trajectory
        self._release = Point2D(release_point.X, release_point.Y)

        # find the launch angle
        self._theta = atan2(self._release.Y - self._ref.Y, self._ref.X - self._release.X)

        # work out initial velocities and coefficients of the parabola
        self._ux = self._velocity * cos(self._theta)
        self._uy = self._velocity * sin(self._theta)
        self._a = -0.5 / (self._ux * self._ux)
        self._b = self._uy / self._ux

        # work out points of the trajectory
        for x in range(0, self.X_MAX):
            xn = x / self._scale
            y = self._ref.Y - (int)((self._a * xn * xn + self._b * xn) * self._scale)
            self._trajectory.append(Point2D(x + self._ref.X, y))

        # turn on the setTraj flag
        self._traj_set = True

    def find_release_point(self, sling, theta):
        """find the release point given the sling location and launch angle, using maximum velocity
         *
         * @param   sling - bounding rectangle of the slingshot
         *          theta - launch angle in radians (anticlockwise from positive direction of the x-axis)
         * @return  the release point on screen
         *
        """

        mag = sling.height * 5
        # print('mag ', mag)
        ref = self.get_reference_point(sling)
        # print('ref ', ref)
        # print('cos theta ',cos(theta))
        #        print('sin theta ',sin(theta))
        release = Point2D(int(ref.X - mag * cos(theta)), int(ref.Y + mag * sin(theta)))

        return release

    def find_release_point_partial_power(self, sling, theta, v_portion):
        """find the release point given the sling location, launch angle and velocity
         *
         * @param   sling - bounding rectangle of the slingshot
         *          theta - launch angle in radians (anticlockwise from positive direction of the x-axis)
         *          v_portion - exit velocity as a proportion of the maximum velocity (maximum self.STRETCH)
         * @return  the release point on screen
         *
        """
        mag = self.get_scene_scale(sling) * self.STRETCH * v_portion
        ref = self.get_reference_point(sling)
        release = Point2D((int)(ref.X - mag * cos(theta)), (int)(ref.Y + mag * sin(theta)))

        return release

    def get_reference_point(self, sling):
        """find the reference point given the sling"""

        p = Point2D(int(sling.X + self.X_OFFSET * sling.width), int(sling.Y + self.Y_OFFSET * sling.width))
        return p

    def predictTrajectory(self, slingshot, launch_point):
        """predicts a trajectory"""
        self.set_trajectory(slingshot, launch_point)
        return self._trajectory

    def get_scene_scale(self, sling):
        """return scene scale determined by the sling size"""
        return sling.height + sling.width

    def find_active_bird(self, birds):
        """finds the active bird, i.e., the one in the slingshot"""
        # assumes that the active bird is the bird at the highest position
        activeBird = None
        for r in birds:
            if activeBird == None or activeBird.Y > r.Y:
                activeBird = r
        return activeBird

    def get_tap_time(self, sling, release, target, tap_interval):
        """finds the active bird, i.e., the one in the slingshot"""
        if tap_interval == 0:
            return 0

        distance = target.X - sling.X
        r = float(tap_interval) / 100
        tap_point = Point2D((int)(distance * r + sling.X), target.Y)
        return self.get_time_by_distance(sling, release, tap_point)


# test
if __name__ == "__main__":
    tp = SimpleTrajectoryPlanner()

    ys = [200, 250]
    xs = [40, 60]
    sling = Rectangle([ys, xs])
    target = Point2D(300, 100)
    traj_pts = tp.estimate_launch_point(sling, target)
    for pt in traj_pts:
        print(pt)

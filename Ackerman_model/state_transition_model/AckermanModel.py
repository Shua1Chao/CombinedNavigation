import numpy as np 
import math

class AckermanModel:

    def __init__(self, position, heading, wheel_base, front_tread,rear_tread, max_steer, min_velocity, max_velocity, drive="rear", tyre_radius=0.5):

        self.rear_position = position   # 后轮中心坐标
        self.heading = heading # 拖拉机朝向(COSθ,SINθ)
        self.wheel_base = wheel_base # 车身长度
        self.front_tread = front_tread # 前轮间距
        self.rear_tread = rear_tread # 后轮间距
        self.max_steer = max_steer # 最大转弯角度
        self.min_velocity = min_velocity # 最小速度

        self.drive = drive
        self.tyre_radius = tyre_radius
        self.max_velocity = max_velocity

        self.front_position = self.rear_position + self.heading * self.wheel_base   # Center of front axel
        self.normal = np.array([self.heading[1], -self.heading[0]])     # Right vector of car (sinθ,-cosθ)

        self.front_left_position = self.front_position - self.normal * self.front_tread/2
        self.front_right_position = self.front_position + self.normal * self.front_tread/2
        self.rear_left_position = self.rear_position - self.normal * self.rear_tread/2
        self.rear_right_position = self.rear_position + self.normal * self.rear_tread/2


        left_tyre_p1 = self.front_left_position + self.heading * self.tyre_radius
        left_tyre_p2 = self.front_left_position - self.heading * self.tyre_radius

        right_tyre_p1 = self.front_right_position + self.heading * self.tyre_radius
        right_tyre_p2 = self.front_right_position - self.heading * self.tyre_radius

        self.front_left_tyre = [left_tyre_p1, left_tyre_p2]
        self.front_right_tyre = [right_tyre_p1, right_tyre_p2]

        left_tyre_p1 = self.rear_left_position + self.heading * self.tyre_radius
        left_tyre_p2 = self.rear_left_position - self.heading * self.tyre_radius

        right_tyre_p1 = self.rear_right_position + self.heading * self.tyre_radius
        right_tyre_p2 = self.rear_right_position - self.heading * self.tyre_radius

        self.rear_left_tyre = [left_tyre_p1, left_tyre_p2]
        self.rear_right_tyre = [right_tyre_p1, right_tyre_p2]

    def _line_intersection(self, line1, line2):
        xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
        ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

        def det(a, b):
            return a[0] * b[1] - a[1] * b[0]

        div = det(xdiff, ydiff)
        if div == 0:
            return None, None

        d = (det(*line1), det(*line2))
        x = det(d, xdiff) / div
        y = det(d, ydiff) / div
        return x, y

    def update(self, steering_angle, step_size):

        if steering_angle < -self.max_steer:
            steering_angle = -self.max_steer
        elif steering_angle > self.max_steer:
            steering_angle = self.max_steer

        if step_size < self.min_velocity:
            step_size = self.min_velocity
        elif step_size > self.max_velocity:
            step_size = self.max_velocity

        if self.drive == "rear":
            heading_angle = math.degrees(math.atan2(self.heading[1], self.heading[0])) # Absolute heading angle
            if heading_angle < 0:
                heading_angle += 360


            steering_angle_abs = heading_angle - steering_angle # Absolute steering angle
            if steering_angle_abs < 0:
                steering_angle_abs += 360

            heading_abs_vector = self.rear_position + np.array([self.heading[1], -self.heading[0]])
            steering_abs_vector = self.front_position + np.array([math.sin(math.radians(steering_angle_abs)), -math.cos(math.radians(steering_angle_abs))])

            x, y = self._line_intersection((self.rear_position, heading_abs_vector), (self.front_position,steering_abs_vector))

            if not x == None:
                arc_center = np.array([x, y])

                R = np.linalg.norm(arc_center - self.front_position)
                r = np.linalg.norm(arc_center - self.rear_position)
                theta_step = math.degrees(step_size/r)

                rear_base_arc_vector = self.rear_position - arc_center
                rear_base_arc_angle = math.degrees(math.atan2(rear_base_arc_vector[1], rear_base_arc_vector[0]))

                front_base_arc_vector = self.front_position - arc_center
                front_base_arc_angle = math.degrees(math.atan2(front_base_arc_vector[1], front_base_arc_vector[0]))

                if steering_angle < 0:
                    rear_arc_angle = rear_base_arc_angle + theta_step
                    front_arc_angle = front_base_arc_angle + theta_step
                elif steering_angle > 0:
                    rear_arc_angle = rear_base_arc_angle - theta_step
                    front_arc_angle = front_base_arc_angle - theta_step
                
                if not steering_angle is None and not steering_angle == 0:

                    new_rear_x = arc_center[0] + r * math.cos(math.radians(rear_arc_angle))
                    new_rear_y = arc_center[1] + r * math.sin(math.radians(rear_arc_angle))

                    new_front_x = arc_center[0] + R * math.cos(math.radians(front_arc_angle))
                    new_front_y = arc_center[1] + R * math.sin(math.radians(front_arc_angle))

                    self.rear_position = np.array([new_rear_x, new_rear_y])
                    self.front_position = np.array([new_front_x, new_front_y])
                    diff = self.front_position - self.rear_position
                    self.heading = diff/np.linalg.norm(diff)
                    self.normal = np.array([self.heading[1], -self.heading[0]])

                    self.front_left_position = self.front_position - self.normal * self.front_tread/2
                    self.front_right_position = self.front_position + self.normal * self.front_tread/2
                    self.rear_left_position = self.rear_position - self.normal * self.rear_tread/2
                    self.rear_right_position = self.rear_position + self.normal * self.rear_tread/2

                    left_turn_vector = self.front_left_position - arc_center
                    left_turn_vector = np.array([left_turn_vector[1], -left_turn_vector[0]])
                    left_turn_vector = left_turn_vector/np.linalg.norm(left_turn_vector)
                    left_tyre_p1 = self.front_left_position + left_turn_vector * self.tyre_radius
                    left_tyre_p2 = self.front_left_position - left_turn_vector * self.tyre_radius

                    right_turn_vector = self.front_right_position - arc_center
                    right_turn_vector = np.array([right_turn_vector[1], -right_turn_vector[0]])
                    right_turn_vector = right_turn_vector/np.linalg.norm(right_turn_vector)
                    right_tyre_p1 = self.front_right_position + right_turn_vector * self.tyre_radius
                    right_tyre_p2 = self.front_right_position - right_turn_vector * self.tyre_radius

            if x == None or steering_angle == 0:
                self.rear_position += self.heading * step_size
                self.front_position += self.heading * step_size
                self.front_left_position += self.heading * step_size
                self.front_right_position += self.heading * step_size
                self.rear_left_position += self.heading * step_size
                self.rear_right_position += self.heading * step_size

                left_tyre_p1 = self.front_left_position + self.heading * self.tyre_radius
                left_tyre_p2 = self.front_left_position - self.heading * self.tyre_radius

                right_tyre_p1 = self.front_right_position + self.heading * self.tyre_radius
                right_tyre_p2 = self.front_right_position - self.heading * self.tyre_radius

            self.front_left_tyre = [left_tyre_p1, left_tyre_p2]
            self.front_right_tyre = [right_tyre_p1, right_tyre_p2]

            left_tyre_p1 = self.rear_left_position + self.heading * self.tyre_radius
            left_tyre_p2 = self.rear_left_position - self.heading * self.tyre_radius

            right_tyre_p1 = self.rear_right_position + self.heading * self.tyre_radius
            right_tyre_p2 = self.rear_right_position - self.heading * self.tyre_radius

            self.rear_left_tyre = [left_tyre_p1, left_tyre_p2]
            self.rear_right_tyre = [right_tyre_p1, right_tyre_p2]


    def get_axel_corners(self):

        corners = [self.front_left_position, self.front_right_position, self.rear_right_position, self.rear_left_position]
        return corners
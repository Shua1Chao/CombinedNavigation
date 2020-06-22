from cv2 import cv2
import numpy as np
from math import acos,asin,pi

from state_transition_model.AckermanModel import AckermanModel
from draw_utils.draw_tools import draw_ackerman_model

def touch(car,jiange,flag):
    if flag:
        if car.front_right_position[1] >= jiange - 0.2 or car.front_left_position[0] <= -5.8 or car.front_right_position[0] <= -5.8:
            return True
        else:
            return False
    else:
        if car.rear_left_position[1] <= 0.2 or car.rear_right_position[1] >= jiange - 0.2 \
                or ((car.front_right_position[0] < car.front_left_position[0]) and car.front_left_position[0] > -3.2):
            return True
        else:
            return False


if __name__ == "__main__":

    display = np.ones((720, 1280, 3), dtype=np.uint8) * 255
    center = np.array([display.shape[1] / 2, display.shape[0] / 2]).astype(np.int32)  # Pixel space world center
    mtp_ratio = 20.0  # Meter to pixel scale factor

    # Time calculations
    delay_milliseconds = 100
    dt = delay_milliseconds / 1000

    init_position = np.array([0.0, 0.0])
    init_heading = np.array([0.0, 1.0])
    wheel_base = 4.35
    front_tread = 1.562
    rear_tread = 2.12
    max_steer_angle = 45
    min_velocity = -10.0 * dt
    max_velocity = 55.55 * dt

    for steer_angle in range(-45, -30):
        car = AckermanModel(init_position, init_heading, wheel_base, front_tread,rear_tread, max_steer_angle, min_velocity, max_velocity)

    # draw_ackerman_model(display, center, car, mtp_ratio)

    # cv2.imshow('image', cv2.flip(display, 0))
    # key = cv2.waitKey(0)
        print(steer_angle)
        velocity = 0.2
        jiange = 2 + wheel_base


        flag = True
        times = 0
        while car.heading[1] + 1 >= 0.001:
            print(asin(car.heading[1]))
            if touch(car,jiange,flag):
                if flag:
                    flag = False
                    car.update(-steer_angle,-velocity)
                else:
                    flag = True
                    car.update(steer_angle,velocity)
                times = times + 1
            else:
                if flag:
                    car.update(steer_angle,velocity)
                else:
                    car.update(-steer_angle,-velocity)
            # print(car.front_right_position[1])



            # if flag and car.front_right_position[1] <= jiange:
            #     car.update(steer_angle, velocity)
            # elif flag and car.front_right_position[1] >= jiange:
            #     car.update(-steer_angle,-velocity)
            #     flag = False
            #     times = times + 1
            # elif not flag and car.rear_left_position[1] > 0:
            #     car.update(-steer_angle, -velocity)
            # elif not flag and car.rear_left_position[1] <= 0:
            #     car.update(steer_angle, velocity)
            #     flag = True
            # #     times = times + 1
            display = np.ones((720, 1280, 3), dtype=np.uint8) * 255
            draw_ackerman_model(display, center, car, mtp_ratio)

            cv2.imshow('image', cv2.flip(display, 0))
            key = cv2.waitKey(10)
        print(car.heading[1])
        print(times,asin(car.heading[1])+ pi/2)
            # Drawing car axel frame

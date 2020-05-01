import cv2
import numpy as np
import matplotlib.pyplot as plt

# the dimension of the output image
IMAGE = (500, 500, 3)
# centre coordinate of the circle
CIRCLE_CENTRE = (250, 250)
# radius of the circle
CIRCLE_RADIUS = 100
# minimum and maximum are used to calculate
# random points. These are the minimum X and
# Y coordinates of the square that will be
# used to calculate random points
MIN = 150
MAX = 350
# top left coordinate of the square
RECT_TL = (MIN, MIN)
# bottom right coordinate of the square
RECT_BR = (MAX, MAX)
# color code of various colors (required by
# OpenCV)
RED = (0, 0, 255)
BLACK = (0, 0, 0)
BLUE = (255, 0, 0)
GREEN = (0, 255, 0)


def in_circle(x, y):
    """
    Calculates if the point is inside or outside the
    circle using the point coordinate equation
    Args:
        x: x coordinate
        y: y coordinate

    Returns:
        True if the point is inside the circle of radius
        `CIRCLE_RADIUS`
    """
    return (x - CIRCLE_CENTRE[0])**2 \
           + (y - CIRCLE_CENTRE[1])**2 < CIRCLE_RADIUS ** 2


def get_random_points():
    """
    Calculates a random coordinate using the uniform distribution
    Returns:
        (rand_x, rand_y): tuple containing the x, y coordinates
        bool: True if the point is inside the circle of radius
            `CIRCLE_RADIUS`
    """
    rand_x = np.random.randint(MIN, MAX)
    rand_y = np.random.randint(MIN, MAX)
    return (rand_x, rand_y), in_circle(rand_x, rand_y)


def draw_random_point(frame):
    """
    Draws a random point in an OpenCV frame (which is a numpy array)
    The point will have a green color if it's inside the circle, else
    it will be red.
    Args:
        frame: input frame (numpy array)

    Returns:
        frame: the numpy array with the point drawn
        bool: True if the point is inside the circle of radius
            `CIRCLE_RADIUS`
    """
    (x, y), inside = get_random_points()
    color = GREEN if inside else BLACK
    frame = cv2.circle(frame, (x, y), 1, thickness=2, color=color)
    return frame, inside


def base_frame():
    """
    Creates a base frame with circle and square drawn.
    Returns:
        frame: a numpy array with circle and square drawn.
    """
    frame = np.zeros(IMAGE) + 255
    frame = cv2.circle(frame, CIRCLE_CENTRE, 100, color=BLUE, thickness=2)
    frame = cv2.rectangle(frame, RECT_TL, RECT_BR, color=RED, thickness=2)
    return frame


def render_drawing(frame, points, inside, pi_value):
    """
    Renders a drawing with the circle, square, and the various points
    drawn using the Monte Carlo simulation.

    Args:
        frame: the base frame
        points: the number of points drawn
        inside: number of points inside circle
        pi_value: approximate value of pie calculated after
            Monte Carlo simulation.
    """
    total_txt = f"Total number of points: {points}"
    count_txt = f"Number of points in circle: {inside}"
    pi_txt = f"Pi approximation: {pi_value}"
    cv2.putText(frame, total_txt, (20, 40), fontFace=1,
                color=BLACK, fontScale=1)
    cv2.putText(frame, count_txt, (20, 60), fontFace=1,
                color=BLACK, fontScale=1)
    cv2.putText(frame, pi_txt, (20, 80), fontFace=1,
                color=BLACK, fontScale=1)
    cv2.imshow('Pi Approximation', frame)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        cv2.destroyAllWindows()


def calculate_pi(render=False, points=200):
    """
    Calculates the approximate value of pi using Monte Carlo
    Simulation.

    Args:
        render: If the result should be rendered
        points: the number of points to draw

    Returns:
        approximate value of pi
    """
    frame = base_frame()
    inside = 0
    for i in range(points):
        frame, is_inside = draw_random_point(frame)
        inside += is_inside
    approx_pi = (float(inside) / points) * 4
    if render:
        render_drawing(frame, points, inside, approx_pi)
    return approx_pi


def draw_histogram(number_points=200, num_iterations=200):
    """
    Draws the histogram of the calculated PI variables

    Args:
        number_points: number of points to draw while approximating pi
        num_iterations: number of Monte Carlo iterations to run
    """
    pis = []
    for i in range(num_iterations):
        pis.append(calculate_pi(False, number_points))
    plt.hist(pis, bins=35, edgecolor='k')
    plt.title(f"Histogram of pi values. "
              f"Mean = {np.mean(pis):.2f} Std = {np.std(pis):.2f}")
    plt.xlabel("Pi values")
    plt.xlim((2.8, 3.5))
    plt.ylabel("Number of instances")
    plt.show()


if __name__ == '__main__':
    draw_histogram(number_points=1000, num_iterations=1000)

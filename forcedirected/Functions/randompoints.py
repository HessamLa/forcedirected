import numpy as np

def generate_random_points_on_sphere(n, d, sphere_radius=1.0):
    '''
    Generates n random points in on surface of a d-dimensional sphere with radius sphere_size.
    '''
    P = np.random.normal(0, 1, size=(n, d))
    # Normalize the points to have unit norm
    norms = np.linalg.norm(P, axis=1, keepdims=True)
    P = P / norms * sphere_radius
    return P
    
def generate_random_points_normal(n, d, mean=0, s=1):
    '''
    Generates n random points in a d-dimensional, normally distributed around the mean point.
    '''
    # Generate random points from a standard normal distribution
    P = np.random.normal(0, 1, size=(n, d))
    # Normalize the points to have unit norm
    norms = np.linalg.norm(P, axis=1, keepdims=True)
    P = P / norms
    # Scale the points to have the desired standard deviation
    distances = np.random.normal(0, s, size=(n, 1))
    P *= distances
    P = P + mean
    return P

def generate_random_points(n, d, mean=0, standard_dev=1):
    """
    Generates n random points in a d-dimensional space with normal distribution along each axis.
    """
    P = np.random.normal(mean, standard_dev, size=(n, d))
    return P
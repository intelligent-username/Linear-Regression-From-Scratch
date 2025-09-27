# Naive implementation for simple amount of points for now
# Small number of points and only (x,y)

def linear_regression(points: list[tuple[float, float]]):
    """
    Performs simple linear regression using the least squares method.
    
    Preconditions:
        points: List of (x, y) coordinate tuples representing data points
        
    Returns:
        tuple: (slope, y_intercept) of the best-fit line y = mx + b
        
    Raises:
        ValueError: If all x values are identical (no variance in x)
        ValueError: If fewer than 2 points are provided
        
    Note:
        This is a basic implementation that will be enhanced soon:
          - Gradient Descent will be added
          - And other optimization techniques
    """
    if len(points) < 2:
        raise ValueError("At least 2 points are required for linear regression")
    
    sum_x = 0
    sum_y = 0
    sum_xy = 0
    sum_x2 = 0

    for x, y in points:
        sum_x += x
        sum_y += y
        sum_x2 += x * x
        sum_xy += x * y

    
    length = len(points)
    avg_x = sum_x / length
    avg_y = sum_y / length
    
    # the covariance between x and y divided by the variance of x.
    denominator = sum_x2 - length * avg_x * avg_x
    
    if denominator == 0:
        raise ValueError("All x values are identical - cannot perform linear regression")

    m = (sum_xy - length * avg_x * avg_y) / denominator
    b = avg_y - m * avg_x

    
    return (m, b)


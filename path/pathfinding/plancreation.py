def find_marked_points(coordinate_array, mark_value=1):
    """
    Find all points in the coordinate system marked with a specific value.
    
    Args:
        coordinate_array: 2D array with coordinate values
        mark_value: The value to search for (default: 1 for balls)
    
    Returns:
        List of (x, y) tuples for all marked points
    """
    marked_points = []
    
    for y in range(len(coordinate_array)):
        for x in range(len(coordinate_array[0])):
            if coordinate_array[y][x] == mark_value:
                marked_points.append((x, y))
    
    return marked_points


def create_lines_between_marked_points(coordinate_array, mark_value=1):
    """
    Create line segments connecting all marked points in the coordinate system.
    
    Args:
        coordinate_array: 2D array with coordinate values
        mark_value: The value to search for (default: 1 for balls)
    
    Returns:
        List of line segments, each line is a tuple ((x1, y1), (x2, y2))
    """
    marked_points = find_marked_points(coordinate_array, mark_value)
    lines = []
    
    # Connect each point to all other points
    for i in range(len(marked_points)):
        for j in range(i + 1, len(marked_points)):
            point1 = marked_points[i]
            point2 = marked_points[j]
            lines.append((point1, point2))
    
    return lines


def create_lines_nearest_neighbor(coordinate_array, mark_value=1):
    """
    Create line segments connecting marked points to their nearest neighbors.
    This creates a more sparse graph suitable for pathfinding.
    
    Args:
        coordinate_array: 2D array with coordinate values
        mark_value: The value to search for (default: 1 for balls)
    
    Returns:
        List of line segments, each line is a tuple ((x1, y1), (x2, y2))
    """
    marked_points = find_marked_points(coordinate_array, mark_value)
    lines = []
    connected = set()
    
    for point in marked_points:
        # Find nearest unconnected neighbor
        min_distance = float('inf')
        nearest = None
        
        for other_point in marked_points:
            if other_point == point:
                continue
            
            # Create edge tuple (always smaller point first for deduplication)
            edge = tuple(sorted([point, other_point]))
            if edge in connected:
                continue
            
            # Calculate Euclidean distance
            distance = ((point[0] - other_point[0])**2 + (point[1] - other_point[1])**2)**0.5
            
            if distance < min_distance:
                min_distance = distance
                nearest = other_point
        
        if nearest:
            edge = tuple(sorted([point, nearest]))
            connected.add(edge)
    
    return list(connected)


def plan():
    #Pathfinding and movement code here, using coordsystem as input



    pass
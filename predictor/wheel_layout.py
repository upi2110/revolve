# wheel_layout.py

# European roulette wheel layout (clockwise)
EU_WHEEL = [
    0, 32, 15, 19, 4, 21, 2, 25, 17, 34,
    6, 27, 13, 36, 11, 30, 8, 23, 10, 5,
    24, 16, 33, 1, 20, 14, 31, 9, 22, 18,
    29, 7, 28, 12, 35, 3, 26
]

def get_neighbors(number, n=4):
    """
    Returns a list of neighbors around the given number on the wheel.
    Includes the number itself plus n numbers on either side.
    """
    if number not in EU_WHEEL:
        raise ValueError("Number not found on European wheel.")
    
    index = EU_WHEEL.index(number)
    total = len(EU_WHEEL)
    neighbors = []
    for offset in range(-n, n + 1):
        neighbors.append(EU_WHEEL[(index + offset) % total])
    return neighbors

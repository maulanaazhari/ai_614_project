class EnvConfig01:
    start = (0, 0)
    goal = (9, 9)
    obstacles = [
        [-8.5, -8.5, 7.25, 7.25],
        [0.75, 0.75, 7.25, 7.25],
        [-8.5, 0.75, 7.25, 7.25],
        [0.75, -8.5, 7.25, 7.25]
    ]

class EnvConfig00:
    start = (0, 0)
    goal = (8.5, 8.5)
    obstacles = []

class EnvConfig02:
    start = (-5, -2)
    goal = (5, 2)
    obstacles = [
        [0, -7, 2, 14],
    ]
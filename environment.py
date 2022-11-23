
class Environment:
    ROW_COUNT = 10
    COL_COUNT = 10

    def __init__(self, person, goal) -> None:
        self.person = person
        self.goal = goal
        self.holes = {(1,1), (1,2), (2,2), (2,6), (5,5), (6,8)}

    def next_cell(self, x,y,action):
        assert x >= 0 and x < self.COL_COUNT, "x out of range"
        assert y >= 0 and y < self.ROW_COUNT, "y out of range"
        assert action in ['North', 'East', 'South', 'West', 'Down'], "invalid action"

        #if the person has fallen in the hole, then the person cannot move
        if action == "Down":
            return (x,y)
        
        if ( y >= 0 and y<= 3) and (x == 0 or x == 3 or x == 7) and action == 'East':
            return (x, y)
        if (y >= 6 and y<= 9) and (x == 2 or x == 7) and action == 'East':
            return (x, y)
        if (y >= 4 and y<= 7) and (x == 5) and action == 'East':
            return (x, y)
        if (y >= 0 and y<= 3) and (x == 1 or x == 4 or x == 8) and action == 'West':
            return (x, y)
        if (y >= 6 and y<= 9) and (x == 3 or x == 8) and action == 'West':
            return (x, y)
        if (y >= 4 and y<= 7) and (x == 6) and action == 'West':
            return (x, y)
        if action == 'North':
            if (y + 1 >= self.ROW_COUNT):
                return (x, y)
            return (x, y+1)
        elif action == 'East':
            if (x + 1 >= self.COL_COUNT):
                return (x, y)
            return (x+1, y)
        elif action == 'South':
            if (y - 1 < 0):
                return (x, y)
            return (x, y-1)
        elif action == 'West':
            if (x - 1 < 0):
                return (x, y)
            return (x-1, y)

class State:
    def __init__(self, curr_x, curr_y, goal_x, goal_y) -> None:
        self.curr_x = curr_x
        self.curr_y = curr_y
        self.goal_x = goal_x
        self.goal_y = goal_y

    def __str__(self) -> str:
        return f"({self.curr_x},{self.curr_y},{self.goal_x},{self.goal_y})"
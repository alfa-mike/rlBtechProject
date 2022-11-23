from filecmp import cmp
import matplotlib.pyplot as plt
import matplotlib.animation as ani
import numpy as np



# maze = [[2 for _ in range(10)] for _ in range(10)]

# plt.rcParams["figure.figsize"] = [7.00, 7.00]
# plt.rcParams["figure.autolayout"] = True

# fig, ax = plt.subplots()

# def update(i):
#     global maze
#     # im_normed = np.random.rand(10, 10)
#     if i==0:
#         maze[i][i]=4
#     else:
#         maze[i][i]=4
#         maze[i-1][i-1]=2
#     ax.imshow(maze,cmap="Greens")
#     # ax.set_axis_off()
#     ax.maze(True)

# anim = ani.FuncAnimation(fig, update, frames=10, interval=1000)

# plt.show()
maze = [
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1] 
        ]

array = [(4,0),(4,1),(4,2),(4,3),(5,3),(6,3),(6,4)]

plt.rcParams["figure.figsize"] = [7.00, 10.00]
plt.rcParams["figure.autolayout"] = True
# plt.grid()

fig, ax = plt.subplots()
# ax.set(aspect = 1,
#        xlim =(0, 10),
#        ylim =(0, 10))



def update(t):
    global maze
    global array
    if t==len(array):
        return 
    if t==0:
        idx_x,idx_y = 9-array[t][1],array[t][0]
        
        maze[idx_x][idx_y]=10
        maze[9-array[-1][1]][array[-1][0]] = 20
    
    else:
        prev_idx_x,prev_idx_y = 9-array[t-1][1],array[t-1][0]
        idx_x,idx_y = 9-array[t][1],array[t][0]

        maze[prev_idx_x][prev_idx_y]=1
        maze[idx_x][idx_y]=10
        maze[9-array[-1][1]][array[-1][0]] = 20

    ax.imshow(maze,cmap="Greens")
    # ax.set_axis_off()


anim = ani.FuncAnimation(fig, update, frames=len(array), interval=1000)
# ax.imshow(maze)

plt.show()


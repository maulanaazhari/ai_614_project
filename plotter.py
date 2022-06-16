import pickle
# import cv2
from tqdm import tqdm
import os
import concurrent.futures
# from rrt import RRT
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# filename = 'rrt_20220613-153154.pkl'

# plot_data = RRT.PlotData.load_pickle(filename)

# try:
#     os.mkdir(filename.split('.')[0])
# except:
#     pass

def plot_square(obs):
    plt.plot([obs[0], obs[0]+obs[2], obs[0]+obs[2], obs[0], obs[0]],
            [obs[1], obs[1], obs[1]+obs[3], obs[1]+obs[3], obs[1]],
            "-k")
        
def plot_circle(x, y, size, color="-b"):  # pragma: no cover
    deg = list(range(0, 360, 5))
    deg.append(0)
    xl = [x + size * math.cos(np.deg2rad(d)) for d in deg]
    yl = [y + size * math.sin(np.deg2rad(d)) for d in deg]
    plt.plot(xl, yl, color)

def draw_graph_and_save(
    filename,
    rnd,
    node_list,
    start,
    end,
    obstacles,
    i, 
    play_area,
    final_path,
    final_path_length,
    min_rand,
    max_rand
):

    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')

    for obs in obstacles:
        rect = patches.Rectangle((obs[0], obs[1]), obs[2], obs[3], linewidth=1, edgecolor='k', facecolor='k')
        # plot_square(obs)
        ax.add_patch(rect)
        
    if rnd is not None:
        plt.plot(rnd.x, rnd.y, "yo", markersize=2)
        # if self.robot_radius > 0.0:
        #     self.plot_circle(rnd.x, rnd.y, self.robot_radius, '-r')

    for node in node_list:
        if node.parent:
            plt.plot(node.path_x, node.path_y, "-b", linewidth=0.1)
        plt.plot(node.x, node.y, 'bo', markersize=0.5)

    if play_area is not None:
        plt.plot([play_area.xmin, play_area.xmax,
                play_area.xmax, play_area.xmin,
                play_area.xmin],
                [play_area.ymin, play_area.ymin,
                play_area.ymax, play_area.ymax,
                play_area.ymin],
                "-k")

    plt.plot(start.x, start.y, "xr")
    plt.plot(end.x, end.y, "xr")
    plt.axis([min_rand, max_rand, min_rand, max_rand])

    if final_path is not None:
        plt.plot([x for (x, y) in final_path], [y for (x, y) in final_path], '-g')
    plt.title("iter num: {}, path length: {}".format(i+1, round(final_path_length, 3) if final_path_length is not None else None))
    plt.savefig("{}/{}.png".format(filename.split('.')[0], i))
    plt.close()
    return i

# with concurrent.futures.ProcessPoolExecutor(max_workers=6) as executor:
#     results = [
#         executor.submit(
#             draw_graph_and_save,
#             plot_data.filename,
#             plot_data.data_list[i]['rnd'],
#             plot_data.data_list[i]['node_list'],
#             plot_data.start,
#             plot_data.end,
#             plot_data.obstacles,
#             plot_data.data_list[i]['i'],
#             plot_data.play_area,
#             plot_data.data_list[i]['final_path'],
#             plot_data.data_list[i]['final_path_length'],
#             plot_data.min_rand,
#             plot_data.max_rand
#         )
#         for i in range(len(plot_data))
#     ]

#     for f in concurrent.futures.as_completed(results):
#         print(f.result())

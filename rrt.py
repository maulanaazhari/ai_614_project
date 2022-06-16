"""

Path planning Sample Code with Randomized Rapidly-Exploring Random Trees (RRT)

author: AtsushiSakai(@Atsushi_twi)

"""
import time
import os
import math
import random
from tqdm import tqdm
import matplotlib
import copy
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np
show_animation = True
import pickle
from configs import *
from plotter import draw_graph_and_save
import concurrent.futures
import copy

class RRT:
    """
    Class for RRT planning
    """
    class Node:
        """
        RRT Node
        """

        def __init__(self, x, y):
            self.x = x
            self.y = y
            self.path_x = []
            self.path_y = []
            self.parent = None
            self.cost = 0

    class PlotData:
        def __init__(self, filename, start, end, min_rand, max_rand, play_area, obstacles):
            self.start = start
            self.end = end
            self.min_rand = min_rand
            self.max_rand = max_rand
            self.play_area = play_area
            self.data_list = []
            self.obstacles = obstacles
            self.filename = filename

        def add_data(self, dct):
            self.data_list.append(dct)

        def save(self):
            filehandler = open(self.filename, "wb")
            pickle.dump(self, filehandler)
            filehandler.close()

        @staticmethod
        def load_pickle(filename):
            filehandler = open(filename, "rb")
            obj = pickle.load(filehandler)
            obj.filename = filename
            filehandler.close()
            return obj

        def __len__(self):
            return len(self.data_list)


        def plot_square(self, obs):
            plt.plot([obs[0], obs[0]+obs[2], obs[0]+obs[2], obs[0], obs[0]],
                    [obs[1], obs[1], obs[1]+obs[3], obs[1]+obs[3], obs[1]],
                    "-k")
        
        @staticmethod
        def plot_circle(x, y, size, color="-b"):  # pragma: no cover
            deg = list(range(0, 360, 5))
            deg.append(0)
            xl = [x + size * math.cos(np.deg2rad(d)) for d in deg]
            yl = [y + size * math.sin(np.deg2rad(d)) for d in deg]
            plt.plot(xl, yl, color)

        def draw_graph(self, dct):
            # for stopping simulation with the esc key.
            for k,v in dct.items():
                setattr(self, k, v)
            rnd = self.rnd

            plt.clf()
            plt.gcf().canvas.mpl_connect(
                'key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])
            if rnd is not None:
                plt.plot(rnd.x, rnd.y, "yo", markersize=2)
                # if self.robot_radius > 0.0:
                #     self.plot_circle(rnd.x, rnd.y, self.robot_radius, '-r')

            for node in self.node_list:
                if node.parent:
                    plt.plot(node.path_x, node.path_y, "-b", linewidth=0.1)
                plt.plot(node.x, node.y, 'bo', markersize=0.5)

            for obs in self.obstacles:
                self.plot_square(obs)

            if self.play_area is not None:
                plt.plot([self.play_area.xmin, self.play_area.xmax,
                        self.play_area.xmax, self.play_area.xmin,
                        self.play_area.xmin],
                        [self.play_area.ymin, self.play_area.ymin,
                        self.play_area.ymax, self.play_area.ymax,
                        self.play_area.ymin],
                        "-k")

            plt.plot(self.start.x, self.start.y, "xr")
            plt.plot(self.end.x, self.end.y, "xr")
            plt.axis("equal")
            plt.axis([self.min_rand, self.max_rand, self.min_rand, self.max_rand])

            if self.final_path is not None:
                plt.plot([x for (x, y) in self.final_path], [y for (x, y) in self.final_path], '-g')
            plt.title("iter num: {}, path length: {}".format(self.i+1, round(self.final_path_length, 3) if self.final_path_length is not None else None))
            plt.pause(0.001)

        @staticmethod
        def draw_graph_and_save(dct):
            # for stopping simulation with the esc key.
            for k,v in dct.items():
                setattr(self, k, v)
            rnd = self.rnd

            fig = plt.figure()
            ax = fig.add_subplot(111, aspect='equal')
        
            if rnd is not None:
                plt.plot(rnd.x, rnd.y, "yo", markersize=2)
                # if self.robot_radius > 0.0:
                #     self.plot_circle(rnd.x, rnd.y, self.robot_radius, '-r')

            for node in self.node_list:
                if node.parent:
                    plt.plot(node.path_x, node.path_y, "-b", linewidth=0.1)
                plt.plot(node.x, node.y, 'bo', markersize=0.5)

            for obs in self.obstacles:
                self.plot_square(obs)

            if self.play_area is not None:
                plt.plot([self.play_area.xmin, self.play_area.xmax,
                        self.play_area.xmax, self.play_area.xmin,
                        self.play_area.xmin],
                        [self.play_area.ymin, self.play_area.ymin,
                        self.play_area.ymax, self.play_area.ymax,
                        self.play_area.ymin],
                        "-k")

            plt.plot(self.start.x, self.start.y, "xr")
            plt.plot(self.end.x, self.end.y, "xr")
            plt.axis([self.min_rand, self.max_rand, self.min_rand, self.max_rand])

            if self.final_path is not None:
                plt.plot([x for (x, y) in self.final_path], [y for (x, y) in self.final_path], '-g')
            plt.title("iter num: {}, path length: {}".format(self.i+1, round(self.final_path_length, 3) if self.final_path_length is not None else None))
            plt.savefig("{}/{}.png".format(self.filename.split('.')[0], self.i))
            plt.close()


    class AreaBounds:

        def __init__(self, area):
            self.xmin = float(area[0])
            self.xmax = float(area[1])
            self.ymin = float(area[2])
            self.ymax = float(area[3])


    def __init__(self,
                 start,
                 goal,
                 obstacle_list,
                 rand_area,
                 expand_dis=1,
                 path_resolution=0.05,
                 bias_sampling=0,
                 max_iter=500,
                 robot_radius=0.0,
                 stop_when_path_is_found = False
                 ):
        """
        Setting Parameter

        start:Start Position [x,y]
        goal:Goal Position [x,y]
        obstacleList:obstacle Positions [[x,y,size],...]
        randArea:Random Sampling Area [min,max]
        play_area:stay inside this area [xmin,xmax,ymin,ymax]
        robot_radius: robot body modeled as circle with given radius

        """
        self.start = self.Node(start[0], start[1])
        self.end = self.Node(goal[0], goal[1])
        self.min_rand = rand_area[0]
        self.max_rand = rand_area[1]

        self.play_area = self.AreaBounds((rand_area[0], rand_area[1], rand_area[0], rand_area[1]))

        self.expand_dis = expand_dis
        self.path_resolution = path_resolution
        self.bias_sampling = bias_sampling
        self.max_iter = max_iter
        self.obstacle_list = obstacle_list
        self.obstacles = obstacle_list
        self.node_list = []
        self.nd_array = np.array([])
        self.robot_radius = robot_radius

        self.stop_when_path_is_found = stop_when_path_is_found
        self.final_path = None
        self.final_node_idx = None
        self.final_path_length = None
        self.i = 0

        self.filename = "rrt_{}".format(time.strftime("%Y%m%d-%H%M%S"))
        os.mkdir(self.filename)
        # self.plot_data = self.PlotData(self.filename, self.start, self.end, self.min_rand, self.max_rand, self.play_area, self.obstacles)
        self.executor = concurrent.futures.ProcessPoolExecutor(max_workers=6)

    def add_new_node(self, node):
        if len(self.node_list) ==0:
            self.nd_array = np.array([node.x, node.y]).reshape(-1,2)
        else:    
            self.nd_array = np.append(self.nd_array, np.array([node.x, node.y]).reshape(-1,2), axis=0)
        self.node_list.append(node)
        

    def get_nearest_node(self, rnd_node):
        dlist = [(node.x - rnd_node.x)**2 + (node.y - rnd_node.y)**2
                 for node in self.node_list]
        minind = dlist.index(min(dlist))

        return self.node_list[minind]
    
    # def get_nearest_node(self, query_node):
    #     lx = self.nd_array[:,0] - query_node.x
    #     ly = self.nd_array[:,1] - query_node.y
    #     l = lx**2 + ly**2
    #     minind = np.argmin(l)

    #     return self.node_list[minind]

    def planning(self, animation=True):
        """
        rrt path planning

        animation: flag for animation on or off
        """

        # self.node_list = [self.start]
        self.add_new_node(self.start)
        for i in tqdm(range(self.max_iter)):
            self.i = i
            rnd_node = self.create_random_node()
            nearest_node = self.get_nearest_node(rnd_node)

            new_node = self.steer(nearest_node, rnd_node, self.expand_dis)
            if self.check_if_outside_play_area(new_node, self.play_area) and \
               self.is_node_safe(new_node):
                self.add_new_node(new_node)

            if self.calc_dist_to_goal(self.node_list[-1].x,
                                      self.node_list[-1].y) <= self.expand_dis:
                final_node = self.steer(self.node_list[-1], self.end,
                                        self.expand_dis)

                if self.is_node_safe(final_node):
                    self.generate_final_course(len(self.node_list) - 1)
                    if self.stop_when_path_is_found:
                        break

            self.save_data(rnd=rnd_node)


    def steer(self, from_node, to_node, extend_length=float("inf")):

        new_node = self.Node(from_node.x, from_node.y)
        d, theta = self.calc_distance_and_angle(new_node, to_node)

        new_node.path_x = [new_node.x]
        new_node.path_y = [new_node.y]

        if extend_length > d:
            extend_length = d

        n_expand = math.floor(extend_length / self.path_resolution)

        for _ in range(n_expand):
            new_node.x += self.path_resolution * math.cos(theta)
            new_node.y += self.path_resolution * math.sin(theta)
            new_node.path_x.append(new_node.x)
            new_node.path_y.append(new_node.y)

        d, _ = self.calc_distance_and_angle(new_node, to_node)
        if d <= self.path_resolution:
            new_node.path_x.append(to_node.x)
            new_node.path_y.append(to_node.y)
            new_node.x = to_node.x
            new_node.y = to_node.y

        # print(d, to_node.x, to_node.y, new_node.x, new_node.y, from_node.x, from_node.y)
        new_node.parent = from_node

        return new_node

    def generate_final_course(self, goal_ind):
        path = [[self.end.x, self.end.y]]
        length = 0
        node = self.node_list[goal_ind]
        while node.parent is not None:
            length += self.calc_dist_2_nodes(node, node.parent)
            path.append([node.x, node.y])
            node = node.parent
        path.append([node.x, node.y])

        if self.final_path_length is None or self.final_path_length > length:
            self.final_path_length = length
            self.final_path = path
            self.final_node_idx = goal_ind

    def calc_dist_2_nodes(self, node1, node2):
        return math.sqrt((node1.x-node2.x)**2 + (node1.y-node2.y)**2)

    def calc_dist_to_goal(self, x, y):
        dx = x - self.end.x
        dy = y - self.end.y
        return math.hypot(dx, dy)

    def create_random_node(self):
        if random.randint(0, 100) > self.bias_sampling*100:
            rnd = self.Node(
                random.uniform(self.min_rand, self.max_rand),
                random.uniform(self.min_rand, self.max_rand))
        else:  # goal point sampling
            rnd = self.Node(self.end.x, self.end.y)
        return rnd

    def save_data(self, rnd=None):
        if self.executor is not None:
            self.executor.submit(
                draw_graph_and_save,
                self.filename,
                rnd,
                copy.deepcopy(self.node_list),
                self.start,
                self.end,
                self.obstacles,
                self.i,
                self.play_area,
                copy.deepcopy(self.final_path),
                self.final_path_length,
                self.min_rand,
                self.max_rand
            )

    @staticmethod
    def check_if_outside_play_area(node, play_area):

        if play_area is None:
            return True  # no play_area was defined, every pos should be ok

        if node.x < play_area.xmin or node.x > play_area.xmax or \
           node.y < play_area.ymin or node.y > play_area.ymax:
            return False  # outside - bad
        else:
            return True  # inside - ok

    @staticmethod
    def check_collision(node, obstacleList, robot_radius):

        if node is None:
            return False

        for (ox, oy, size) in obstacleList:
            dx_list = [ox - x for x in node.path_x]
            dy_list = [oy - y for y in node.path_y]
            d_list = [dx * dx + dy * dy for (dx, dy) in zip(dx_list, dy_list)]

            if min(d_list) <= (size+robot_radius)**2:
                return False  # collision

        return True  # safe

    @staticmethod
    def calc_distance_and_angle(from_node, to_node):
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        d = math.hypot(dx, dy)
        theta = math.atan2(dy, dx)
        return d, theta

    @classmethod
    def line_line_intersect(cls, line1, line2):
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2

        try:
            uA = ((x4-x3)*(y1-y3) - (y4-y3)*(x1-x3)) / ((y4-y3)*(x2-x1) - (x4-x3)*(y2-y1))
            uB = ((x2-x1)*(y1-y3) - (y2-y1)*(x1-x3)) / ((y4-y3)*(x2-x1) - (x4-x3)*(y2-y1))
        except ZeroDivisionError:
            return True

        if (uA >= 0 and uA <= 1) and (uB >= 0 and uB <= 1):
            return True
        return False

    @classmethod
    def line_box_intersect(cls, line, box):
        left = cls.line_line_intersect(line, (box[0], box[1], box[0], box[1]+box[3]))
        right = cls.line_line_intersect(line, (box[0]+box[2], box[1], box[0]+box[2], box[1]+box[3]))
        top = cls.line_line_intersect(line, (box[0], box[1], box[0]+box[2], box[1]))
        bottom = cls.line_line_intersect(line, (box[0], box[1]+box[3], box[0]+box[2], box[1]+box[3]))

        intersect = left or right or top or bottom

        return intersect

    def is_line_collide(self, line):
        for ob in self.obstacles:
            is_collide = self.line_box_intersect(line, ob)
            if is_collide:
                return True
        return False

    def is_node_safe(self, node):
        line = (node.x, node.y, node.parent.x, node.parent.y)
        return not self.is_line_collide(line)

    def run(self):
        
        self.planning(animation=show_animation)
        if self.executor is not None:
            self.executor.shutdown(wait=True)
            print("Execution finished!!!")

if __name__ == '__main__':
    # executor = concurrent.futures.ProcessPoolExecutor(max_workers=6)
    env = EnvConfig02()
    rrt = RRT(
        start=env.start,
        goal=env.goal,
        rand_area=[-10, 10],
        obstacle_list=env.obstacles,
        # play_area=[0, 10, 0, 14]
        robot_radius=0.8,
        max_iter=5000
        )
    rrt.run()
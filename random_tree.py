"""

Path planning Sample Code with RRT*

author: Atsushi Sakai(@Atsushi_twi)

"""

import math
import os
import sys

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np

from rrt import RRT

show_animation = True

from configs import *


class RandomTree(RRT):
    """
    Class for RRT Star planning
    """

    class Node(RRT.Node):
        def __init__(self, x, y):
            super().__init__(x, y)
            self.cost = 0.0

    def __init__(self,
                 start,
                 goal,
                 obstacle_list,
                 rand_area,
                 expand_dis=1,
                 path_resolution=0.5,
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

        """
        super().__init__(start,
                        goal,
                        obstacle_list,
                        rand_area,
                        expand_dis,
                        path_resolution,
                        bias_sampling,
                        max_iter,
                        robot_radius,
                        stop_when_path_is_found)
    
    def get_random_node(self):
        node = np.random.choice(self.node_list)
        return node

    def planning(self, animation=True):
        """
        rrt path planning

        animation: flag for animation on or off
        """

        # self.node_list = [self.start]
        self.add_new_node(self.start)
        for i in range(self.max_iter):
            self.i = i
            rnd_node = self.create_random_node()
            p_node = self.get_random_node()

            new_node = self.steer(p_node, rnd_node, self.expand_dis)
            line = (new_node.x, new_node.y, new_node.parent.x, new_node.parent.y)
            if self.check_if_outside_play_area(new_node, self.play_area) and \
               (not self.is_line_collide(line)):
                self.add_new_node(new_node)

            # if animation and (i + 1)% 10 == 0:
            #     self.draw_graph(rnd_node)

            if self.calc_dist_to_goal(self.node_list[-1].x,
                                      self.node_list[-1].y) <= self.expand_dis:
                final_node = self.steer(self.node_list[-1], self.end,
                                        self.expand_dis)

                line = (final_node.x, final_node.y, final_node.parent.x, final_node.parent.y)
                if not self.is_line_collide(line):
                    self.generate_final_course(len(self.node_list) - 1)
                    if self.stop_when_path_is_found:
                        break
        # plt.show()
            self.save_data(rnd=rnd_node)


class RandomTreeInformed(RandomTree):
    class Node(RandomTree.Node):
        def __init__(self, x, y):
            super().__init__(x, y)
            self.cost = 0.0

    def __init__(self,
                 start,
                 goal,
                 obstacle_list,
                 rand_area,
                 expand_dis=1,
                 path_resolution=0.5,
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

        """
        super().__init__(start,
                        goal,
                        obstacle_list,
                        rand_area,
                        expand_dis,
                        path_resolution,
                        bias_sampling,
                        max_iter,
                        robot_radius,
                        stop_when_path_is_found)

        self.heuristic_dist_list = []
    
    def get_shortest_node(self):
        idx = np.argmin(self.heuristic_dist_list)
        return self.node_list[idx]

    def add_new_node(self, node):
        self.node_list.append(node)
        self.heuristic_dist_list.append(self.calc_dist_to_goal(node.x, node.y))

    def planning(self, animation=True):
        """
        rrt path planning

        animation: flag for animation on or off
        """

        # self.node_list = [self.start]
        self.add_new_node(self.start)
        for i in range(self.max_iter):
            self.i = i
            rnd_node = self.create_random_node()
            p_node = self.get_shortest_node()

            new_node = self.steer(p_node, rnd_node, self.expand_dis)
            if self.check_if_outside_play_area(new_node, self.play_area) and \
               self.is_node_safe(new_node):
                self.add_new_node(new_node)

            # if animation and (i + 1)% 10 == 0:
            #     self.draw_graph(rnd_node)

            if self.calc_dist_to_goal(self.node_list[-1].x,
                                      self.node_list[-1].y) <= self.expand_dis:
                final_node = self.steer(self.node_list[-1], self.end,
                                        self.expand_dis)

                # line = (final_node.x, final_node.y, final_node.parent.x, final_node.parent.y)
                if self.is_node_safe(final_node):
                    self.generate_final_course(len(self.node_list) - 1)
                    if self.stop_when_path_is_found:
                        break
        # plt.show()
            self.save_data(rnd=rnd_node)

if __name__ == '__main__':
    env = EnvConfig02()
    rtree = RandomTreeInformed(
        start=env.start,
        goal=env.goal,
        rand_area=[-10, 10],
        obstacle_list=env.obstacles,
        # play_area=[0, 10, 0, 14]
        robot_radius=0.8,
        max_iter=5000
        )
    rtree.run()
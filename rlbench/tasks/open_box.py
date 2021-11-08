from typing import List, Union
from pyrep.objects.joint import Joint
from pyrep.objects.dummy import Dummy
from pyrep.objects.cartesian_path import CartesianPath
from rlbench.backend.task import Task
from rlbench.backend.conditions import JointCondition

from pyrep import PyRep
from rlbench.backend.robot import Robot
import math

class OpenBox(Task):

    def __init__(self, pyrep: PyRep, robot: Robot):
        super().__init__(pyrep, robot)
        self.stage = 0
        self.waypoints = []


    def init_task(self):
        box_joint = Joint('joint')
        self.register_success_conditions([JointCondition(box_joint, 1.9)])

    def init_episode(self, index: int) -> List[str]:
        self.stage = 0
        self.waypoints = [way.get_waypoint_object() for way in self.get_waypoints()]
        return ['open box',
                'open the box lid',
                'open the box',
                'grasp the lid and open the box']

    def variation_count(self):
        return 1
        
    def reward(self, dense = 0) -> Union[float, None]:
        """Allows the user to customise the task and add reward shaping."""
        if dense == 0:
            return None
        else:
            #obj = self.robot.get_grasped_objects() 
            
            success, terminate = self.success()
            if success:
                return 10

            robot_tip = self.robot.arm.get_tip()
            tip_pose = robot_tip.get_pose()

            open_condition = all(x > 0.9 for x in self.robot.gripper.get_open_amount())
            current_grip = 1 if open_condition else 0
            #print("Testing: ", self.robot.gripper.get_open_amount(),self.stage)

            # prepare for grasp
            if self.stage == 0:
                wp_pose = self.waypoints[0].get_pose()
                grip_desired = 1

            # grasp lid
            elif self.stage == 1:
                wp_pose = self.waypoints[1].get_pose()
                grip_desired = 0

            # opening lid
            elif self.stage == 2:
                if dense == 1:
                    return 0
                wp_pose = self.waypoints[3].get_pose()
                grip_desired = 0
            else:
                return 0
            

            dist = [(a - b)**2 for a, b in zip(wp_pose, tip_pose)]
            dist_pose = math.sqrt(sum(dist[0:3]))
            d_rot2 = [(a + b)**2 for a, b in zip(wp_pose[3:], tip_pose[3:])]
            dist_rot = math.sqrt(min(sum(dist[3:]),sum(d_rot2)))

            if dense == 1:
                if (dist_pose < 0.005 and dist_rot < 0.05) and current_grip == grip_desired:
                    self.stage += 1
                    return 1
                else:
                    return 0
            else:
                if (dist_pose < 0.005 and dist_rot < 0.05) and current_grip == grip_desired:
                    self.stage += 1
                return -dist
        

import time
import itertools
import numpy as np
import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer
import cv2

xml_scene = """
<mujoco model="touchtest">
  <compiler autolimits="true"/>

  <extension>
    <plugin plugin="mujoco.sensor.touch_grid"/>
  </extension>

  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1=".1 .2 .3" rgb2=".2 .3 .4" width="300" height="300" mark="edge" markrgb=".2 .3 .4"/>
    <material name="grid" texture="grid" texrepeat="3 1" texuniform="true"/>
    <hfield file="touch_sensor_related/a.png" size="0.2 0.2 0.02 0.015"/>
  </asset>

  <visual>
      <headlight ambient=".7 .7 .7" diffuse=".2 .2 .2" specular="0.1 0.1 0.1"/>
      <map znear="0.01"/>
      <scale contactwidth=".02" contactheight=".5"/>
  </visual>

  <default>
    <geom friction="0.4" solimp="0 0.95 0.02"/>
  </default>

  <statistic center="0 0 1" extent="1" meansize=".1"/>

  <worldbody>
    <light pos="1 0 .3" dir="-1 0 -.3"/>
    <light pos="-1 0 .3" dir="1 0 -.3"/>
    <geom name="floor" pos="0 0 -0.01" type="plane" size="1 1 .01"/>

    <geom name="a" type="hfield" hfield="a" rgba=".5 .5 .7 1"/>

    <body name="ball" pos="0 0 1">
      <joint name="x" type="slide" axis="1 0 0" damping="1"/>
      <joint name="y" type="slide" axis="0 1 0" damping="1"/>
      <joint name="z" type="slide" axis="0 0 1"/>
      <joint name="rx" axis="1 0 0" springdamper="0.2 1"/>
      <joint name="ry" axis="0 1 0" springdamper="0.2 1"/>
      <geom type="ellipsoid" size=".3 .5 .1" mass="0.1" rgba=".5 .5 .5 .3"/>
      <site name="touch" pos="0 0 .1"/>
    </body>
  </worldbody>

  <sensor>
    <plugin name="touch" plugin="mujoco.sensor.touch_grid" objtype="site" objname="touch">
      <config key="size" value="7 7"/>
      <config key="fov" value="45 45"/>
      <config key="gamma" value="0"/>
      <config key="nchannel" value="3"/>
    </plugin>
  </sensor>
</mujoco>
"""

# ##使用mujoco自带API接口
# model = mujoco.MjModel.from_xml_string(xml_scene)
# # model = mujoco.MjModel.from_xml_path("tactile_assets/insertion/iiwa_robot.xml")
# Data = mujoco.MjData(model)
# viewer = mujoco.viewer.launch_passive(model, Data)
# #Close the viewer automatically after 30 wall-seconds.
# start = time.time()
#
# while time.time() - start < 300:
#   mujoco.mj_step(model, Data)
#   viewer.sync()


#----------------------------------------------------------------------------------------------------------------------
# 第一步：设置世界，采用自带的默认世界MujocoWorldBase
import numpy as np
from robosuite.models import MujocoWorldBase
world = MujocoWorldBase()

#第二步：创建自己的机器人
from robosuite.models.robots import UR5e
# mujoco_robot1 = Panda()
mujoco_robot1 = UR5e()

#我们可以通过创建一个抓手实例并在机器人上调用 add_gripper 方法来为机器人添加一个抓手。
from robosuite.models.grippers import gripper_factory
gripper = gripper_factory('Robotiq85Gripper')
mujoco_robot1.add_gripper(gripper)

#要将机器人添加到世界中，我们将机器人放置到所需位置并将其合并到世界中
mujoco_robot1.set_base_xpos([0.5,0,0.8])#刚好能放置在桌子上，桌子高度为0.8
world.merge(mujoco_robot1)
#第三步：创建桌子。我们可以初始化创建桌子和地平面,TableArena代表的是一个拥有桌子的整体环境
from robosuite.models.arenas import TableArena

mujoco_arena = TableArena()
mujoco_arena.set_origin([0.8, 0, 0])#这里的set_origin是对所有对象应用恒定偏移。在X轴偏移0.8
world.merge(mujoco_arena)

#第四步：添加对象。创建一个球并将其添加到世界中。

from robosuite.models.objects import BallObject
sphere1 = BallObject(
    name="sphere1",
    size=[0.04],
    rgba=[0, 0.5, 0.5, 1]).get_obj()
sphere1.set('pos', '1.0 -1.0 1.0')
world.worldbody.append(sphere1)

from robosuite.models.objects import BoxObject
sphere2 = BoxObject(
    name = "sphere2",
    size = [0.07, 0.07, 0.07],
    rgba=[0, 0.5, 0.5, 1]).get_obj()
sphere2.set('pos', '1.0 -0.5 1.0')
world.worldbody.append(sphere2)

from robosuite.models.objects import CapsuleObject
sphere3 = CapsuleObject(
    name = "sphere3",
    size = [0.07,  0.07],
    rgba=[0.5, 0.5, 0.5, 1]).get_obj()#颜色(三个0.5是黑色，A值代表透明度)
sphere3.set('pos', '1.0 0 1.0')
world.worldbody.append(sphere3)

from self_object import Desk_Surface
desk_surface = Desk_Surface(name="desk_surface")
desk_surface_object = desk_surface.get_obj()
desk_surface_object.set("pos", '1.0 -0.1 1.0')
desk_surface_object.set("quat", '0 1 0 0')
world.worldbody.append(desk_surface_object)
world.merge_assets(desk_surface)

from robosuite.utils.mjcf_utils import new_joint
from robosuite.utils import OpenCVRenderer
from robosuite.utils.binding_utils import MjRenderContextOffscreen, MjSim

#第5步：运行模拟。一旦我们创建了对象，我们可以通过运行mujoco_py获得一个模型
model = world.get_model(mode="mujoco")
sim = MjSim(model)
viewer = OpenCVRenderer(sim)
render_context = MjRenderContextOffscreen(sim, device_id=-1)
sim.add_render_context(render_context)
viewer.render()
while True:
    # Step through sim
    sim.step()
    viewer.render()

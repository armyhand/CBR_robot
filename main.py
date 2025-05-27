import time
import itertools
import numpy as np
import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer
import cv2
from motion_controller import controller
from motion_planning import trac_plan
from scipy.spatial.transform import Rotation as R

xml_scene = """
<mujoco model="scene">
  <!--include file="tactile_assets/insertion/iiwa_robot.xml"-->
  <include file="tactile_assets/insertion/scene.xml"/>
  
  <asset>  
    <texture name="texplane" type="2d" builtin="checker" rgb1=".2 .3 .4" rgb2=".1 0.15 0.2" width="512" height="512" mark="cross" markrgb=".8 .8 .8"/>
    <material name="matplane" reflectance="0.3" texture="texplane" texrepeat="1 1" texuniform="true"/>
  </asset>
  

  <worldbody>
    <light pos="1 0 .3" dir="-1 0 -.3"/>
    <light pos="-1 0 .3" dir="1 0 -.3"/>

    <!--geom name="a" type="box" size="1 1 .1" rgba=".5 .5 .7 1"-->

  </worldbody>
</mujoco>
"""
# 控制关节的示例函数
def move_joint(actuator_name, target_position):
    # 获取actuator的ID
    actuator_id = model.actuator(actuator_name).id
    # 设置控制信号
    Data.ctrl[actuator_id] = target_position
    # 执行一步仿真
    mujoco.mj_step(model, Data)
    # 更新可视化

def show_tactile(tactile, size=(480, 480), max_shear=0.05, max_pressure=0.1,
                 name='tactile'):  # Note: default params work well for 16x16 or 32x32 tactile sensors, adjust for other sizes
    nx = tactile.shape[2]
    ny = tactile.shape[1]

    loc_x = np.linspace(0, size[1], nx)
    loc_y = np.linspace(size[0], 0, ny)

    img = np.zeros((size[0], size[1], 3))

    for i in range(0, len(loc_x), 1):
        for j in range(0, len(loc_y), 1):
            dir_x = np.clip(tactile[0, j, i] / max_shear, -1, 1) * 20
            dir_y = np.clip(tactile[1, j, i] / max_shear, -1, 1) * 20

            color = np.clip(tactile[2, j, i] / max_pressure, 0, 1)
            r = color
            g = 1 - color

            cv2.arrowedLine(img, (int(loc_x[i]), int(loc_y[j])), (int(loc_x[i] + dir_x), int(loc_y[j] - dir_y)),
                            (0, g, r), 4, tipLength=0.5)

    cv2.imshow(name, img)

    return img

def render(highres=False, camera_idx=1):

    if highres:
        renderer = mujoco.Renderer(model, height=480, width=480)
        renderer.update_scene(Data, camera=camera_idx)
        img = renderer.render() / 255
    else:
        renderer = mujoco.Renderer(model, height=480, width=480)
        renderer.update_scene(Data, camera=camera_idx)
        img = renderer.render() / 255

    return img


##使用mujoco自带API接口
model = mujoco.MjModel.from_xml_string(xml_scene)
# model = mujoco.MjModel.from_xml_path("tactile_assets/insertion/iiwa_robot.xml")
Data = mujoco.MjData(model)
viewer = mujoco.viewer.launch_passive(model, Data)
#Close the viewer automatically after 30 wall-seconds.
start = time.time()

# # 运行示例
# time.sleep(1)  # 暂停1秒
# move_joint('fingers_actuator', 0)  # 完全关闭手指
#
# duration=1
# step_time=0.01
# symlog_tactile = True
#
# steps = int(duration / step_time)
# for i in range(steps):
#     move_joint('fingers_actuator', 255)# 完全打开手指
#     move_joint('base_actuator_z', 0.1)
#     img = render(highres=True, camera_idx=1)
#     cv2.imshow('img', img[:, :, ::-1])
#
#     tactiles_right = Data.sensor('touch_right').data.reshape((3, 32, 32))
#     tactiles_right = tactiles_right[[1, 2, 0]]  # zxy -> xyz
#     tactiles_left = Data.sensor('touch_left').data.reshape((3, 32, 32))
#     tactiles_left = tactiles_left[[1, 2, 0]]  # zxy -> xyz
#     tactiles = np.concatenate((tactiles_right, tactiles_left), axis=0)
#     if symlog_tactile:
#         tactiles = np.sign(tactiles) * np.log(1 + np.abs(tactiles))
#     img_tactile1 = show_tactile(tactiles[:3], name='tactile_right')
#     img_tactile2 = show_tactile(tactiles[3:], name='tactile_left')
#     cv2.waitKey(1)
# time.sleep(1)
# for i in range(steps):
#     move_joint('fingers_actuator', 0)# 完全闭合手指
#     move_joint('base_actuator_yaw', 3.0)
#     img = render(highres=True, camera_idx=1)
#     cv2.imshow('img', img[:, :, ::-1])
#
#     tactiles_right = Data.sensor('touch_right').data.reshape((3, 32, 32))
#     tactiles_right = tactiles_right[[1, 2, 0]]  # zxy -> xyz
#     tactiles_left = Data.sensor('touch_left').data.reshape((3, 32, 32))
#     tactiles_left = tactiles_left[[1, 2, 0]]  # zxy -> xyz
#     tactiles = np.concatenate((tactiles_right, tactiles_left), axis=0)
#     if symlog_tactile:
#         tactiles = np.sign(tactiles) * np.log(1 + np.abs(tactiles))
#     img_tactile1 = show_tactile(tactiles[:3], name='tactile_right')
#     img_tactile2 = show_tactile(tactiles[3:], name='tactile_left')
#     cv2.waitKey(1)
# time.sleep(1)

while time.time() - start < 300:
  mujoco.mj_step(model, Data)
  viewer.sync()
# #
# #   # Rudimentary time keeping, will drift relative to wall clock.
# # 关闭查看器
# viewer.close()
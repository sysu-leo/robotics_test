import sys
import os
from dm_control import mjcf
from envs.cassie import cassie_locomotion_env
from terrain_generator import TerrainGenerator, OUTPUT_SCENE_PATH
import random
import string
from pathlib import Path

scene_dict = {
    'flat': 'envs/cassie/model/flat.xml',
    'hill': 'envs/cassie/model/hill.xml',
    'block': 'envs/cassie/model/bblock.xml',
    'stair': 'envs/cassie/model/stair.xml',
    'ridges': 'envs/cassie/model/ridges.xml',
    'test': 'envs/cassie/model/scene_terrain.xml',
}

def generate_flat_scene(input, output):
    tg = TerrainGenerator(input, output, 'flat')

    t_path = Path(OUTPUT_SCENE_PATH)
    tg.AddBox(position=[0.0, 0.0, -0.05],
              euler=[0.0, 0.0, 0.0],
              size=[20, 20, 0.1])
    os.makedirs(t_path.parent, exist_ok=True)
    tg.Save()

def generate_hill_scene(input, output):
    tg = TerrainGenerator(input, output, 'hill')

    t_path = Path(OUTPUT_SCENE_PATH)
    os.makedirs(t_path.parent, exist_ok=True)
    tg.AddPerlinHeighField(position=[0.0, 0.0, 0.0], size=[20, 20])
    tg.Save()

def generate_stair_scene(input, output):
    tg = TerrainGenerator(input, output, 'stair')

    t_path = Path(OUTPUT_SCENE_PATH)
    os.makedirs(t_path.parent, exist_ok=True)
    tg.AddBox(position=[0.0, 0.0, -0.05],
              euler=[0.0, 0.0, 0.0],
              size=[20, 20, 0.1])
    tg.AddStairs(init_pos=[0.0, 0.0, 0.0], yaw=0.0)
    tg.Save()

def builder(export_path, scene_name):

    print("Modifying XML model...")
    scene_path = scene_dict[scene_name]
    cassie_model = mjcf.from_path(scene_path)

    # set njmax and nconmax
    cassie_model.size.njmax = -1
    cassie_model.size.nconmax = -1
    cassie_model.statistic.meansize = 0.1
    cassie_model.statistic.meanmass = 2

    # modify skybox
    for tx in cassie_model.asset.texture:
        if tx.type=="skybox":
            tx.rgb1 = '1 1 1'
            tx.rgb2 = '1 1 1'


    unused_leg_joints = [ 'left-shin', 'right-shin',
                         'left-tarsus', 'right-tarsus',
                         'left-foot-crank', 'right-foot-crank',]

    equaility = ['left-plantar-rod', 'left-achilles-rod', 'right-achilles-rod',
                 'right-plantar-rod', 'right-heel-spring', 'left-heel-spring']

    used_leg_joints = ['left-hip-roll', 'left-hip-yaw', 'left-hip-pitch', 'left-knee', 'left-foot',
                  'right-hip-roll', 'right-hip-yaw', 'right-hip-pitch', 'right-knee', 'right-foot']

    cassie_model.find('geom', 'floor').remove()
    cassie_model.worldbody.add('body', name='floor')
    cassie_model.find('body', 'floor').add('geom', name='floor', type="plane", size="0 0 0.25", material="groundplane")
    # export model
    mjcf.export_with_assets(cassie_model, out_dir=os.path.dirname(export_path), out_file_name=export_path, precision=5)
    print("Exporting XML model to ", export_path)
    return

if __name__=='__main__':
    tmp_env_path = '/home/leon/cassie_rl_test/log/tmp/cassie_test_3.xml'
    scene_name = 'stair'

    input_path = "envs/cassie/model/scene.xml"
    output_path = "envs/cassie/model/"

    generate_stair_scene(input_path, output_path)

    builder(tmp_env_path, scene_name)
    t_mujoco_env = cassie_locomotion_env.CassieLocomotionEnv(tmp_env_path)

    print("num of joints = %d"%(t_mujoco_env.model.nu))
    print("dim of pos = %d" % (t_mujoco_env.model.nq))
    print("dim of freedom = %d" % (t_mujoco_env.model.nv))
    print()
    print("dim of activate = %d" % (t_mujoco_env.model.na))
    t_mujoco_env.reset(t_mujoco_env.get)
    while True:
        t_mujoco_env.render()

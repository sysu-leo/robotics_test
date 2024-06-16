import mujoco_py as mp
import mujoco

model_path = '/home/leon/cassie_rl_test/log/cassie scene/scene_terrain.xml'

model = mujoco.MjModel.from_xml_path(model_path)

sim = mp.MjSim(model)
viewer = mp.MjViewer(sim)

while True:
    sim.step()
    viewer.render()
import pybullet as p
import platform
import logging


class Simulator:

    def __init__(self,
                 gravity=9.8,
                 physics_timestep=1 / 120.0,
                 mode='gui'):
        # physics simulator
        self.gravity = gravity
        self.physics_timestep = physics_timestep
        self.mode = mode

        # TODO: remove this, currently used for testing only
        self.objects = []

        self.use_pb_renderer = False

        if self.mode in ['gui', 'pbgui']:
            self.use_pb_renderer = True

        self.load()
        self.body_links_awake = 0

    def set_timestep(self, physics_timestep):
        self.physics_timestep = physics_timestep
        p.setTimeStep(self.physics_timestep)

    def reload(self):
        self.disconnect()
        self.load()

    def load(self):
        if self.use_pb_renderer:
            self.cid = p.connect(p.GUI)
        else:
            self.cid = p.connect(p.DIRECT)
        p.setTimeStep(self.physics_timestep)
        p.setGravity(0, 0, -self.gravity)
        p.setPhysicsEngineParameter(enableFileCaching=0)

        self.visual_objects = {}
        self.robots = []
        self.scene = None

    def load_without_pybullet_vis(load_func):
        """
        Load without pybullet visualizer
        """
        def wrapped_load_func(*args, **kwargs):
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, False)
            res = load_func(*args, **kwargs)
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, True)
            return res
        return wrapped_load_func

    @load_without_pybullet_vis
    def import_scene(self,scene):
        new_object_pb_ids = scene.load()
        self.objects += new_object_pb_ids
        self.scene = scene
        return new_object_pb_ids

    @load_without_pybullet_vis
    def import_ig_scene(self, scene):
        new_object_ids = scene.load()
        self.objects += new_object_ids
        self.scene = scene
        return new_object_ids

    @load_without_pybullet_vis
    def import_object(self, obj):
        new_object_pb_id = obj.load()
        self.objects += [new_object_pb_id]
        return new_object_pb_id

    @load_without_pybullet_vis
    def import_robot(self, robot):
        ids = robot.load()
        self.robots.append(robot)
        return ids

    def step(self):
        p.stepSimulation()

    def isconnected(self):
        """
        :return: pybullet is alive
        """
        return p.getConnectionInfo(self.cid)['isConnected']

    def disconnect(self):
        """
        Clean up the simulator
        """
        if self.isconnected():
            print("******************PyBullet Logging Information:")
            p.resetSimulation(physicsClientId=self.cid)
            p.disconnect(self.cid)
            print("PyBullet Logging Information******************")

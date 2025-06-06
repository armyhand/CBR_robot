import numpy as np
from robosuite.models.objects import MujocoXMLObject, CompositeObject
from robosuite.utils.mjcf_utils import xml_path_completion, array_to_string, find_elements, CustomMaterial, add_to_dict, RED, GREEN, BLUE
from robosuite.models.objects import BoxObject
import robosuite.utils.transform_utils as T


import pathlib
absolute_path = pathlib.Path(__file__).parent.absolute()


class Desk_Surface(MujocoXMLObject):
    def __init__(self, name):

        super().__init__(str(absolute_path) + "/" + "desk_surface.xml",
                         name=name,
                         joints=None,
                         obj_type="all",
                         duplicate_collision_geoms=True)

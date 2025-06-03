_base_ = ["../_base_/default.py"]

variant = "scalar_rgb"

# *******************************************************************
# define the common parameters out of the dict for easy modification
# below can be changed by '--options' in command line
sensor_origin = [2, 2, 2]
sensor_target = [0, 0, 0.1]
sensor_up = [0, 0, 1]
resolution = [1920, 1080]
emitter_intensity = [3, 3, 3]  # list to specify rgb intensity
emitter_origin = [4, -4, 20]
emitter_target = [0, 0, 0]
emitter_up = [0, 0, 1]
# ------------------------------------------------------------------
# !!! below can NOT be changed by '--options' in command line !!!
ele_shape = "sphere"
ele_size = 0.012
ele_color = [0.8, 0.8, 0.8]
spp = 256  # render samples per pixel
# *******************************************************************


# ---------------------------------------------------------------------------------
# custom part of original parameters for specified scenes
specify = {
    "plane.ply": {
        "modify_scene": dict(
            sensor=dict(
                film=dict(width=1920, height=1080),
                sampler=dict(sample_count=512),
            ),
        ),
        "modify_transform": {
            # "flip": dict(type="Flip", axis="y"),
            "rotation": dict(type="Rotation", axes=['x'], angles=[90]),
            # "fps": dict(type="DownSample", n_sample=1024, algorithm="fps"),
            "translate": dict(type="Translate", translate=[0.0, 0, 0.25]),
        },
    },
    "office.ply": {
        "modify_scene": dict(
            sensor=dict(
                to_world=dict(
                    type="LookAt",
                    origin=[1.8, 1.8, 2],
                    target=[0, 0, 0.55],
                    up=sensor_up,
                ),
                film=dict(width=2000, height=1800),
                sampler=dict(sample_count=256),
            ),
            emitter=dict(
                value=1.9,  # emitter intensity for rgb
                to_world=dict(
                    transforms=[
                        dict(
                            type="LookAt",
                            origin=[4, 4, 20],
                            target=emitter_target,
                            up=emitter_up,
                        ),
                        dict(type="Scale", scale=[60, 60, 1]),
                    ],
                ),
            ),
        ),
        "modify_transform": {
            # "rainbow_color": dict(type="RainbowColor", enable=False),
            "colorizer": dict(type="Colorizer", color=None),
            "normalize": dict(type="Normalize", scale_range=1.5),
            "rotation": dict(type="Rotation", axes=['z'], angles=[120]),
            # "fps": dict(type="DownSample", n_sample=10240, algorithm="random"),
            "translate": dict(type="Translate", translate=[0.0, 0, 0.55]),
        },
        # "modify_element_shape": dict(
        #     to_world=dict(
        #         transforms=[
        #             dict(type="Translate", translate=None),  # position of element
        #             dict(type="Scale", scale=0.005),
        #         ]
        #     )
        # ), # TODO support modify the element shape
    },
}
# ---------------------------------------------------------------------------------

# define the base shape of element in scenes, the 'None' will be filled in the main process
element_shape = dict(
    type="Shape",
    assign=ele_shape,
    identy="general",  # id of element
    to_world=dict(
        type="ChainTransform",
        transforms=[
            dict(type="Translate", translate=None),  # position of element
            dict(type="Scale", scale=ele_size),  # size of element
        ],
    ),
    bsdf=dict(
        type="Bsdf", assign="diffuse", color=ele_color
    ),  # materia of element, 'reflectance' is the color of element
)

# transformation applied on the raw data
pre_transform = {
    "to_numpy": dict(type="ToNumpy"),
    # "fps": dict(type="DownSample", n_sample=10240),
    "normalize": dict(type="Normalize", scale_range=1.0),
    # "rainbow_color": dict(type="RainbowColor", enable=True),
    "colorizer": dict(type="Colorizer", color='rainbow'),
    "translate": dict(type="Translate", translate=[0, 0, 0.55]),
}

base_scene = dict(
    type="Scene",
    assign="scene",
    sensor=dict(
        type="Sensor",
        assign="perspective",
        to_world=dict(
            type="LookAt", origin=sensor_origin, target=sensor_target, up=sensor_up
        ),
        fov=39.3077,
        near_clip=0.1,
        far_clip=100,
        film=dict(
            type="Film",
            assign="hdrfilm",
            width=resolution[0],
            height=resolution[1],
            filter="gaussian",
        ),  # sensor resolution and filter
        sampler=dict(type="Sampler", assign="independent", sample_count=spp),
    ),
    emitter=dict(
        type="Emitter",
        assign="area",
        value=emitter_intensity,  # emitter intensity for rgb
        shape="rectangle",
        to_world=dict(
            type="ChainTransform",
            transforms=[
                dict(
                    type="LookAt",
                    origin=emitter_origin,
                    target=emitter_target,
                    up=emitter_up,
                ),
                dict(type="Scale", scale=[20, 20, 2]),
            ],
        ),
    ),
    integrator=dict(type="Integrator", assign="path", max_depth=-1),
    shape=dict(
        type="Shape",
        assign="rectangle",
        to_world=dict(
            type="ChainTransform",
            transforms=[
                dict(type="Translate", translate=[0, 0, 0]),
                dict(type="Scale", scale=[100, 100, 1]),
            ],
        ),
        bsdf=dict(
            type="Bsdf",
            assign="roughplastic",
            distribution="ggx",
            alpha=0.1,
        ),
    ),
    # emitter_1=dict(
    #     type="Emitter",
    #     identy="1",
    #     assign="area",
    #     value=1.0,
    #     shape="rectangle",
    #     to_world=dict(
    #         type="ChainTransform",
    #         transforms=[
    #             dict(
    #                 type="LookAt",
    #                 origin=[4, -4, 20],
    #                 target=[4, -4, 0],
    #                 up=emitter_up,
    #             ),
    #             dict(type="Scale", scale=[20, 20, 2]),
    #         ],
    #     ),
    # ),  # append a new emitter
)

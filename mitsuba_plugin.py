from typing import Union

from transform import build_transform, ChainTransform, Translate
from utils.registry import Registry

PLUGIN = Registry("plugin")


def build_plugin(cfg):
    if cfg is not None:
        return PLUGIN.build(cfg=cfg)


@PLUGIN.register_module()
class Bsdf:
    def __init__(self, assign, identy="base", **kwargs):
        self.assign = assign
        self.identy = f"bsdf_{identy}"
        if self.assign == "diffuse":
            self.color = kwargs.get("color", [1, 1, 1])
        elif self.assign == "roughplastic":
            self.distribution = kwargs.get("distribution", "beckmann")
            self.alpha = kwargs.get("alpha", 0.1)
            self.int_ior = kwargs.get("int_ior", 1.49)
            self.color = kwargs.get("color", [1, 1, 1])
        elif self.assign == "ref":
            self.ref_identy = kwargs.get("ref_identy", None)
        else:
            self.params = kwargs
            print(
                f"bsdf {self.assign} is not supported yet."
                f"Use custom parameters {self.params}"
            )

    def inst(self, content_only=False):
        base = {self.identy: {"type": self.assign}}
        if self.assign == "diffuse":
            base[self.identy].update(
                {"reflectance": {"type": "rgb", "value": self.color}}
            )
        elif self.assign == "roughplastic":
            base[self.identy].update(
                {
                    "distribution": self.distribution,
                    "alpha": self.alpha,
                    "int_ior": self.int_ior,
                    "diffuse_reflectance": {
                        "type": "rgb",
                        "value": self.color,
                    },
                }
            )
        elif self.assign == "ref":
            base[self.identy].update({"id": self.ref_identy})
        else:
            base[self.identy].update(self.params)
        return base if not content_only else base[self.identy]


@PLUGIN.register_module()
class Emitter:
    def __init__(self, assign, value, identy="base", **kwargs):
        self.assign = assign
        self.identy = f"emitter_{identy}"
        self.value = value
        if self.assign == "constant":
            pass
        elif self.assign == "area":
            self.shape = kwargs.get("shape", "rectangle")
            self.to_world = build_transform(kwargs.get("to_world"))
        else:
            self.param = kwargs
            print(
                f"emitter {self.assign} is not supported yet."
                f"Use custom parameters {self.param}"
            )

    def inst(self, content_only=False):
        base = dict()
        if self.assign == "constant":
            base = {self.identy: {"type": self.assign}}
            base[self.identy].update({"radiance": {"type": "rgb", "value": self.value}})
        elif self.assign == "area":
            base = {self.identy: {"type": self.shape}}
            base[self.identy].update(
                {
                    "to_world": self.to_world.func(),
                    "emitter": {
                        "type": self.assign,
                        "radiance": {"type": "rgb", "value": self.value},
                    },
                }
            )
        return base if not content_only else base[self.identy]


@PLUGIN.register_module()
class Film:
    def __init__(self, assign, width, height, identy="base", **kwargs):
        self.assign = assign
        self.identy = f"film_{identy}"
        self.width = width
        self.height = height
        if self.assign == "hdrfilm":
            self.filter = kwargs.get("filter", "gaussian")
        else:
            self.param = kwargs
            print(
                f"film {self.assign} is not supported yet."
                f"Use custom parameters {kwargs}"
            )

    def inst(self, content_only=False):
        base = {
            self.identy: {
                "type": self.assign,
                "width": self.width,
                "height": self.height,
            }
        }
        if self.assign == "hdrfilm":
            base[self.identy].update({"rfilter": {"type": self.filter}})
        else:
            base[self.identy].update(self.param)
        return base if not content_only else base[self.identy]


@PLUGIN.register_module()
class Integrator:
    def __init__(self, assign, identy="base", **kwargs):
        self.assign = assign
        self.identy = f"integrator_{identy}"
        if self.assign == "path":
            self.max_depth = kwargs.get("max_depth", -1)
        else:
            self.param = kwargs
            print(
                f"integrator {self.assign} is not supported yet."
                f"Use custom parameters {self.param}"
            )

    def inst(self, content_only=False):
        base = {self.identy: {"type": self.assign}}
        if self.assign == "path":
            base[self.identy].update({"max_depth": self.max_depth})
        else:
            base[self.identy].update(self.param)
        return base if not content_only else base[self.identy]


@PLUGIN.register_module()
class Sampler:
    def __init__(self, assign, sample_count, identy="base", **kwargs):
        self.assign = assign
        self.identy = f"sampler_{identy}"
        self.sample_count = sample_count

    def inst(self, content_only=False):
        base = {self.identy: {"type": self.assign, "sample_count": self.sample_count}}
        return base if not content_only else base[self.identy]


class BaseSensor:
    def __init__(self, assign, fov, to_world):
        self.assign = assign
        self.fov = fov
        self.to_world = build_transform(to_world)

    def __call__(self):
        return {"type": self.assign, "fov": self.fov, "to_world": self.to_world.func()}


@PLUGIN.register_module()
class Sensor(BaseSensor):
    def __init__(
        self,
        assign,
        to_world,
        fov,
        film: Film = None,
        sampler: Sampler = None,
        identy="base",
        **kwargs,
    ):
        super().__init__(assign, fov, to_world)
        self.assign = assign
        self.identy = f"sensor_{identy}"
        if self.assign == "perspective":
            self.near_clip = kwargs.get("near_clip", 0.01)
            self.far_clip = kwargs.get("far_clip", 10000)
        else:
            self.param = kwargs
            print(
                f"sensor {self.assign} is not supported yet."
                f"Use custom parameters {self.param}"
            )
        self.film = build_plugin(film)
        self.sampler = build_plugin(sampler)

    def inst(self, content_only=False):
        base = {self.identy: super().__call__()}
        if self.assign == "perspective":
            base[self.identy].update(
                {"near_clip": self.near_clip, "far_clip": self.far_clip}
            )
        else:
            base[self.identy].update(self.param)
        if self.film is not None:
            base[self.identy].update(self.film.inst())
        if self.sampler is not None:
            base[self.identy].update(self.sampler.inst())
        return base if not content_only else base[self.identy]


@PLUGIN.register_module()
class MultiSensors:
    def __init__(
        self,
        *sensors: BaseSensor,
        film: Film = None,
        sampler: Sampler = None,
        identy="base",
    ):
        self.identy = f"sensor_multi_{identy}"
        self.film = build_plugin(film)
        self.sampler = build_plugin(sampler)
        self.sensors = []
        for sensor in sensors:
            self.sensors.append(build_plugin(sensor))

    def inst(self):
        base = {self.identy: {"type": "batch"}}
        for i, sensor in enumerate(self.sensors):
            base[self.identy].update({f"sensor_{i}": sensor()})
        if self.film is not None:
            base[self.identy].update(self.film.inst())
        if self.sampler is not None:
            base[self.identy].update(self.sampler.inst())
        return base


@PLUGIN.register_module()
class Shape:
    def __init__(self, assign, bsdf: Bsdf = None, identy="base", **kwargs):
        self.assign = assign
        self.identy = f"shape_{identy}"
        self.bsdf = build_plugin(bsdf)
        self.to_world = (
            build_transform(kwargs.get("to_world")) if kwargs.get("to_world") else None
        )
        if self.assign == "rectangle":  # TODO add more params
            pass
        elif self.assign == "sphere":
            pass
        elif self.assign == "cube":
            pass
        else:
            self.param = kwargs
            print(
                f"shape {self.assign} is not supported yet."
                f"Use custom parameters {self.param}"
            )

    def inst(self, identy=None, content_only=False):
        identy = self.identy if identy is None else identy
        base = {identy: {"type": self.assign}}
        base[identy].update({"to_world": self.to_world.func()})
        if self.assign == "rectangle":
            pass
        elif self.assign == "sphere":
            pass
        elif self.assign == "cube":
            pass
        else:
            base[identy].update(self.param)
        if self.bsdf is not None:
            base[identy].update(self.bsdf.inst())
        return base if not content_only else base[identy]

    def update_coord(self, coord):
        if isinstance(self.to_world, ChainTransform):
            self.to_world.update(dict(type="Translate", translate=coord))
        elif isinstance(self.to_world, Translate):
            self.to_world.translate = coord

    def update_color(self, color):
        self.bsdf.color = color


@PLUGIN.register_module()
class Scene:
    def __init__(
        self, assign, **elements: Union[Sensor, Integrator, Shape, Emitter, Bsdf]
    ):
        self.assign = assign
        self.ele = dict()
        for n, e in elements.items():
            if e is not None:
                self.ele[n] = build_plugin(e)

    def inst(self):
        base = {"type": self.assign}
        for n, e in self.ele.items():
            base.update(e.inst())
        return base

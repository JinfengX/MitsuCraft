import json
import os
import sys

import mitsuba as mi
import numpy as np
from plyfile import PlyData

from mitsuba_plugin import build_plugin
from utils.misc import (
    complie_regex,
    is_dict,
    insert_after_substring,
)
from utils.path import (
    new_suffix,
    check_file_exist,
    scandir_regex,
    mkdir_or_exist,
    fopen,
)


class FileTool:
    def __init__(self, cfg):
        self.logger = cfg.logger
        self.variant = cfg.variant
        self.work_dir = cfg.work_dir
        self.save_path = cfg.save_path
        self.file = cfg.file
        self.regex = cfg.regex
        self.recur = cfg.recursive
        self.processed_dir = cfg.processed_dir
        self.save_type = cfg.save_type
        self.specify = cfg.specify
        self.task = None

        self.dump_dir = "convert"

    def get_task(self):
        file_task = self._check_file_path()
        regex_task = self._regex_search()
        task = set(file_task + regex_task)
        self.task = {
            t: {
                "path": os.path.join(self.work_dir, t),
                "raw": None,
                "convert": None,
                "scene": None,
            }
            for t in task
        }
        self._get_processed_file()

        return self.task

    def _check_file_path(self):
        if self.file is None:
            return []
        files = list()
        for file in self.file:
            try:
                check_file_exist(os.path.join(self.work_dir, file))
                files.append(file)
            except FileNotFoundError:
                self.logger.warning(f"File not found: {file}")
        return files

    def _regex_search(self):
        return (
            list(
                scandir_regex(
                    self.work_dir, complie_regex(self.regex), recursive=self.recur
                )
            )
            if self.regex is not None
            else []
        )

    def _get_processed_file(self):
        if self.processed_dir is not None:
            for task in self.task.keys():
                for suffix in self.save_type:
                    path = os.path.join(
                        self.processed_dir,
                        self.dump_dir,
                        new_suffix(task, f".{suffix}"),
                    )
                    if os.path.isfile(path):
                        cur_path = self.get_path(task)
                        if not any(
                            cur_path.endswith(f".{sf}") for sf in self.save_type
                        ):
                            self.update_path(task, path)
                        elif cur_path.endswith(".xml"):
                            continue
                        elif cur_path.endswith(".dict") and suffix == "xml":
                            self.update_path(task, path)

    def load_raw(self, task):
        task_path = self.get_path(task)
        if task_path.endswith(".ply"):
            self.logger.info(f"Loading ply file from {task_path}")
            self.update_raw(task, self.load_ply(task_path))
        elif task_path.endswith('.pth'):
            self.logger.info(f"Loading pth file from {task_path}")
            self.update_raw(task, self.load_pth(task_path))
        elif task_path.endswith(".dict"):
            self.logger.info(f"Loading dict file from {task_path}")
            self.update_convert(task, self.load_dict(task_path))
        # elif task_path.endswith(".xml"):
        #     self.logger.info(f"Loading xml file from {task_path}")
        #     self.update_convert(task, self.load_xml(task_path))

    @staticmethod
    def load_ply(path):
        ply = PlyData.read(path)
        vertex = ply["vertex"]
        components = []
        if "x" in vertex and "y" in vertex and "z" in vertex:
            xyz = np.column_stack([vertex[t] for t in ("x", "y", "z")])
            components.append(xyz)
        if "red" in vertex and "green" in vertex and "blue" in vertex:
            rgb = np.column_stack([vertex[t] for t in ("red", "green", "blue")])
            if rgb.max() > 1.0:
                rgb = rgb / 255
            components.append(rgb)
        assert components, f"No valid components found in ply file: {path}"

        return np.column_stack(components)

    @staticmethod
    def load_pth(path):
        import torch

        return torch.load(path)

    @staticmethod
    def load_dict(path):
        with fopen(path, "rb") as f:
            content = f.read()
        return json.loads(content, object_hook=dict_to_miscene)

    @staticmethod
    def load_xml(path):
        with fopen(path, "r") as f:
            content = f.read()
        return content

    def convert(self, task, scene, element_shape=None):
        raw_data = self.get_raw(task)
        if raw_data is None:
            file_type = self.get_path(task).split(".")[-1]
            self.logger.info(f"Skip conversion with input file type: {file_type}")
        else:
            self.logger.info(f"Converting file ...")
            if isinstance(raw_data, np.ndarray):
                raw_data = raw_data.tolist()
            scene = scene.inst()
            base_ele = build_plugin(element_shape)
            for i, ele in enumerate(raw_data):
                base_ele.update_coord(ele[:3])
                if len(ele) == 6:
                    base_ele.update_color(ele[3:6])
                ele_i = base_ele.inst(f"shape_{i}")
                scene.update(ele_i)
            self.update_convert(task, scene)

    def dump(self, task):
        if self.get_raw(task) is None:
            self.logger.info(f"Skip dumping with input: {self.get_path(task)}")
            return
        task_base_dir = os.path.dirname(task)
        mkdir_or_exist(os.path.join(self.save_path, self.dump_dir, task_base_dir))
        dict_data = None
        if "dict" in self.save_type:
            convert = self.get_convert(task)
            save_path = os.path.join(
                self.save_path, self.dump_dir, new_suffix(task, ".dict")
            )
            self.logger.info(f"Saving converted file to {save_path}")
            try:
                import orjson

                dict_data = orjson.dumps(
                    convert,
                    option=orjson.OPT_PASSTHROUGH_SUBCLASS | orjson.OPT_SERIALIZE_NUMPY,
                    default=miscene_to_dict,
                )
                with fopen(save_path, "wb") as f:
                    f.write(dict_data)
            except ImportError:
                self.logger.warning("orjson not found, using json instead")
                dict_data = json.dumps(convert, default=miscene_to_dict)
                with fopen(save_path, "w") as f:
                    f.write(dict_data)
        if "xml" in self.save_type and dict_data is not None:
            save_path = os.path.join(
                self.save_path, "convert", new_suffix(task, ".xml")
            )
            self.logger.info(f"Saving converted file to {save_path}")
            mi.xml.dict_to_xml(scene_dict=self.get_convert(task), filename=save_path)

    def parse(self, task):
        path = self.get_path(task)
        convert = self.get_convert(task)
        raw = self.get_raw(task)
        if convert is None and raw is None and path.endswith(".xml"):
            self.logger.info(f"Parsing xml file ...")
            scene = mi.load_file(path, parallel=True)
        elif is_dict(convert):
            self.logger.info(f"Parsing dict ...")
            scene = mi.load_dict(convert, parallel=True)
        else:
            raise "Invalid type for convert, should be str or dict"
        self.update_scene(task, scene)

    def get_path(self, task):
        return self.task[task]["path"]

    def update_path(self, task, path):
        self.task[task].update({"path": path})

    def get_raw(self, task):
        return self.task[task]["raw"]

    def update_raw(self, task, raw_data):
        self.task[task].update({"raw": raw_data})

    def get_convert(self, task):
        return self.task[task]["convert"]

    def update_convert(self, task, converted):
        self.task[task].update({"convert": converted})

    def get_scene(self, task):
        return self.task[task]["scene"]

    def update_scene(self, task, scene):
        self.task[task].update({"scene": scene})

    def if_load_saved(self, task):
        if (
            self.get_path(task).endswith(".dict")
            or self.get_path(task).endswith(".xml")
        ) and self.get_raw(task) is None:
            return True
        else:
            return False


class RenderTool:
    def __init__(self, cfg):
        self.logger = cfg.logger
        self.save_path = cfg.save_path
        self.save_image_type = cfg.save_image_type
        self.task = dict()

        self.image_path = "image"

    def render(self, task, scene, update_element):
        self.logger.info(f"Rendering scene: {task}")
        scene = self.edit_scene(scene, update_element)
        render = mi.render(scene)
        self.update_render(task, render)

    def edit_scene(self, scene, update_element):
        assert isinstance(scene, mi.Scene)
        if update_element is None or len(update_element) == 0:
            return scene
        update_scene = build_plugin(
            {"type": "Scene", "assign": "scene", **update_element}
        )
        update_miscene = mi.load_dict(update_scene.inst())
        update_params = mi.traverse(update_miscene)
        params = mi.traverse(scene)
        for ele_name, ele_cfg in update_params.items():
            params[ele_name] = ele_cfg
        params.update()
        return scene

    def save(self, task):
        task_base_dir = os.path.dirname(task)
        mkdir_or_exist(os.path.join(self.save_path, self.image_path, task_base_dir))
        for tp in self.save_image_type:
            path = os.path.join(self.save_path, self.image_path, new_suffix(task, tp))
            self.logger.info(f"Saving image to {path}")
            mi.util.write_bitmap(path, self.get_render(task))

    def get_render(self, task):
        return self.task[task]["render"]

    def update_render(self, task, render):
        if task in self.task:
            self.task[task].update({"render": render})
        else:
            self.task.update({task: {"render": render}})


def dict_to_miscene(obj):
    _type = obj.get("_type")
    value = obj.get("value")
    if _type and value:
        module, cls = _type.rsplit(".", 1)
        if "mitsuba" in _type:
            # Extract module and class information
            if obj.get("variant") is not None:
                module_wt_variant = insert_after_substring(
                    module, f"{module}", f".{mi.variant()}"
                )
            else:
                module_wt_variant = module

            if "Transform4f" in cls:
                # If the class is Transform, create a matrix
                return getattr(sys.modules[module_wt_variant], "Transform4f")(value)
            # else:
            #     # Create an instance with the given value
            #     return getattr(sys.modules[module], cls)(value)

        # elif "drjit" in _type:
        #     # Create an instance with the given value
        #     return getattr(sys.modules[module], cls)(value)
    return obj


def miscene_to_dict(obj):
    _type = type(obj)
    module, cls = _type.__module__, _type.__name__
    if "mitsuba" in module:
        variant = mi.variant()
        module_wo_variant = module.replace(f".{variant}", "")
        if "Transform" in cls:
            value = np.array(obj.matrix).tolist()
        else:
            raise f"Unable to serialize type '{_type}' in mitsuba. Please implement the function."
        return {
            "_type": f"{module_wo_variant}.{cls}",
            "variant": variant if variant in module else None,
            "value": value,
        }
    # elif "drjit" in module:
    #     return {"_type": f"{module}.{cls}", "value": list(obj)}

    raise TypeError

from mitsuba_plugin import build_plugin
from mitsuba_util import FileTool, RenderTool
from transform import ChainTransform
from utils.misc import updated_dict


class Runner:
    def __init__(self, cfg):
        self.cfg = cfg
        self.logger = cfg.logger
        self.logger.info("=> Building mitsuba plugins ...")
        self.base_scene = build_plugin(self.cfg.base_scene)
        self.logger.info("=> Building transforms ...")
        self.pre_transform = self.build_transform(self.cfg.pre_transform)
        self.logger.info("=> Building specify parameters ...")
        self.specify = self.build_specify()
        self.logger.info("=> Initializing file tool ...")
        self.file_tool = FileTool(self.cfg)
        self.logger.info("=> Initializing render tool ...")
        self.render_tool = RenderTool(self.cfg)

    def run(self):
        self.logger.info("!!!!! Start Running !!!!!")
        for task in self.file_tool.get_task():
            info = f"------------ Processing task: {task} ------------"
            self.logger.info(info)
            if task in self.specify:
                self.logger.info(f"with specify parameters: {self.cfg.specify[task]}")

            self.file_tool.load_raw(task)
            if not self.file_tool.if_load_saved(task):
                self.logger.info("Transforming the raw data ...")
                transform = (
                    self.pre_transform
                    if task not in self.specify
                    else self.specify[task]["transform"]
                )
                self.file_tool.update_raw(task, transform(self.file_tool.get_raw(task)))
            else:
                self.logger.info(
                    f"Task '{task}' has saved data, skip the transform steps."
                )

            scene = (
                self.base_scene
                if task not in self.specify
                else self.specify[task]["scene"]
            )
            self.file_tool.convert(task, scene, self.cfg.element_shape)
            self.file_tool.dump(task)
            self.file_tool.parse(task)
            update_element = (
                self.specify[task]["update_element"]
                if self.file_tool.if_load_saved(task)
                else None
            )
            self.render_tool.render(
                task, self.file_tool.get_scene(task), update_element
            )
            self.render_tool.save(task)
            self.logger.info("-" * len(info))

    @staticmethod
    def build_transform(transform):
        return ChainTransform(transform.values())

    def build_specify(
        self,
    ):
        rtn = dict()
        for task, task_cfg in self.cfg.specify.items():
            self.logger.info(f"Building specify parameters for {task} ...")
            sp_scene = task_cfg.get("modify_scene", None)
            sp_transform = task_cfg.get("modify_transform", None)
            rtn[task] = {
                "scene": build_plugin(updated_dict(self.cfg.base_scene, sp_scene)),
                "transform": self.build_transform(
                    updated_dict(self.cfg.pre_transform, sp_transform)
                ),
                "update_element": dict(),
            }
            if sp_scene is not None:
                for name, value in task_cfg["modify_scene"].items():
                    old_value = self.cfg.base_scene.get(name, dict())
                    rtn[task]["update_element"].update(
                        {name: updated_dict(old_value, value)}
                    )
        return rtn

import os

import mitsuba as mi

from runner import Runner
from utils.config import argument_parser, config_parser
from utils.logger import get_logger


def main():
    args = argument_parser().parse_args()
    cfg = config_parser(args.config_file, args)
    logger = get_logger(
        name="root", log_file=os.path.join(cfg.save_path, "run.log"), file_mode="a"
    )
    logger.info(f"Running with args: {args}")
    logger.info(f"Running with config: {cfg}")
    cfg.logger = logger
    mi.set_variant(cfg.variant)

    runner = Runner(cfg)
    runner.run()


if __name__ == "__main__":
    main()

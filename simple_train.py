from deepforest import main
from pathlib import Path as P

PROJECT_DIR = P("/home/beaulima/crim/projects/effigis/effigis-src/DeepForest")
CONFIG_NAME = "deepforest_config.yml"
CONFIG_FILE = PROJECT_DIR.joinpath(CONFIG_NAME)
LABEL_DICT = {"Tree": 0}
NUM_CLASSES = len(LABEL_DICT)


m = main.deepforest(label_dict=LABEL_DICT, num_classes=NUM_CLASSES, config_file=CONFIG_FILE)

m.config["workers"] = 0

m.create_trainer()
m.use_release(check_release=True)
m.trainer.fit(m)
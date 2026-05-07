import sys
sys.path.append("../../")

from src.utils.dataloader import SP500WindowDataset
import yaml

with open("../../configs/ohlc_conditional.yaml") as f:
    config = yaml.safe_load(f)

data_cfg = config["data"]
ds = SP500WindowDataset(
    root_dir=config['train']["data_root"],
    seq_len=config['train']["seq_len"],
    stride=config['train']["stride"],
    columns=tuple(data_cfg.get("columns", ("date", "log_adj_close"))),
)
print(len(ds))

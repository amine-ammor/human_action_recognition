from pathlib import Path
from ploomber_engine.ipython import PloomberClient

dir_file = Path(__file__).parent
client = PloomberClient.from_path(str(dir_file.joinpath("../notebooks/keypoint_detector_usage.ipynb")),cwd="./")
paths = dir_file.joinpath("../data/inputs/kth_dataset/")
paths  = paths.rglob("./*.avi")
from tqdm import tqdm
for path in tqdm(paths):
    print(path)
    path_video_input = Path(path).resolve()
    client.execute(parameters=dict(path_video_input=path_video_input))
from pathlib import Path
from ploomber_engine.ipython import PloomberClient

dir_file = Path(__file__).parent
paths = dir_file.joinpath("../data/inputs/kth_dataset/")
paths  = paths.rglob("./*.avi")
from tqdm import tqdm
for path in tqdm(paths):
    print(path)
    path_video_input = Path(path).resolve()
    suffix = Path(path_video_input).suffix
    path_video_output = Path(str(path_video_input).replace("inputs","keypoints/on_videos").replace(suffix,"_out"+".mp4"))
    if not(path_video_output.exists()):
        try:
            client = PloomberClient.from_path(str(dir_file.joinpath("../notebooks/keypoint_detector_usage.ipynb")),cwd="./")
            client.execute(parameters=dict(path_video_input=path_video_input,
                                          path_video_output=path_video_output))
            del client
        except:
            pass
        finally:
            import torch
            t = torch.cuda.get_device_properties(0).total_memory
            r = torch.cuda.memory_reserved(0)
            a = torch.cuda.memory_allocated(0)
            f = r-a 
            print(f"remainning memory  is: {f}")

import torchvision.io as torch_io
import torch,numpy as np
from itertools import islice

from . import video_editor
def _read_and_split_video(path_video_input,truncate=None):
    frames,_,metadata = torch_io.read_video(path_video_input,output_format="TCHW")
    frames = torch.stack(list(islice(frames,truncate)))
    fps = metadata["video_fps"]
    return frames,fps    

class VideoIO:
    """
        class that takes read a video, and prepares it a sequence of batch to be fed to a neural
        network, handles the conversion to a torch tensor, and holds an instance of VideoEditor,
        to process a copy of the original frames, into new frames , and save this copy after
    """
    def __init__(self,path_video_input,truncate):
        self.path_video_input = path_video_input
        self.truncate = truncate

        self.torch_frames,self.fps = _read_and_split_video(self.path_video_input,self.truncate)
        self.np_frames = self.torch_frames.moveaxis(1,-1).numpy()
        
    
    def write(self,np_frames,path_output):
        assert type(np_frames) == np.ndarray
        frame_torch_8bit = torch.from_numpy(np_frames)
        torch_io.write_video(path_output,frame_torch_8bit,fps=self.fps)

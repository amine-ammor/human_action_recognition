from .video_io import VideoIO
from .video_editor import VideoEditor
import torch


class VideoHandler:
    """
        class that read and write a video, apply processing on its frames,and 
        prepare data to be fed to a neural network,
        for the targeted use-case , the video is supposed to fit in the memory while being red
    """
    def __init__(self,path_input,truncate=None,batch_size=5) -> None:
        self.truncate = truncate
        self.batch_size = batch_size
        self.video_io = VideoIO(path_input,truncate)
        self.video_editor = VideoEditor(self.video_io.np_frames)


    def get_loader(self,transforms,device):
        dset_of_frames = torch.utils.data.TensorDataset(transforms(self.video_io.torch_frames))
        dloader_of_frames = torch.utils.data.DataLoader(dset_of_frames,batch_size=self.batch_size,shuffle=False)
        dloader_of_frames = (batch[0].to(device) for batch in dloader_of_frames)
        return dloader_of_frames

    def write(self,path_output):
        self.video_io.write(self.video_editor.np_frames,path_output)
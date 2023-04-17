from pathlib import Path
import numpy as np
from torch.utils.data import Dataset
from functools import cached_property
from collections import OrderedDict
from itertools import islice
from pathlib import Path


from action_transformer import keypoints_io


def uniform_padding(array,max_shape,value):
    dim = len(array.shape) # of set of the arrays
    padding_width = ((0,max_shape-array.shape[0]),*[(0,0) for _ in range(dim-1)])
    new_array = np.pad(array,padding_width,constant_values=value)
    return new_array
        
    
def read_and_filter_keypoints(path_file,with_frame_padding,default_shape=(17,2),max_number_of_frames=None):
    if with_frame_padding:
        assert isinstance(max_number_of_frames,int),"a number of maximum frames must be provided"
        with_frame_padding = False
        keypoints,is_frame_present = read_and_filter_keypoints(path_file,with_frame_padding,default_shape,
                                                              max_number_of_frames)
        assert len(keypoints)<=max_number_of_frames
        assert len(is_frame_present)<=max_number_of_frames
        
        keypoints = uniform_padding(keypoints,max_number_of_frames,np.nan)
        is_frame_present = uniform_padding(is_frame_present,max_number_of_frames,False)
    else:
        keypoints,scores = keypoints_io.load(path_file)
        keypoints = [keypoints_frames[np.argmax(scores_frames)] if len(scores_frames)>0 else None for (keypoints_frames,scores_frames) in zip(keypoints,scores)]

        is_frame_present = np.array([el is not None for el in keypoints])
        keypoints = np.array([el if keep else np.nan*np.ones(default_shape)  for el,keep in zip(keypoints,is_frame_present)])
        keypoints = keypoints.astype("float32")
    keypoints = np.nan_to_num(keypoints)
    return keypoints,is_frame_present

def extract_class_name(path_file):
    if Path(path_file).parent.parent.name == "kth_dataset":
        class_name = Path(path_file).parent.name
        return class_name
    else:
        raise ValueError("class name is not defined")

class VideoKeyPointDataset(Dataset):
    memory = None # either None, of object of joblib.Memory
    def __init__(self,folder_video_keypoints,with_frame_padding=True,truncate=None,transform_labels=None):
        self.folder_video_keypoints = folder_video_keypoints
        self.with_frame_padding = with_frame_padding
        self.transform_labels = transform_labels
        self.video_keypoints_field_paths = list(islice(Path(self.folder_video_keypoints).rglob("*.npz"),truncate))
        if self.memory is not None:
            self.read_and_filter_keypoints = self.memory.cache(read_and_filter_keypoints)
        else:
            self.read_and_filter_keypoints = read_and_filter_keypoints
    
    @cached_property
    def number_of_frames(self):
        assert not(self.with_frame_padding),"can't get the exact number of frames if padding is activated"
        res = {i:len(self.__getitem__(i)["kpts"]) for i in range(len(self))}
        return res
    
    @cached_property
    def max_number_of_frames(self):
        res = max(self.number_of_frames.values())
        return res

    
    def __getitem__(self,i):
        outputs = OrderedDict()
        path_file = self.video_keypoints_field_paths[i]
        if self.with_frame_padding:
            assert "max_number_of_frames" in self.__dict__,"""
            max_number_of_frames should be computed at least once,with self.with_frame_padding
            set to False before using padding 
            """
            kpts,is_detection_present = self.read_and_filter_keypoints(path_file,
                                                          with_frame_padding=self.with_frame_padding,
                                                          max_number_of_frames=self.max_number_of_frames)
            outputs["number_of_frames"]=self.number_of_frames[i]
        else:
            kpts,is_detection_present = self.read_and_filter_keypoints(path_file,
                                                          with_frame_padding=self.with_frame_padding,
                                                              max_number_of_frames=None)
        outputs.update(kpts=kpts,
                      is_detection_present=is_detection_present)
        outputs["class_name"] = extract_class_name(path_file)
        if self.transform_labels is not None:
            outputs["class_name"] = self.transform_labels(outputs["class_name"])
        return outputs
    
    def __len__(self):
        return len(self.video_keypoints_field_paths)
    

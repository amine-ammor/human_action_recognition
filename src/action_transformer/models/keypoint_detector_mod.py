import torch
from torch import nn
import torchvision.io as torch_io
from torchvision.models.detection import keypointrcnn_resnet50_fpn,KeypointRCNN_ResNet50_FPN_Weights


# class NoDetectionError(Exception):
#     """raise this error if there were no detection in the frames"""

class KeyPointDetector(nn.Module):
    """wrapper module around the pytorch keypoint detector
        that applies the keypoint detection on frame basis
    """
    def __init__(self,device,threshold):
        super().__init__()
        weights = KeypointRCNN_ResNet50_FPN_Weights.COCO_V1
        self.keypoint_detector = keypointrcnn_resnet50_fpn(weights=weights).eval()
        self.transforms = weights.transforms()
        self.keypoint_detector.to(device)
        self.threshold = threshold
    
    def _filter_detection(self,key_pts_per_detection):
        filtered_detection = key_pts_per_detection["scores"]<self.threshold
        res = [(values,scores) for values,scores,filtered 
                                in zip(key_pts_per_detection["keypoints"],
                                       key_pts_per_detection["scores"]
                                       ,filtered_detection) if not(filtered)]
        if len(res) == 0:
            keypoints = torch.zeros(0,0,0)
            scores = torch.zeros(0)
        else:
            keypoints,scores =  list(zip(*res))
            keypoints = torch.stack(keypoints)[...,:2]
            scores = torch.stack(scores)
        return keypoints,scores

    def forward(self,batch):
        """
            same input as the input of the pose detector
            batch of shape (N,C,H,W)
            returns list of detections per each frame, 
        """
        detections = self.keypoint_detector(batch)
        filtered_detection = [self._filter_detection(detection) for detection in detections]
        key_pts_per_frame,scores_pre_frame = list(zip(*filtered_detection))
        return key_pts_per_frame,scores_pre_frame

    
    @torch.no_grad()
    def predict(self,batch_frame):
        key_pts_per_frame,scores_pre_frame = self(batch_frame)
        key_pts_per_frame = [keypoints_per_detection.cpu().detach().numpy().astype("int32") 
                                 for keypoints_per_detection in key_pts_per_frame ]
        scores_pre_frame = [score.cpu().detach().numpy() for score in scores_pre_frame]
        return key_pts_per_frame,scores_pre_frame

    @property
    def device(self):
        return next(self.parameters()).device

    @torch.no_grad()
    def predict_on_whole_video(self,video_handler):
        """
            video_editor object of VideoEditor class that can draw a set of keypoints on an image
        """
        frames_as_batches = video_handler.get_loader(self.transforms,device=self.device)
        outputs = [self.predict(batch_frame) for batch_frame in frames_as_batches]
        # import pdb;pdb.set_trace()
        unbatched_outputs = [(keypts,score) for (batch_keypts,batch_scores) in outputs for (keypts,score) in zip(batch_keypts,batch_scores)]
        key_pts_per_frame,scores_pre_frame = zip(*unbatched_outputs)
        video_handler.video_editor.draw(key_pts_per_frame)
        return key_pts_per_frame,scores_pre_frame
import cv2
import time
import numpy as np
import os
from tqdm import tqdm
import pandas as pd
import csv
import json
import pdb

class ComputeOpticalFlow:
    """This class is computing optical flow of given video."""
    def __init__(self, video): 
        self.video=video

    # Transform Video to .npy Format
    def get_optical_flow(self):
        """Calculate dense optical flow of input video
        Args:
            video: the input video with shape of [frames,height,width,channel]. dtype=np.array
        Returns:
            flows_x: the optical flow at x-axis, with the shape of [frames,height,width,channel]
            flows_y: the optical flow at y-axis, with the shape of [frames,height,width,channel]
        """
        # initialize the list of optical flows
        video=(self.video)
        gray_video = []
        for i in range(len(video)):
            img = cv2.cvtColor(video[i], cv2.COLOR_RGB2GRAY)
            gray_video.append(np.reshape(img,(224,224,1)))

        flows = []
        for i in range(0,len(video)-1):
            # calculate optical flow between each pair of frames
            flow = cv2.calcOpticalFlowFarneback(
                gray_video[i], gray_video[i+1], None, 0.5, 3, 15, 3, 5, 1.2, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
            # subtract the mean in order to eliminate the movement of camera
            flow[..., 0] -= np.mean(flow[..., 0])
            flow[..., 1] -= np.mean(flow[..., 1])
            # normalize each component in optical flow
            flow[..., 0] = cv2.normalize(flow[..., 0],None,0,255,cv2.NORM_MINMAX)
            flow[..., 1] = cv2.normalize(flow[..., 1],None,0,255,cv2.NORM_MINMAX)
            # Add into list
            flows.append(flow)

        # Padding the last frame as empty array
        flows.append(np.zeros((224,224,2)))

        return np.array(flows, dtype=np.float32)


class VideoHandler:
    """ Implementing this class we can convery a Video to npy format and Transfer all the videos and save it to targeted directory"""
    def __init__(self, file_path, file_dir, save_dir):
        """ Initialisation function
        Arguments:
            file path=video_path
            file_dir: source folder of target videos
            save_dir: destination folder of output .npy files
        """
        self.file_path=file_path
        self.file_dir=file_dir
        self.save_dir=save_dir
    
    def video_to_npy(self, resize=(224,224)):
        """Load video and tansfer it into .npy format
        Args:
            file_path: the path of video file
            resize: the target resolution of output video
        Returns:
            frames: gray-scale video
            flows: magnitude video of optical flows
        """
        # Load video
        file_path=self.file_path
        cap = cv2.VideoCapture(file_path)
        # Get number of frames
        len_frames = int(cap.get(7))
        # Extract frames from video
        try:
            frames = []
            for i in range(len_frames-1):
                _, frame = cap.read()
                frame = cv2.resize(frame,resize, interpolation=cv2.INTER_AREA)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = np.reshape(frame, (224,224,3))
                frames.append(frame)
        except Exception:
            print("Error: ", file_path, len_frames,i)
        finally:
            frames = np.array(frames)
            cap.release()

        # Get the optical flow of video
        obj=ComputeOpticalFlow(frames)
        flows=obj.get_optical_flow()
        result = np.zeros((len(flows),224,224,5))
        result[...,:3] = frames
        result[...,3:] = flows

        return result

    def save_to_npy(file_dir, save_dir):
        """Transfer all the videos and save them into specified directory
        Args:
            file_dir: source folder of target videos
            save_dir: destination folder of output .npy files
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            # List the files
        videos = os.listdir(file_dir)
        for v in tqdm(videos):
            # Split video name
            video_name = v.split('.')[0]
            # Get src
            video_path = os.path.join(file_dir, v)
            # Get dest
            save_path = os.path.join(save_dir, video_name+'.npy')
            # Load and preprocess video
            obj=VideoHandler(video_path,"","")
            data = obj.video_to_npy(resize=(224,224))
            data = np.uint8(data)
            # Save as .npy file
            np.save(save_path, data)

        return None


if __name__=="__main__":
    src_path = '/home/ameya/Documents/programming/pyhton/videos.json' #path of json file. [clip (.mp4):label]
    target_path = '/home/ameya/Documents/programming/pyhton'
    filename = 'videos_npy.json'
    os.makedirs(target_path,exist_ok=True)

    annotations = json.load(open(src_path)) # [clip(.mp4):label]
    f_c = sum(x=='violent' for x in annotations.values())
    n_c = sum(x=='non-violent' for x in annotations.values())
    
    npy_dict = dict()
    for source, label in tqdm(annotations.items()):
        if(label=="violent" or label =="non-violent"):
            save_path = os.path.join(target_path, os.path.basename(source).replace('.mp4','.npy')) # ./npy_dir/abc.npy
            npy_dict[save_path] = label
        else:
            continue
      
        obj=VideoHandler(source,src_path,target_path)
        data=obj.video_to_npy(resize=(224,224))
        data = np.uint8(data)
        # Save as .npy file
        np.save(save_path, data)
    json.dump(npy_dict, open(filename, "w"), indent=4)
    
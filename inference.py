import tensorflow as tf
import json
import numpy as np
import cv2
from glob import glob
#from blur import blur_video
import time
import glob
import os
from sklearn.metrics import classification_report

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--infile', help="Path of Profile json file", default="profileInfo.json")
args = parser.parse_args()

class ComputeNormalisedData:
    "This class normalises the data which is given as input"
    def __init__(self,data):
        self.data=data
    
    def normalize(self):
        data=self.data
        mean = np.mean(data)
        std = np.std(data)
        return (data-mean)/std


class ComputeOpticalFlow:
    "This class computes the optical flow of the video"
    def __init__(self, video): 
        self.video=video
        
    def get_optical_flow(self, shape=(224, 224)):
        """Calculate dense optical flow of input video
        Args:
            video: the input video with shape of [frames,height,width,channel]. dtype=pred_label = prediction_label(pred, th)np.array
        Returns:
            flows_x: the optical flow at x-axis, with the shape of [frames,height,width,channel]
            flows_y: the optical flow at y-axis, with the shape of [frames,height,width,channel]
        """
        # initialize the list of optical flows
        video=self.video
        gray_video = []
        for i in range(len(video)):
            img = cv2.cvtColor(video[i], cv2.COLOR_RGB2GRAY)
            gray_video.append(np.reshape(img,(*shape,1)))

        flows = []
        for i in range(0,len(video)-1):
            # calculate optical flow between each pair of frames
            flow = cv2.calcOpticalFlowFarneback(gray_video[i], gray_video[i+1], None, 0.5, 3, 15, 3, 5, 1.2, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
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

class VideoToFramesConverter:
    "Converts given video to frames and frames_window based on the function given"
    def __init__(self,file_path):
        self.file_path=file_path
       
    def video_to_frames(self, resize=(224,224), window_size=64, hop_size=32):
        file_path=self.file_path
        # Load video
        cap = cv2.VideoCapture(file_path)
        # Get number of frames
        len_frames = min(int(cap.get(7)), max_frames+1)
        # Extract frames from video
        try:
            frames = []
            for i in range(len_frames-1):
                _, frame = cap.read()
                frame = cv2.resize(frame,resize, interpolation=cv2.INTER_AREA)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = np.reshape(frame, (224,224,3))
                frames.append(frame)
        except:
            print("Error: ", file_path, len_frames,i)
        finally:
            frames = np.array(frames)
            cap.release()

        # Get the optical flow of video
        obj1=ComputeOpticalFlow(frames)
        flows = obj1.get_optical_flow()
        obj1=ComputeNormalisedData(frames)
        obj2=ComputeNormalisedData(flows)
        result = np.zeros((len(flows),224,224,5))
        result[...,:3] = obj1.normalize()
        result[...,3:] = obj2.normalize()

        return result

    def video_to_frames_window(self, shape=(224,224), window_size=64, hop_size=32):
        file_path=self.file_path
        cap = cv2.VideoCapture(file_path)
        frames = []
        success, frame = cap.read()
        while success:
            frame = cv2.resize(frame, shape, interpolation=cv2.INTER_AREA)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = np.reshape(frame, (*shape,3))
            frames.append(frame)
            if len(frames)>=window_size:
                obj1=FramesToNumpyConverter(frames)
                yield obj1.convert_frames(shape=shape)
                frames = frames[hop_size:]
            success, frame = cap.read()
        if frames:
            m, r = window_size//len(frames), window_size%len(frames)
            frames = frames*m + frames[:r]
            obj1=FramesToNumpyConverter(frames)
            yield obj1.convert_frames(shape=shape)
        cap.release()
        
class FramesToNumpyConverter:
    "Converts frames to Numpy"      
    def __init__(self,frames):
        self.frames=frames
    
    def convert_frames(self, shape=(224, 224)):
        frames=self.frames
        frames_np = np.array(frames)
        # print(frames_np.shape)
        obj1=ComputeOpticalFlow(frames)
        flows = obj1.get_optical_flow(shape)
        result = np.zeros((len(flows),*shape,5))
        obj1=ComputeNormalisedData(frames)
        obj2=ComputeNormalisedData(flows)
        result[...,:3] = obj1.normalize()
        result[...,3:] = obj2.normalize()
        return result

class ComputeBlurredVoilence:
    "This class blurrs the voilence of given Video"
    def __init__(self,video_path,model):
            self.video_path=video_path
            self.model=model
    
    def blur_violence(self):
        video_path=self.video_path
        model=self.model
        window_size = 64
        hop_size = 32
        st, et = 0, window_size
        vid = cv2.VideoCapture(video_path)
        fps = vid.get(cv2.CAP_PROP_FPS)
        vid.release()
        output = dict()
        hop_size = 64
        violence_threshold = 0.1
        cnt = 0
        prob = 0
        basename = os.path.splitext(os.path.basename(video_path))[0]
        dict_ = {}
        obj1=VideoToFramesConverter(video_path)
        for d in obj1.video_to_frames_window(window_size=window_size, hop_size=hop_size):
            #print(st/fps, et/fps)
            #st += hop_size
            #et += hop_size
            pred = model.predict(np.expand_dims(d, axis=0))
            prob += pred[0][0]
            cnt += 1
            output[(st, et)] = pred[0]
            print(st/fps, et/fps, pred[0])
            t = []
            t.append(st/fps)
            t.append(et/fps)
            dict_[str(t)] = pred[0].tolist()
            st += hop_size
            et += hop_size
        return dict_

    # timestamps = [[]]
    # for (s_t, e_t), v in output.items():
    #     if v[0]>=violence_threshold:
    #         if len(timestamps[-1])==0:
    #             timestamps[-1].extend((s_t/fps, e_t/fps))
    #         else:
    #             timestamps[-1][-1] = e_t/fps
    #     else:
    #         if timestamps[-1]:
    #             timestamps[-1][-1] = s_t/fps
    #             timestamps.append([])
    # if len(timestamps[-1]) == 0:
    #     timestamps.pop()
    # print(*timestamps)

    # return timestamps

class ComputePrediction:
    "This class computes the prediction"
    def __init__(self,pred_dict,th):
            self.pred_dict=pred_dict
            self.th=th
    
    def prediction_label(self):
        pred_dict=self.pred_dict
        th=self.th
        for timestamp, scores in pred_dict.items():
            if(scores[0] > th):
                return "fight" 
        return "nonfight"


if __name__ == '__main__':
    path = args.infile
    profileInfo = json.load(open(path))
    model_path = profileInfo["model"]
    th = float(profileInfo["th"])
    data_loading = profileInfo["data_loading"]
    print(f"model path: {model_path}, th: {th}")
    print(f"data_loading: {data_loading}")
    dirpath = profileInfo["dirpath"]
    jsonpath = profileInfo["jsonPath"]
    print(f"dirpath: {dirpath}")
    print(f"json: {jsonpath}")
    print("\n\n")

    pred_filename = "test_data_predictions.json"
    model = tf.keras.models.load_model(model_path)

    if data_loading == "dir":
        test_dir = profileInfo["dirpath"]
        dirname = test_dir
        pred_filepath = os.path.join(dirname, pred_filename)
        fight_clips = glob.glob(os.path.join(dirname, "fight/*.mp4"))
        nonfight_clips = glob.glob(os.path.join(dirname, "nonfight/*.mp4"))

    if data_loading == "json":
        json_filepath = profileInfo["jsonPath"]
        dirname = os.path.dirname(json_filepath)
        pred_filepath = os.path.join(dirname, pred_filename)
        test_clips = json.load(open(json_filepath))
        fight_clips = [clips for clips in test_clips if test_clips[clips] == "violent"]
        nonfight_clips = [clips for clips in test_clips if test_clips[clips] == "non-violent"]
    
    print("\nTest Data Stats Below:")
    print(f"Fight Clips: {len(fight_clips)}, NonFight Clips: {len(nonfight_clips)}\n")

    output = {}
    tp, tn, fp, fn = 0, 0, 0, 0

    for clip in fight_clips:
        print("\nProcessing:", clip)
        obj=ComputeBlurredVoilence(clip,model)
        pred = obj.blur_violence()
        # prediction label return fight if fight score is greater than th for any timestamp. else return nonfight
        obj=ComputePrediction(pred,th)
        pred_label = obj.prediction_label()
        if(pred_label == "fight"):
            tp += 1
        else:
            fn += 1
        basename = os.path.basename(clip)
        output[basename] = {"True_Class": "fight", "Pred_Class": pred_label}


    for clip in nonfight_clips:
        print("\nProcessing:", clip)
        obj=ComputeBlurredVoilence(clip,model)
        pred = obj.blur_violence()
        # prediction label return fight if fight score is greater than th for any timestamp. else return nonfight
        obj=ComputePrediction(pred,th)
        pred_label = obj.prediction_label()
        if(pred_label == "fight"):
            fp += 1
        else:
            tn += 1
        basename = os.path.basename(clip)
        output[basename] = {"True_Class": "nonfight", "Pred_Class": pred_label}

    acc = ((tp + tn)/(tp + tn + fp + fn))*100.0
    print(f"\n\nModel's Report:")
    print(f"TP: {tp} \t FP: {fp}\nFN: {fn} \t TN: {tn}\n\nAccuracy: {acc}")
    json.dump(output, open(pred_filename, "w"), indent=4)

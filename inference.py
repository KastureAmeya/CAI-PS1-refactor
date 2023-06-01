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

def normalize(data):
    mean = np.mean(data)
    std = np.std(data)
    return (data-mean)/std

def get_optical_flow(video, shape=(224, 224)):
    """Calculate dense optical flow of input video
    Args:
        video: the input video with shape of [frames,height,width,channel]. dtype=np.array
    Returns:
        flows_x: the optical flow at x-axis, with the shape of [frames,height,width,channel]
        flows_y: the optical flow at y-axis, with the shape of [frames,height,width,channel]
    """
    # initialize the list of optical flows
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


def video_to_frames(file_path, resize=(224,224), window_size=64, hop_size=32):
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
    flows = get_optical_flow(frames)

    result = np.zeros((len(flows),224,224,5))
    result[...,:3] = normalize(frames)
    result[...,3:] = normalize(flows)

    return result

def convert_frames(frames, shape=(224, 224)):
    frames_np = np.array(frames)
    # print(frames_np.shape)
    flows = get_optical_flow(frames, shape)
    result = np.zeros((len(flows),*shape,5))
    result[...,:3] = normalize(frames)
    result[...,3:] = normalize(flows)
    return result

def video_to_frames_window(file_path, shape=(224,224), window_size=64, hop_size=32):
    cap = cv2.VideoCapture(file_path)
    frames = []
    success, frame = cap.read()
    while success:
        frame = cv2.resize(frame, shape, interpolation=cv2.INTER_AREA)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = np.reshape(frame, (*shape,3))
        frames.append(frame)
        if len(frames)>=window_size:
            yield convert_frames(frames, shape=shape)
            frames = frames[hop_size:]
        success, frame = cap.read()
    if frames:
        m, r = window_size//len(frames), window_size%len(frames)
        frames = frames*m + frames[:r]
        yield convert_frames(frames, shape=shape)
    cap.release()

def blur_violence(video_path, model):
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
    for d in video_to_frames_window(video_path, window_size=window_size, hop_size=hop_size):
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
def prediction_label(pred_dict, th):
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
        pred = blur_violence(clip, model)
        # prediction label return fight if fight score is greater than th for any timestamp. else return nonfight
        pred_label = prediction_label(pred, th)
        if(pred_label == "fight"):
            tp += 1
        else:
            fn += 1
        basename = os.path.basename(clip)
        output[basename] = {"True_Class": "fight", "Pred_Class": pred_label}


    for clip in nonfight_clips:
        print("\nProcessing:", clip)
        pred = blur_violence(clip, model)
        # prediction label return fight if fight score is greater than th for any timestamp. else return nonfight
        pred_label = prediction_label(pred, th)
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

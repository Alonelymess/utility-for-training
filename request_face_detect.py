import argparse
import base64
import json
import time
import concurrent.futures
from threading import Thread
import numpy as np
import queue as q

import cv2
import requests
from collections import deque
import threading

def read_frames(cam, queue):
    # Check if video opened successfully
    cam_id = cam['id']
    cap = cam['cap']
    print('Cam ID: ', cam_id)
    if cap.isOpened() == False:
        print("Error opening video stream or file")
        return

    frame_cnt = 0

    # Read until video is completed
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            
            # Preprocess the frame
            # image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = cv2.resize(frame, (640, 640), interpolation = cv2.INTER_LINEAR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            
            # Encode the frame into byte data
            data = cv2.imencode(".jpg", image)[1].tobytes()
            queue.append({'id': cam_id,
                            'size': frame.shape,
                            'frame': frame_cnt,
                            'image': data})
            frame_cnt += 1

            # For videos, add a sleep so that we read at 30 FPS
            if not isinstance(cam_id, int):
                time.sleep(1.0 / 30)

        # Break the loop
        else:
            break

    print("Done reading {} frames".format(frame_cnt))

    # When everything done, release the video capture object
    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()
    return

def spawn_camera(device):
    if isinstance(device, int):
        device = [device]
        
    for i in range(len(device)):
        try:
            # print(device[i])
            device[i] = int(device[i])
        except:
            pass
    
    cams = [{'id': cam, 
             'cap': cv2.VideoCapture(cam)} 
            for cam in device]

    return cams


def convert_arg(args):

    # Create a VideoCapture object and read from input file
    # If the input is the camera, pass 0 instead of the video file name


    try:
        device = int(args.input)
    except:
        try:
            device = args.input.split(',')
        except:
            device = args.input


    
    # print(device, type(device))
        
    
    cams = spawn_camera(device)
    # threads = [Thread(target=read_frames, args=(cam,)) for cam in cams]
    # [t.start() for t in threads]
    # [t.join() for t in threads]
    return cams
    
    
    


def send_frames(payload, snd_cnt):


    api = "http://127.0.0.1:8080/predictions/face-client"
    snd_cnt += len(payload)
    payload = json.dumps(payload)
    
    response = requests.post(api, data=payload, headers=headers)

    return (response, snd_cnt)
    


def calculate_fps(start_time, snd_cnt):

    end_time = time.time()

    fps = 1.0 * args.batch_size / (end_time - start_time)


    print(
        "With Batch Size {}, FPS at frame number {} is {:.1f}".format(
            args.batch_size, snd_cnt, fps
        )
    )
    return fps


def batch_and_send_frames(args, queue):

    # Initialize variables
    count, exit_cnt, snd_cnt, log_cnt = 0, 0, 0, 20
    payload, futures = {}, []
    start_time = time.time()
    fps = 0


    while True:

        # Exit condition for the while loop. Need a better logic
        if len(queue) == 0:
            exit_cnt += 1
            # By trial and error, 1000 seemed to work best
            if exit_cnt >= 1000:
                print(
                    "Length of queue is {} , snd_cnt is {}".format(len(queue), snd_cnt)
                )
                break

       
        # Batch the frames into a dict payload
        while queue and count < args.batch_size:
            data = queue.popleft()
            img = data['image']
            im_b64 = base64.b64encode(img).decode("utf8")
            
            payload[str(count)] = {'id': data['id'], 
                                    'size': data['size'],
                                    'frame': data['frame'],
                                    'image': im_b64}
            count += 1
            
            
        

        if count >= args.batch_size:
            response, snd_cnt = send_frames(payload, snd_cnt)

            results = json.loads(response.content.decode("UTF-8"))
            
            print(results)
            

            # Reset for next batch
            start_time = time.time()
            payload = {}
            count = 0
        

        # Sleep for 10 ms before trying to send next batch of frames
        time.sleep(args.sleep)

    

    return


def read_and_send(cam):
    """
    Parallelized function.
    """
    
    # Read frames are placed here and then processed
    queue = deque([])
    
    # Start the read_frames worker
    read_worker_thread = threading.Thread(target=read_frames, args=(cam, queue))
    read_worker_thread.daemon = True
    read_worker_thread.start()
    
    # Start the batch_and_send_frames worker
    batch_and_send_thread = threading.Thread(target=batch_and_send_frames, args=(args, queue))
    batch_and_send_thread.daemon = True
    batch_and_send_thread.start()
    
    # # Start the send_keypoints worker
    # send_keypoints_thread = threading.Thread(target=send_keypoints, args=(keypoints_queue,))
    # send_keypoints_thread.daemon = True
    # send_keypoints_thread.start()
    
    
    batch_and_send_thread.join()
    read_worker_thread.join()
    # send_keypoints_thread.join()
        
    

if __name__ == "__main__":

    try:
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--batch_size",
            help="Batch frames on TorchServe side for inference",
            type=int,
            default=1,
        )
        parser.add_argument(
            "--input",
            help="Path to video file or device id",
            default=r"videos\cut.mp4",
        )
        parser.add_argument(
            "--sleep",
            help="Sleep between 2 subsequent requests in seconds",
            type=float,
            default=0.01,
        )
        
        start_time = time.time()
        
        args = parser.parse_args()

        headers = {"Content-type": "application/json", "Accept": "text/plain"}
        
        cams = convert_arg(args)
        
        print('Cams: ', cams)
        # Start the read and send process parallelly use thread
        threads = [Thread(target=read_and_send, args=(cam,)) for cam in cams]
        
        [t.start() for t in threads]
        [t.join() for t in threads]
        
        print('Time: ', time.time() - start_time)
    except Exception as e:
        print(e)
        pass

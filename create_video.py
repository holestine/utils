import os, cv2
import numpy as np
 
def create_video(out_dir, filename, is_rgb = True):
    '''
    Creates a video 
 
    Parameters
    ----------
    out_dir : str
        The directory to store the video
    filename : str
        The name of the video
    is_rgb : bool
        Flag to specify whether or not the video will be in color
    '''
 
    fps = 5
    width = 1200
    height = 800
    
    # Initialize the video writer
    video_path = os.path.join(out_dir, filename)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height), is_rgb)

    # Create a bunch of frames
    for i in range(20):
        
        img = np.zeros((height, width, 3), np.uint8)
        
        # Write the frame number on the image
        img = cv2.putText(img, str(i), (width-100, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)

        cv2.circle(img,(10*i,20*i), 10, (0,255,255), -1)

        cv2.imshow('img', img)

        out.write(img)

    # Save the video
    out.release()
 
create_video('out', 'video.mp4')

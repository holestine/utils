import pandas as pd
import os

# WFLW to 300W Mappings taken from observations
# Jaw                  0..32 -> 0..16
# Left Eyebrow         33..41 -> 17..21
# Right Eyebrow        42..50 -> 22..26
# Nose                 51..59 -> 27..35
# Left Eye             60..67 -> 36..41
# Right Eye            68..75 -> 42..47
# Top of Top Lip       76..82 -> 48..54
# Bottom of Bottom Lip 83..87 -> 55..59 
# Bottom of Top Lip    88..92 -> 60..64
# Top of Bottom Lip    93..95 -> 65..67
# Left Pupil           96
# Right  Pupil         97

JAW_BEGIN           = 0
JAW_END             = 32
LEFT_EYEBROW_BEGIN  = 33
RIGHT_EYEBROW_BEGIN = 42
NOSE_BEGIN          = 51
NOSE_END            = 59
LEFT_EYE_BEGIN      = 60
LEFT_EYE_END        = 67
LEFT_EYE_IGNORES    = [62, 66]
RIGHT_EYE_BEGIN     = 68
RIGHT_EYE_END       = 75
RIGHT_EYE_IGNORES   = [70, 74]
MOUTH_BEGIN         = 76 
MOUTH_END           = 95

def convert_wflw_row(wflw_pts):
    """
    Convert a single row of WFLW style facial landmarks to 300W style landmarks.

    Args:
        wflw_pts (float[]): Array of values representing x,y coordinates for WFLW landmarks
    
    Returns:
        converted_pts (float[]): Array of values representing x,y coordinates for 300W landmarks
    """

    converted_pts = []

    # JAW: include every other point which includes the extremities
    for i in range(JAW_BEGIN, JAW_END+1, 2):
        converted_pts.extend([wflw_pts[2*i], wflw_pts[2*i+1]])

    # Right Eyebrow: keep top portion
    for i in range(RIGHT_EYEBROW_BEGIN, RIGHT_EYEBROW_BEGIN+5, 1):
        converted_pts.extend([wflw_pts[2*i],wflw_pts[2*i+1]])

    # Left Eyebrow: keep top portion
    for i in range(LEFT_EYEBROW_BEGIN, LEFT_EYEBROW_BEGIN+5, 1):
        converted_pts.extend([wflw_pts[2*i],wflw_pts[2*i+1]])

    # Nose: keep everything 
    for i in range(NOSE_BEGIN, NOSE_END+1, 1):
        converted_pts.extend([wflw_pts[2*i],wflw_pts[2*i+1]])

    # Right Eye: ignore some indices
    for i in range(RIGHT_EYE_BEGIN, RIGHT_EYE_END+1, 1):
        if i not in RIGHT_EYE_IGNORES:
            converted_pts.extend([wflw_pts[2*i],wflw_pts[2*i+1]])

    # Left Eye: ignore some indices
    for i in range(LEFT_EYE_BEGIN, LEFT_EYE_END+1, 1):
        if i not in LEFT_EYE_IGNORES:
            converted_pts.extend([wflw_pts[2*i],wflw_pts[2*i+1]])

    # Mouth: keep everything
    for i in range(MOUTH_BEGIN, MOUTH_END+1, 1):
        converted_pts.extend([wflw_pts[2*i],wflw_pts[2*i+1]])

    return converted_pts

def convert_wflw_file(wflw_csv, output_path):
    """
    Convert WFLW style facial landmarks to 300W style landmarks and save results.

    Args:
        wflw_csv (str): Path to WFLW CSV
        output_path (str): Path to output 300W CSV
    """

    # Read wflw style annotations into dataframe
    wflw_df = pd.read_csv(wflw_csv, header=0, sep=',')

    # Convert a row at a time inplace.
    for row in range(len(wflw_df)):

        # Get a row of WFLW style annotations
        wflw_xy_points = wflw_df.iloc[row].values[1:]

        # Convert and replace old values with new values
        converted_pts = convert_wflw_row(wflw_xy_points)
        wflw_df.iloc[row, 1:len(converted_pts)+1] = converted_pts

    # Now that all conversions are done remove the remaining wflw points
    df = wflw_df.drop(columns=wflw_df.columns[len(converted_pts)+1:])

    # Save the converted dataframe
    os.makedirs(os.path.dirname(output_path), exist_ok=True) 
    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    convert_wflw_file('raw_data/WFLW_landmarks_test.csv',  'dataset/300W_landmarks_test.csv')
    #convert_wflw_file('raw_data/WFLW_landmarks_train.csv', 'dataset/300W_landmarks_train.csv')

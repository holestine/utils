import pandas as pd
import os

# Constants for 300W facial landmarks
#RIGHT_EYEBROW_BEGIN = 22
#LEFT_EYEBROW_END    = 21
#TOP_OF_NOSE = 27
#CENTER_BOTTOM_LIP = 57

# Constants for WFLW facial landmarks
RIGHT_EYEBROW_BEGIN = 42
LEFT_EYEBROW_END    = 41
TOP_OF_NOSE = 51
CENTER_BOTTOM_LIP = 85

def normalize_xy_values(x_values, y_values):
    """
    Normalize x and y values for 300W style facial landmarks.

    Args:
        x_values (float[]): Array of values representing x coordinates for 300W landmarks
        y_values (float[]): Array of values representing y coordinates for 300W landmarks
    
    Returns:
        x_values(float[]): The normalized x values for 300W landmarks
        y_values(float[]): The normalized y values for 300W landmarks
    """

    # Correct rotations
    if x_values[LEFT_EYEBROW_END] < x_values[RIGHT_EYEBROW_BEGIN]: # x is flipped
        x_values = -x_values
    if y_values[TOP_OF_NOSE] < y_values[CENTER_BOTTOM_LIP]: # y is flipped
        y_values = -y_values

    # Normalize x and y values
    x_values = x_values - x_values.min()
    x_values = x_values / x_values.max()
    y_values = y_values - y_values.min()
    y_values = y_values / y_values.max()

    return x_values, y_values

def normalize_file(csv, output_path):
    """
    Normalize 300W facial landmarks and save results.

    Args:
        csv (str): Path to 300W CSV
        output_path (str): Path to normalized output 300W CSV
    """

    # Read annotations into dataframe
    df=pd.read_csv(csv, header=0, sep=',')

    # Convert a row at a time inplace.
    for row in range(len(df)):
        # Seperate x and y values
        x_values = df.iloc[row, 1::2].values
        y_values = df.iloc[row, 2::2].values

        # Normalize x and y values
        x_values, y_values = normalize_xy_values(x_values, y_values)
        
        # Recombine x and y values
        vals = []
        for i in range(len(x_values)):
            vals.extend([x_values[i], y_values[i]])

        # Could alternatively calculate vals like this but it's less readable
        #vals = [val for pair in zip(x_values, y_values) for val in pair]
        
        # Replace old values with normalized values
        df.iloc[row, 1:len(vals)+1] = vals

    # Save normalized dataframe
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    

if __name__ == "__main__":
    normalize_file('raw_data/WFLW_landmarks_test.csv', 'dataset/WFLW_normalized_landmarks_test.csv')


    
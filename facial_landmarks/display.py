import pandas as pd
from matplotlib import pyplot as plt

def plot(ax, x, y, color, txt="", alpha=1.0):
    """
    Plots an individual point

    Args:
        ax (object): Subplot for target figure
        x (float): x coordinate
        y (float): y coordinate
        color (bool): Color of glyph
        txt (str): Text associated with point 
        alpha (float): Opaqueness of glyph (set to 0.0 for transparent)
    """
    ax.plot(x, y, color, alpha=alpha)
    ax.text(x, y, txt)

def display_csv(lhs_csv, rhs_csv, figTitle="", copy_lhs=False, hide_axis=True):
    """
    Display facial landmarks from two seperate csv files

    Args:
        lhs_csv (str): Path to csv to display on the left hand side
        rhs_csv (str): Path to csv to display on the right hand side
        figTitle (str): Title for the figure
        copy_lhs (bool): Copy the left hand side glyphs to the right hand side (does not include text)
        hide_axis (bool): Hides the x and y axis 
    """

    # Read the two csv files into dataframes
    lhs_df=pd.read_csv(lhs_csv, header=0, sep=',')
    rhs_df=pd.read_csv(rhs_csv, header=0, sep=',')

    if len(lhs_df) != len(rhs_df):
        print("Dataset lengths are mismatched. Exiting...")
        exit(0)

    for row in range(len(lhs_df)):
        # Create figure for plotting
        fig, (ax1, ax2) = plt.subplots(1, 2, num="Row {}".format(row+1))
        fig.suptitle(figTitle)

        # Get xy coords for each side
        lhs_xy_list = lhs_df.iloc[row].values[1:]
        rhs_xy_list = rhs_df.iloc[row].values[1:]

        # Make sure all coordinates are paired
        if (len(lhs_xy_list) % 2 != 0 or len(rhs_xy_list) % 2 != 0):
            print("Not all coordinates are paired. Exiting...")
            exit(0)

        # Plot the points on the left hand side
        num_points_lhs = len(lhs_xy_list)/2
        for i in range(int(num_points_lhs)):
            plot(ax1, lhs_xy_list[2*i], lhs_xy_list[2*i+1], 'r*', str(i))
            if copy_lhs: 
                plot(ax2, lhs_xy_list[2*i], lhs_xy_list[2*i+1], 'r*', "")

        # Plot the points on the right hand side
        num_points_rhs = len(rhs_xy_list)/2
        for i in range(int(num_points_rhs)):
            plot(ax2, rhs_xy_list[2*i], rhs_xy_list[2*i+1], 'g*', str(i))

        if hide_axis:
            ax1.axis('off')
            ax2.axis('off')

        plt.show()

if __name__ == "__main__":
    CSV_WFLW_RAW_TRAIN  = 'raw_data/WFLW_landmarks_train.csv'
    CSV_WFLW_RAW_TEST   = 'raw_data/WFLW_landmarks_test.csv'
    CSV_300W_RAW_TRAIN  = 'raw_data/300W_landmarks_train.csv'
    CSV_300W_CONVERTED  = 'dataset/300W_landmarks_train.csv'
    CSV_WFLW_NORMALIZED = 'dataset/WFLW_normalized_landmarks_test.csv'


    #Display orignal WFLW and 300W points: Use this to explore the two formats
    #display_csv(CSV_WFLW_RAW_TRAIN, CSV_300W_RAW_TRAIN, 'Original WFLW vs. Original 300W Landmarks', True)

    #Display original 300W and converted points: Use this to verify landmarks are ordered correctly
    #display_csv(CSV_300W_RAW_TRAIN, CSV_300W_CONVERTED, 'Original 300W vs. Converted 300W Landmarks', False)

    #Display original WFLW and converted points: Use this to show how the converted points compare to the original
    #display_csv(CSV_WFLW_RAW_TRAIN, CSV_300W_CONVERTED, 'Original WFLW vs. Converted 300W Landmarks', True)

    #Display original and normalized WFLW points: Use this to verify values are normalized
    display_csv(CSV_WFLW_RAW_TEST, CSV_WFLW_NORMALIZED, 'Original WFLW vs. Normalized WFLW Landmarks', False, hide_axis=False)

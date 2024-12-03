import os
import pandas as pd
import seaborn as sns
import matplotlib.patches as mpatches
from matplotlib.dates import YearLocator
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

color_pal = sns.color_palette()

# output_dir_img = f"E:/Quadtree Ensemble Zero Pred Refined Model Increased MinMax Scaler/output_img"
output_dir_img = 'output_img'
if not os.path.exists(output_dir_img):
    os.makedirs(output_dir_img)

# timeseries_dir = f"E:/Quadtree Ensemble Zero Pred Refined Model Increased MinMax Scaler/output_timeseries"
timeseries_dir = 'output_timeseries'
if not os.path.exists(timeseries_dir):
    os.makedirs(timeseries_dir)


class Visualise:
    def __init__(self) -> None:
        pass

    # Visualise the quadtree.
    @staticmethod
    def visualize_quadtree(quadtree):
        plt.figure(figsize=(5, 5))
        # Set Seaborn style
        sns.set_theme(style="white")

        fig, ax = plt.subplots()

        # Recursively plot each rectangle in the quadtree
        def plot_node(node):
            if node is None:
                return
            ax.add_patch(plt.Rectangle((node.boundary.x1, node.boundary.y1),
                                       node.boundary.x2 - node.boundary.x1,
                                       node.boundary.y2 - node.boundary.y1,
                                       fill=False, edgecolor='black'))

            # Color data points and leaf nodes based on density
            if node.points:
                # Calculate density percentage
                num_points = len(node.points)
                density = num_points / quadtree.max_points * 100  # Calculating density compare to maximum point
                # capacity set for each leaf node.

                # Assign colors based on density thresholds
                if density > 60:
                    point_color = '#FF3333'  # PELATI
                    danger_level = 'High Crime'
                elif density > 40:
                    point_color = '#FF9933'  # INDIAN SAFFRON
                    danger_level = 'Medium Crime'
                elif density > 20:
                    point_color = '#FF3399'  # WILD STRAWBERRY
                    danger_level = 'Moderate Crime'
                elif density > 5:
                    point_color = '#3399FF'  # BRILLIANT AZURE
                    danger_level = 'Low Crime'
                else:
                    point_color = '#299954'  # MILITANT VEGAN
                    danger_level = 'Safe'

                # Plot data points and leaf nodes with assigned colors
                x = [point.x for point in node.points]
                y = [point.y for point in node.points]
                ax.scatter(x, y, color=point_color, s=5)

                # Add a legend patch for each danger level
                if danger_level not in legend_patches_dict:
                    legend_patch = mpatches.Patch(color=point_color, label=danger_level)
                    legend_patches_dict[danger_level] = legend_patch

            for child in node.children:
                plot_node(child)

        # Initialize legend patches dictionary
        legend_patches_dict = {}

        # Start plotting from the root node
        plot_node(quadtree)

        # Set plot limits and labels
        ax.set_xlim(quadtree.boundary.x1, quadtree.boundary.x2)
        ax.set_ylim(quadtree.boundary.y1, quadtree.boundary.y2)
        ax.set_aspect('equal', 'box')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        # ax.set_title('Crime Regions Using Quadtree Visualization')

        # Save the plot
        # plt.savefig(f"{output_dir_img}/quadtree.pdf", bbox_inches='tight')
        # output_dir_img = f"E:\quadtree_single_model_output\output_img"  # Delete this temporary path
        plt.savefig(f"{output_dir_img}/quadtree.png", bbox_inches='tight', dpi=500)
        plt.close()

    # Plot Single Plot Time Series.
    @staticmethod
    def time_series_plot_all_dcrs(df, dcr):
        df = df.set_index('CMPLNT_FR_DT')
        dcr = dcr.set_index('CMPLNT_FR_DT')
        # dcr_sorted = dcr.sort_values(by='CMPLNT_FR_DT')

        # Convert index to datetime
        df.index = pd.to_datetime(df.index)
        dcr.index = pd.to_datetime(dcr.index)
        # dcr_sorted.index = pd.to_datetime(dcr_sorted.index)

        # Plot Crime_count
        plt.figure(figsize=(10, 3))  # plt.figure(figsize=(15, 5.5))
        # plt.figure(facecolor='white')  # Set the background color to white
        plt.plot(df.index, df['Crime_count'], label='Actual Crime_count for all DCRs')
        # # Plot Unseen_pred
        # plt.plot(dcr_sorted.index, dcr_sorted['unseen_pred'], label='Predicted Crime_count for all DCRs')

        # Plot Unseen_pred
        plt.plot(dcr.index, dcr['Prediction'],
                 label='Predicted Crime_count for all DCRs')
        # plt.plot(dcr_sorted.index, dcr_sorted['unseen_pred'],
        #          label='Predicted Crime_count for all DCRs')

        # Set x-axis tick locator to show only the year
        plt.gca().xaxis.set_major_locator(YearLocator())

        # Set plot title and labels
        # plt.title('Crime Count Over Time for All DCRs.')
        plt.xlabel('Crimes from 2008-2017')
        plt.ylabel('No. of Crimes/Day')
        plt.xticks(rotation=0)  # Rotate x-axis labels for better readability
        plt.grid(True)
        plt.legend()

        # Save the plot
        # timeseries_dir = f"E:\quadtree_single_model_output\output_timeseries"  # Delete this temporary path
        plt.savefig(f"{timeseries_dir}/time_series_for_all_dcrs.pdf", bbox_inches='tight', dpi=500)
        plt.close()
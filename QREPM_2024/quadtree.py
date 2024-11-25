import pandas as pd
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler
from evaluation import Evaluation
from preprocess import Preprocess
from joblib import dump
from joblib import load
import time
import glob
import csv
import os

# from quadtree import Point, Rectangle, Quadtree

# Created an object of classes.
prp = Preprocess()

# Initialize the scaler
min_max_scaler = MinMaxScaler(feature_range=(100, 105))  # Set 100, 105

# Directory for saving CSV files
# dcr_dir_csv = f"E:/Quadtree Ensemble Zero Pred Refined Model Increased MinMax Scaler/unseen_node_pred_dir_csv"
dcr_dir_csv = 'unseen_node_pred_dir_csv'
if not os.path.exists(dcr_dir_csv):
    os.makedirs(dcr_dir_csv)
# output_dir_csv = f"E:/Quadtree Ensemble Zero Pred Refined Model Increased MinMax Scaler/unseen_output_csv"
# Directory for saving CSV files
output_dir_csv = 'unseen_output_csv'
if not os.path.exists(output_dir_csv):
    os.makedirs(output_dir_csv)

# model_saved = f"E:/Quadtree Ensemble Zero Pred Refined Model Increased MinMax Scaler/model_saved"
model_saved = 'model_saved'
if not os.path.exists(model_saved):
    os.makedirs(model_saved)


class InitialQuadtree:
    def __init__(self) -> None:
        self.evaluation_results = []  # Initialize an empty list to store evaluation results

    # Modeling to generated prediction values.
    @staticmethod
    def set_pred_zero(df):
        df['CMPLNT_FR_DT'] = Quadtree.datetime_to_unix_timestamps(df)
        df['Crime_count'] = round(Quadtree.min_max_scale_values(df, col_name='Crime_count'))
        df['Prediction'] = 0
        return df

    @staticmethod
    def init_quadtree(df):
        # Calculate a suitable minimum value for max_points based on the number of records
        min_points = len(df) // 30
        max_levels = min_points // 5
        if min_points < 1:
            min_points = 1
        if max_levels < 1:
            max_levels = 5

        # Prompt the user for the maximum number of points per node
        while True:
            try:
                max_points = int(
                    input(f"Enter the maximum number of points per node (minimum recommended: {min_points}): "))
                if max_points < min_points:
                    print(f"Please enter a value greater than or equal to {min_points}.")
                    continue
                break
            except ValueError:
                print("Please enter a positive integer value for the maximum number of points per node.")

        # Prompt the user for the maximum number of levels in the quadtree
        while True:
            try:
                max_levels = int(
                    input(f"Enter the maximum number of levels in the quadtree (minimum recommended: {max_levels}): "))
                if max_levels <= 1:
                    raise ValueError
                break
            except ValueError:
                print(
                    "Please enter a positive integer or more than 0 value for the maximum number of levels in the "
                    "quadtree.")

        # Creates a boundary rectangle for the quadtree based on the minimum and maximum longitude and latitude
        # values in the dataset.
        boundary_rectangle = Rectangle(min(df['Longitude']), min(df['Latitude']),
                                       max(df['Longitude']), max(df['Latitude']))

        # Initializes a Quadtree object with the boundary rectangle, max_points, and max_levels.
        quadtree = Quadtree(boundary_rectangle, max_points, max_levels)

        # Iterates over the dataset and extracts relevant data points such as longitude, latitude, index, and other
        # features. Extract data points from Longitude and Latitude columns and insert them into the quadtree
        for label, row in df.iterrows():
            longitude = row['Longitude']
            latitude = row['Latitude']
            index = row['index']
            CMPLNT_FR_DT = row['CMPLNT_FR_DT']
            Scl_Longitude = row['Scl_Longitude']
            Scl_Latitude = row['Scl_Latitude']
            Dayofweek_of_crime = row['Dayofweek_of_crime']
            Quarter_of_crime = row['Quarter_of_crime']
            Month_of_crime = row['Month_of_crime']
            Dayofyear_of_crime = row['Dayofyear_of_crime']
            Dayofmonth_of_crime = row['Dayofmonth_of_crime']
            Weekofyear_of_crime = row['Weekofyear_of_crime']
            Year_of_crime = row['Year_of_crime']
            Distance_From_Central_Point = row['Distance_From_Central_Point']
            Longitude_Latitude_Ratio = row['Longitude_Latitude_Ratio']
            Location_density = row['Location_density']
            Crime_count = row['Crime_count']
            Prediction = row['Prediction']

            # Creates a Point object for each data point with the extracted features and inserts it into the quadtree.
            point = Point(longitude, latitude, index, CMPLNT_FR_DT, Scl_Longitude, Scl_Latitude, Dayofweek_of_crime,
                          Quarter_of_crime, Month_of_crime, Dayofyear_of_crime, Dayofmonth_of_crime,
                          Weekofyear_of_crime, Year_of_crime, Distance_From_Central_Point, Longitude_Latitude_Ratio,
                          Location_density, Crime_count, Prediction)
            quadtree.insert(point)

            # Returns the initialized quadtree.
        return quadtree

# Represents a point with various attributes such as coordinates (x and y) and additional information related to
# crime data (CMPLNT_FR_DT, Scl_Longitude, etc.).
class Point:
    def __init__(self, x, y, index, CMPLNT_FR_DT, Scl_Longitude, Scl_Latitude, Dayofweek_of_crime,
                 Quarter_of_crime, Month_of_crime, Dayofyear_of_crime, Dayofmonth_of_crime, Weekofyear_of_crime,
                 Year_of_crime, Distance_From_Central_Point, Longitude_Latitude_Ratio, Location_density,
                 Crime_count, Prediction):
        self.x = x  # Longitude
        self.y = y  # Latitude
        self.index = index
        self.CMPLNT_FR_DT = CMPLNT_FR_DT
        self.Scl_Longitude = Scl_Longitude
        self.Scl_Latitude = Scl_Latitude
        self.Dayofweek_of_crime = Dayofweek_of_crime
        self.Quarter_of_crime = Quarter_of_crime
        self.Month_of_crime = Month_of_crime
        self.Dayofyear_of_crime = Dayofyear_of_crime
        self.Dayofmonth_of_crime = Dayofmonth_of_crime
        self.Weekofyear_of_crime = Weekofyear_of_crime
        self.Year_of_crime = Year_of_crime
        self.Distance_From_Central_Point = Distance_From_Central_Point
        self.Longitude_Latitude_Ratio = Longitude_Latitude_Ratio
        self.Location_density = Location_density
        self.Crime_count = Crime_count
        self.Prediction = Prediction


""" 
Represents a rectangle with bottom-left and top-right corner coordinates. It provides methods to check if a point
lies within the rectangle (contains_point) and if it intersects with another rectangle (intersects).
"""


class Rectangle:
    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    # Check if a point lies within the rectangle. Returns True if the point lies within the rectangle, False otherwise.
    def contains_point(self, x, y):
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2

    # Check if the rectangle intersects with another rectangle.
    def intersects(self, other):
        return not (self.x2 < other.x1 or self.x1 > other.x2 or self.y2 < other.y1 or self.y1 > other.y2)


"""
Represents the quadtree data structure. It is initialized with a boundary rectangle, maximum number of points per 
node (max_points), and maximum depth (max_levels). The quadtree is recursively divided into four quadrants until each 
quadrant contains no more than max_points or reaches the maximum depth. It provides methods to insert points into the 
quadtree (insert), subdivide a node into quadrants (subdivide), and check if a node is a leaf node (is_leaf).
"""


class Quadtree:
    def __init__(self, boundary, max_points=None, max_levels=None, node_id=0,
                 root_node=None,
                 node_level=0,
                 # tree_level=None,
                 parent=None,
                 df=None,
                 ex_time=None):

        self.model = None  # To hold the current model while traversal through quadtree.
        self.boundary = boundary  # Stores the boundary rectangle of the quadtree.
        self.max_points = max_points if max_points is not None else 4
        self.max_levels = max_levels if max_levels is not None else 10
        self.temp_points = []  # Stores the points that belong to the leaf nodes.
        self.children = []  # Stores the child nodes of the quadtree.
        self.node_level = node_level  # Stores the level of the current node within the quadtree.
        self.node_id = node_id  # Assign a unique identifier to the root node
        self.points = []  # Stores the points that belong to the current node.
        self.parent = parent  # To hold the pointer of the parent node.
        self.df = df  # To store the current dataset while traversal though each node of quadtree.
        self.ex_time = ex_time  # To store execution time of each node.
        self.evaluation_results = []  # Initialize an empty list to store evaluation results

        # Node IDs assignment.
        if root_node is None:
            self.root_node = self  # Assigning root in itself.
            self.global_count = 0  # Set global counter to keep sequence and track on node ids.
        else:
            self.root_node = root_node  # Setting current node id.

        # Tree Level assignment.
        # if tree_level is None:
        #     self.tree_level = self
        #     self.global_level_count = 0
        #
        # else:
        #     self.tree_level = tree_level

        # Ensure that boundary is a Rectangle object
        if not isinstance(self.boundary, Rectangle):
            raise ValueError("Boundary must be a Rectangle object")

    def insert(self, point, node_id=0):  # Added node_id argument for recursive calls
        # Check Boundary: Check if the point is within the boundary of the current node
        self.points.append(point)  # Appending entered data points to the parent nodes.
        if not self.boundary.contains_point(point.x, point.y):
            return False

        # Check Leaf Node: Check if the current node is a leaf node and there is space to insert the point
        if self.is_leaf() and len(self.temp_points) < self.max_points:
            self.temp_points.append(point)
            return True

        # Subdivide Node: If the current node is not a leaf node, or it's full, subdivide it
        if not self.children:
            self.subdivide()

        # Insert into Child Nodes: Attempt to insert the point into the child nodes
        for child in self.children:
            if child.boundary.contains_point(point.x, point.y):
                child.insert(point, child.node_id)  # Pass current node ID to child

    def subdivide(self):
        # Calculate the dimensions of each child node
        x_mid = (self.boundary.x1 + self.boundary.x2) / 2
        y_mid = (self.boundary.y1 + self.boundary.y2) / 2

        # tree_level_count = self.tree_level.global_level_count + 1
        # self.tree_level.global_level_count = tree_level_count

        # Create child nodes representing each quadrant
        node_id_counter = self.root_node.global_count + 1  # Increasing global count for each node.
        self.root_node.global_count = node_id_counter  # Assigned in local variable.
        nw_boundary = Rectangle(self.boundary.x1, y_mid, x_mid, self.boundary.y2)
        nw_quadtree = Quadtree(nw_boundary, self.max_points, self.max_levels, root_node=self.root_node,
                               parent=self)  # tree_level=self.tree_level,
        nw_quadtree.node_id = node_id_counter  # Assigned id to the current node.
        # nw_quadtree.node_level = tree_level_count
        nw_quadtree.node_level = self.node_level + 1
        self.children.append(nw_quadtree)

        node_id_counter = self.root_node.global_count + 1
        self.root_node.global_count = node_id_counter
        ne_boundary = Rectangle(x_mid, y_mid, self.boundary.x2, self.boundary.y2)
        ne_quadtree = Quadtree(ne_boundary, self.max_points, self.max_levels, root_node=self.root_node,
                               parent=self)  # tree_level=self.tree_level,
        ne_quadtree.node_id = node_id_counter
        ne_quadtree.node_level = self.node_level + 1
        self.children.append(ne_quadtree)

        node_id_counter = self.root_node.global_count + 1
        self.root_node.global_count = node_id_counter
        sw_boundary = Rectangle(self.boundary.x1, self.boundary.y1, x_mid, y_mid)
        sw_quadtree = Quadtree(sw_boundary, self.max_points, self.max_levels, root_node=self.root_node,
                               parent=self)  # tree_level=self.tree_level,
        sw_quadtree.node_id = node_id_counter
        sw_quadtree.node_level = self.node_level + 1
        self.children.append(sw_quadtree)

        node_id_counter = self.root_node.global_count + 1
        self.root_node.global_count = node_id_counter
        se_boundary = Rectangle(x_mid, self.boundary.y1, self.boundary.x2, y_mid)
        se_quadtree = Quadtree(se_boundary, self.max_points, self.max_levels, root_node=self.root_node,
                               parent=self)  # tree_level=self.tree_level,
        se_quadtree.node_id = node_id_counter
        se_quadtree.node_level = self.node_level + 1
        self.children.append(se_quadtree)

        for point in self.temp_points:
            for child in self.children:
                if child.boundary.contains_point(point.x, point.y):
                    child.insert(point)
                    break

        self.temp_points = []

    # Check if the current node is a leaf node (i.e., it has no children).
    def is_leaf(self):
        return len(self.children) == 0

    # Recursive method to traversal through quadtree using Depth-First Search Algorithm.
    def traverse_quadtree_modelling(self):
        if self.node_id == 0:
            print(f"Visiting node ID: {self.node_id}")
        self.modelling()
        for child in self.children:
            print(f"Visiting node ID: {child.node_id}")
            print()
            child.traverse_quadtree_modelling()
            print(f"Finished visiting node ID: {child.node_id}")
            print()

    # Train the models recursively. Algorithm performs Depth-First Search traversal process.
    def modelling(self):
        dataframe_id = f"DCR_{self.node_id}"
        points_lst = []

        if not len(self.points) == 0:  # If the current node is not empty, then continue the process otherwise return.

            # Extract points and its attributes and store in df
            for point in self.points:  # Extracting each point from points.
                point_dict = self.extract_data_points(point, dataframe_id)  # Collect attributes belonging to point.
                points_lst.append(point_dict)  # Append each data point and its belonging attributes to list.
            self.df = pd.DataFrame(points_lst)

            """
            Fetch prediction values from parent node. Here we are collecting the prediction values from parent node and 
            assigning to child node.
            """
            if self.parent is not None:
                # print(f"LENGTH OF PARENT PREDICTED VALUES: {len(self.parent.df)}")
                # print(f"y-Pred BEFORE FETCHED FROM PARENT NODE: \n {self.df}")
                self.df = self.update_prediction(self.df, self.parent.df)
                # print(f"y-Pred AFTER FETCHED FROM PARENT NODE: \n {self.df}")

            """
            Model Training and Prediction.
            """
            # Extract features and target variable from the data
            features, target = self.parent_and_leaf_features_target()
            X_train = self.df[features]
            y_train = self.df[target]

            if self.node_id == 0:  # Check if current node is root node.
                # Record the start time
                start_time = time.time()
                self.train_model(X_train, y_train)  # Calling training method.
                # Record the end time
                end_time = time.time()
                # Calculate Execution Time.
                self.ex_time = prp.calculate_execution_time(start_time, end_time, self.df)
            else:  # If current node is not root node. It means these are parent or leaf node.
                parent_model_path = f"{model_saved}/model_{self.parent.node_id}.joblib"
                if os.path.exists(parent_model_path):
                    parent_model = load(parent_model_path)
                    # Record the start time
                    start_time = time.time()
                    self.train_model(X_train, y_train, parent_model)
                    # Record the end time
                    end_time = time.time()
                    # Calculate Execution Time.
                    self.ex_time = prp.calculate_execution_time(start_time, end_time, self.df)
                else:
                    raise FileNotFoundError(f"Parent model file {parent_model_path} does not exist.")

            y_pred = self.model.predict(X_train)
            self.df['Prediction'] = y_pred.round()

    # Recursive method to Predict on Unseen Data. Travers through Depth-First Search.
    def traverse_quadtree_evaluation(self, unseen_data, df_len, p_predictors):
        if self.node_id == 0:
            print(f"Visiting Unseen_data node ID: {self.node_id}")
        self.evaluate_unseen_data(unseen_data, df_len, p_predictors)
        for child in self.children:
            print(f"Visiting Unseen_data node ID: {child.node_id}")
            print()
            child.traverse_quadtree_evaluation(unseen_data, df_len, p_predictors)
            print(f"Finished visiting Unseen_data node ID: {child.node_id}")
            print()

    # Prediction on Unseen Data.
    def evaluate_unseen_data(self, unseen_data, df_len, p_predictors):
        global y_test, X_test
        dataframe_id = f"DCR_{self.node_id}"
        self.df = unseen_data

        if self.model:
            if not len(self.df) == 0:  # If the unseen_df is not empty, then continue the process otherwise return.
                """
                Fetch prediction values from parent node.
                """
                # print(f"LENGTH OF CURRENT NODE: {len(df)}")
                if self.parent is not None:
                    # print(f"LENGTH OF PARENT PREDICTED VALUES: {len(self.parent.df)}")
                    # print(f"y-Pred BEFORE FETCHED FROM PARENT NODE: \n {self.df}")
                    self.df = self.update_prediction(self.df, self.parent.df)
                    # print(f"y-Pred AFTER FETCHED FROM PARENT NODE: \n {self.df}")

                """
                Model Training and Prediction.
                """
                # Extract features and target variable from the data
                features, target = self.parent_and_leaf_features_target()
                X_test = self.df[features]
                y_test = self.df[target]

                # Each node contains the trained model attribute which we are using to make predictions.
                y_pred = self.model.predict(X_test)
                self.df['Prediction'] = y_pred.round()

                # Calculate error matrices for each node.
                self.cal_error_matrices_each_node(y_test, y_pred, dataframe_id, df_len, p_predictors)

            else:
                raise Exception(f"{self.df} is empty!")

    # Calculate Error Matrices for each node
    def cal_error_matrices_each_node(self, y_test, y_pred, dataframe_id, df_len, p_predictors):
        n_len = df_len
        p_predictors = p_predictors
        unseen_eval = Evaluation(y_test, y_pred, n_len, p_predictors)
        mae = round(unseen_eval.mean_absolute_error(), 2)
        rmse = round(unseen_eval.root_mean_squared_error(), 2)
        mape = round(unseen_eval.mean_absolute_percentage_error(), 2)
        me = round(unseen_eval.mean_error(), 2)
        smape = round(unseen_eval.symmetric_mean_absolute_percentage_error())
        r_2 = round(unseen_eval.r_squared(), 2)
        adj_r_2 = round(unseen_eval.adjusted_r_squared(), 2)

        # Check is current node is Parent or Leaf node.
        if self.node_id == 0:
            node_is = "Root_Node"
        elif self.children:
            node_is = "Parent_Node"
        else:
            node_is = "Leaf_Node"

        # Append a result in self-object attribute to access later.
        node_length = len(self.points)
        self.evaluation_results.append(
            {'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'ME': me, 'R_2': r_2, 'Adj_R_2': adj_r_2, 'SMAPE': smape, 'DCR_ID': dataframe_id,
             'Ex_Time': self.ex_time, 'Node_Length': node_length, 'Node_ID': self.node_id,
             'Node_Level': self.node_level, 'Node_Is': node_is, 'Cmax': self.max_points, 'Lmax': self.max_levels})

        # Path to store the CSV file
        output_path = f"{output_dir_csv}/unseen_each_node_eval_matrices_{df_len}.csv"

        # Check if the output file already exists
        file_exists = os.path.isfile(output_path)

        # Perform experiments for each max_points value
        with open(output_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)

            # If the file doesn't exist, write the header row
            if not file_exists:
                writer.writerow(
                    ['MAE', 'RMSE', 'MAPE', 'ME','R_2', 'Adj_R_2', 'SMAPE', 'DCR_ID', 'Ex_Time', 'Node_Length', 'Node_ID', 'Node_Level',
                     'Node_Is', 'Cmax', 'Lmax'])

            # Write the data for this experiment
            writer.writerow(
                [mae, rmse, mape, me, r_2, adj_r_2, smape, dataframe_id, self.ex_time, node_length, self.node_id, self.node_level,
                 node_is, self.max_points, self.max_levels])

        # Save CSV file.
        self.df.to_csv(f"{dcr_dir_csv}/{dataframe_id}_unseen.csv", index=False)

    # Calculate the Average Prediction of Each Row.
    def rows_average_prediction(self, df_len, p_predictors):
        csv_files_path = f"{dcr_dir_csv}/*.csv"
        csv_files = glob.glob(csv_files_path)
        n_len = df_len  # Number of observation.
        p_predictors = p_predictors  # Number of predictors in the model

        # Initialize a list to hold DataFrames
        dfs = []
        bool_var = True
        for file in csv_files:
            if bool_var is True:
                df = pd.read_csv(file)
                dfs.append(df)
                bool_var = False
            else:
                df = pd.read_csv(file)
                dfs.append(df['Prediction'])

        combined_df = pd.concat(dfs, axis=1)

        # Extract columns that contain 'Prediction' in their name
        prediction_columns = [col for col in combined_df.columns if 'Prediction' in col]

        # Calculate the average of the Prediction columns
        combined_df['Average_Prediction'] = combined_df[prediction_columns].mean(axis=1)

        # Drop the Prediction columns from the DataFrame
        combined_df = combined_df.drop(columns=prediction_columns)

        unseen_eval = Evaluation(combined_df['Crime_count'], combined_df['Average_Prediction'], n_len, p_predictors)

        mae = round(unseen_eval.mean_absolute_error(), 2)
        rmse = round(unseen_eval.root_mean_squared_error(), 2)
        mape = round(unseen_eval.mean_absolute_percentage_error(), 2)
        me = round(unseen_eval.mean_error(), 2)
        smape = round(unseen_eval.symmetric_mean_absolute_percentage_error())
        r_2 = round(unseen_eval.r_squared(), 2)
        adj_r_2 = round(unseen_eval.adjusted_r_squared(), 2)

        # Path to store the CSV file
        output_path = f"{output_dir_csv}/single_avg_eval.csv"

        # Check if the output file already exists
        file_exists = os.path.isfile(output_path)

        # Perform experiments for each max_points value
        with open(output_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)

            # If the file doesn't exist, write the header row
            if not file_exists:
                writer.writerow(['MAE', 'RMSE', 'MAPE', 'ME', 'R_2', 'Adj_R_2', 'SMAPE', 'DATA_SIZE', 'Cmax', 'Lmax'])

            # Write the data for this experiment
            writer.writerow([mae, rmse, mape, me, r_2, adj_r_2, smape, df_len, self.max_points, self.max_levels])

        # Inversed Unix timestamps to data time and scaled values into original values.
        combined_df['CMPLNT_FR_DT'] = self.unix_timestamps_to_datetime(combined_df)
        combined_df['Crime_count'] = self.inverse_min_max_scale_values(combined_df, col_name='Crime_count')
        combined_df['Average_Prediction'] = self.inverse_min_max_scale_values(combined_df,
                                                                              col_name='Average_Prediction')

        # Optionally, save the combined DataFrame to a CSV file
        combined_df.to_csv(f"{output_dir_csv}/row_wise_avg_prediction.csv", index=False)

    # Extract current node's data points with other attributes.
    def extract_data_points(self, point, dataframe_id):
        point_dict = {
            "index": point.index,
            "CMPLNT_FR_DT": point.CMPLNT_FR_DT,
            "Scl_Longitude": point.Scl_Longitude,
            "Scl_Latitude": point.Scl_Latitude,
            "Dayofweek_of_crime": point.Dayofweek_of_crime,
            "Quarter_of_crime": point.Quarter_of_crime,
            "Month_of_crime": point.Month_of_crime,
            "Dayofyear_of_crime": point.Dayofyear_of_crime,
            "Dayofmonth_of_crime": point.Dayofmonth_of_crime,
            "Weekofyear_of_crime": point.Weekofyear_of_crime,
            "Year_of_crime": point.Year_of_crime,
            "Distance_From_Central_Point": point.Distance_From_Central_Point,
            "Longitude_Latitude_Ratio": point.Longitude_Latitude_Ratio,
            "Location_density": point.Location_density,
            "node_id": self.node_id,
            "node_level": self.node_level,
            "dataframe_id": dataframe_id,
            "Crime_count": point.Crime_count,
            "Prediction": point.Prediction
        }
        # sample
        return point_dict

    # Convert a datetime format to Unix timestamp.
    @staticmethod
    def datetime_to_unix_timestamps(df):
        df['CMPLNT_FR_DT'] = df['CMPLNT_FR_DT'].astype('int64') // 10 ** 9
        return df['CMPLNT_FR_DT']

    # Convert a Unix timestamp format to datetime.
    @staticmethod
    def unix_timestamps_to_datetime(data):
        data['CMPLNT_FR_DT'] = pd.to_datetime(data['CMPLNT_FR_DT'], unit='s')
        return data['CMPLNT_FR_DT']

    # Scale down the target and predicted values
    @staticmethod
    def min_max_scale_values(df, col_name):
        # Reshape the Crime_count column to a 2D array
        col_counts = df[col_name].values.reshape(-1, 1)

        # Fit and transform the scaled values
        df[col_name] = min_max_scaler.fit_transform(col_counts)

        return df[col_name]

    # Scale up the target and predicted values
    @staticmethod
    def inverse_min_max_scale_values(df, col_name):
        col_counts = df[col_name].values.reshape(-1, 1)  # Reshape the Crime_count column to a 2D array
        df[col_name] = min_max_scaler.inverse_transform(col_counts).round()  # Fit and transform the scaled values
        return df[col_name]

    # Features and target for root node.
    @staticmethod
    def root_features_target():
        # Define features and target variable
        features = ['CMPLNT_FR_DT', 'Scl_Longitude', 'Scl_Latitude', 'Dayofweek_of_crime',
                    'Quarter_of_crime', 'Month_of_crime', 'Dayofyear_of_crime', 'Dayofmonth_of_crime',
                    'Weekofyear_of_crime', 'Year_of_crime', 'Distance_From_Central_Point', 'Longitude_Latitude_Ratio',
                    'Location_density']  # 'CMPLNT_DATETIME', node_id, node_level, 'Hour_of_crime',
        target = 'Crime_count'
        return features, target

    # Features and target for parent and leaf node.
    @staticmethod
    def parent_and_leaf_features_target():
        # Define features and target variable
        features = ['CMPLNT_FR_DT', 'Scl_Longitude', 'Scl_Latitude', 'Dayofweek_of_crime',
                    'Quarter_of_crime', 'Month_of_crime', 'Dayofyear_of_crime', 'Dayofmonth_of_crime',
                    'Weekofyear_of_crime', 'Year_of_crime', 'Distance_From_Central_Point', 'Longitude_Latitude_Ratio',
                    'Prediction', 'Location_density']  # 'CMPLNT_DATETIME', node_id, node_level, 'Hour_of_crime',
        target = 'Crime_count'
        return features, target

    def train_model(self, X_train, y_train, parent_model=None):
        model_path = f"{model_saved}/model_{self.node_id}.joblib"

        self.model = XGBRegressor(
            base_score=0.5, booster='gbtree', n_estimators=1000,
            early_stopping_rounds=50, objective='reg:squarederror',
            max_depth=3, learning_rate=0.01, random_state=100, n_jobs=4
        )

        if parent_model:
            # Continue training from the parent model
            self.model.fit(X_train, y_train, eval_set=[(X_train, y_train)], verbose=100,
                           xgb_model=parent_model.get_booster())
        else:
            # Train the model from scratch
            self.model.fit(X_train, y_train, eval_set=[(X_train, y_train)], verbose=100)

        # Save the current model
        dump(self.model, model_path)

    @staticmethod
    def update_prediction(df, parent_df):
        # Create a dictionary to store the mapping of Index to prediction values in parent_df
        prediction_mapping = dict(zip(parent_df['index'], parent_df['Prediction']))

        # Iterate over the rows of df
        for index, row in df.iterrows():
            # Get the Index value of the current row
            index_value = row['index']

            # Check if the Index value exists in the prediction_mapping
            if index_value in prediction_mapping:
                # Replace the value of prediction in df with the corresponding value from parent_df
                df.at[index, 'Prediction'] = prediction_mapping[index_value]

        return df

    # Method to calculate Error Matrices and display in command prompt.
    def average_calculation(self):
        # Read the CSV file into a DataFrame
        evaluation_df = pd.DataFrame(self.evaluation_results)

        # Calculate the average of MAE, RMSE, MAPE, and ME
        average_mae = round(evaluation_df['MAE'].mean(), 2)
        average_rmse = round(evaluation_df['RMSE'].mean(), 2)
        average_mape = round(evaluation_df['MAPE'].mean(), 2)
        average_me = round(evaluation_df['ME'].mean(), 2)
        average_smape = round(evaluation_df['SMAPE'].mean(), 2)

        print("Average MAE:", average_mae)
        print("Average RMSE:", average_rmse)
        print("Average MAPE:", average_mape)
        print("Average ME:", average_me)
        print("Average SMAPE:", average_smape)

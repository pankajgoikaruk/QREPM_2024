import warnings
import pandas as pd
import os
import csv
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# output_dir_csv = f"E:/Quadtree Ensemble Zero Pred Refined Model Increased MinMax Scaler/unseen_output_csv"
output_dir_csv = 'unseen_output_csv'
if not os.path.exists(output_dir_csv):
    os.makedirs(output_dir_csv)

class Preprocess:
    def __init__(self):
        pass

    # Import data.
    @staticmethod
    def data_import(path):
        """
            Load data from a CSV file.

            Args:
                path (str): Path to the CSV file.

            Returns:
                pd.DataFrame: Loaded data as a DataFrame.
            """
        try:
            data = pd.read_csv(path, encoding='latin-1')
            return data
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {path}")
        except Exception as e:
            raise Exception(f"Error reading file: {e}")

    # Check any null value
    @staticmethod
    def null_values_check(df):
        df = df.dropna()
        return df

    # Combine date and time into a single datetime column and convert date time into DateTime format.
    @staticmethod
    def datetime_convert(df):
        try:
            df['CMPLNT_FR_DT'] = pd.to_datetime(df['CMPLNT_FR_DT'])
            return df
        except Exception as e:
            raise ValueError(f"Datetime conversion failed: {e}")

    # Get sample data.
    @staticmethod
    def get_sample_data(df, start_date=None, end_date=None):
        # Filtering the DataFrame for the chosen date range
        sample_df = pd.DataFrame(df[(df['CMPLNT_FR_DT'] >= start_date) & (df['CMPLNT_FR_DT'] <= end_date)])
        return sample_df

    # Perform time series analysis
    @staticmethod
    def crime_total_count(df):
        # time_series_data = df.groupby('CMPLNT_FR_DT').size().reset_index(name='Crime_count')
        df['Crime_count'] = df.groupby('CMPLNT_FR_DT')['CMPLNT_FR_DT'].transform('count')
        return df

# Adding some new features to Dataframe.
    @staticmethod
    def create_new_features(df):
        """
        Create time series features based on time series index.
        """
        new_df = df.copy()

        # Reset Index to add index column
        new_df.reset_index(inplace=True)

        # Scaling Longitude and Latitude
        scaler = StandardScaler()
        new_df[['Scl_Longitude', 'Scl_Latitude']] = scaler.fit_transform(new_df[['Longitude', 'Latitude']])

        # Calculate the ratio of Longitude and Latitude
        new_df['Longitude_Latitude_Ratio'] = new_df['Longitude'] / new_df['Latitude']

        # Calculate the frequency of occurrence for each longitude and latitude pair
        Location_density = new_df.groupby(['Longitude', 'Latitude']).size().reset_index(name='Location_Density') # IMP
        new_df = pd.merge(new_df, Location_density, on=['Longitude', 'Latitude'], how='left')
        new_df['Location_density'] = new_df['Location_Density']
        # Drop the redundant Location_Density column
        new_df.drop(columns=['Location_Density'], inplace=True)

        # Example: Calculate the distance from a central point
        central_longitude = new_df['Longitude'].mean()
        central_latitude = new_df['Latitude'].mean()
        new_df['Distance_From_Central_Point'] = ((new_df['Longitude'] - central_longitude) ** 2 + (
                new_df['Latitude'] - central_latitude) ** 2) ** 0.5
        new_df['Dayofweek_of_crime'] = new_df[
            'CMPLNT_FR_DT'].dt.dayofweek  # Day of the Week of Crime. Example: Monday, Tuesday....
        new_df['Quarter_of_crime'] = new_df['CMPLNT_FR_DT'].dt.quarter  # Quarter of the Crime.
        new_df['Month_of_crime'] = new_df['CMPLNT_FR_DT'].dt.month  # Month of the Crime.
        new_df['Year_of_crime'] = new_df['CMPLNT_FR_DT'].dt.year  # Year of the Crime.
        new_df['Dayofyear_of_crime'] = new_df['CMPLNT_FR_DT'].dt.dayofyear  # Day of the Year of the Crime.
        new_df['Dayofmonth_of_crime'] = new_df['CMPLNT_FR_DT'].dt.day  # Day of the Month of the Crime.
        new_df['Weekofyear_of_crime'] = new_df['CMPLNT_FR_DT'].dt.isocalendar().week  # Week of the Year of the Crime.

        # Rearrange columns
        new_df = new_df[
            ['index', 'CMPLNT_FR_DT', 'Longitude', 'Latitude', 'Scl_Longitude', 'Scl_Latitude', 'Dayofweek_of_crime',
             'Quarter_of_crime', 'Month_of_crime', 'Dayofyear_of_crime', 'Dayofmonth_of_crime', 'Weekofyear_of_crime',
             'Year_of_crime', 'Distance_From_Central_Point', 'Longitude_Latitude_Ratio', 'Location_density',
             'Crime_count']]
        return new_df

    # Data split for training and testing.
    @staticmethod
    def train_val_test_df_split(df, train_size):
        train_per = train_size

        split_index = int(len(df) * train_per)

        seen_df = df[:split_index]
        unseen_df = df[split_index:]

        return seen_df, unseen_df


    # Calculate Execution Time.
    @staticmethod
    def calculate_execution_time(start_time, end_time, df):
        # Calculate the elapsed time
        elapsed_time_sec = end_time - start_time
        elapsed_time_min = round(elapsed_time_sec / 60, 2)

        print(f"Execution time: {elapsed_time_min:.2f} minutes for data points {len(df)}")

        # Path to store the CSV file
        output_path = f"{output_dir_csv}/execution_time.csv"

        # Check if the output file already exists
        file_exists = os.path.isfile(output_path)

        # Perform experiments for each max_points value
        with open(output_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)

            # If the file doesn't exist, write the header row
            if not file_exists:
                writer.writerow(['Total Dataset', 'Execution Time (minutes)'])

            # Write the data for this experiment
            writer.writerow([len(df), elapsed_time_min])

        return elapsed_time_min
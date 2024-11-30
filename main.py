import pandas as pd
from preprocess import Preprocess
from quadtree import InitialQuadtree
import os
import time
import logging

# Directory for saving pre-processed data.
preprocess_dir_csv = 'preprocess_data_csv'
if not os.path.exists(preprocess_dir_csv):
    os.makedirs(preprocess_dir_csv)

# Directory for saving unseen data output.
output_dir_csv = 'unseen_output_csv'
if not os.path.exists(output_dir_csv):
    os.makedirs(output_dir_csv)

# Directory for saving prediction CSV files.
dcr_dir_csv = 'unseen_node_pred_dir_csv'
if not os.path.exists(dcr_dir_csv):
    os.makedirs(dcr_dir_csv)

''' ###################################### DATA PREPROCESSING ###################################### '''
# Optional: Record the start time
start_time = time.time()

# Created an object of classes.
prp = Preprocess()
quad = InitialQuadtree()
data_file_name = "USA_Crime_2008_to_2009.csv"

if not os.path.exists(f"{preprocess_dir_csv}/pre_pro_{data_file_name}"):
    # Step 1: Load crime data from csv file.
    data_path = f'C:/Users/goika/Quadtree/data/{data_file_name}'

    data = prp.data_import(data_path)

    # Step 2: Extract important columns.
    required_columns = ['CMPLNT_FR_DT', 'Longitude', 'Latitude']
    df = data[required_columns]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required columns: {col}")

    # Step 3:  Check null values.
    df = prp.null_values_check(df)

    # Step 4: Convert date time into DateTime format.
    df = prp.datetime_convert(df)

    # # Optional: Get sample data.
    # start_date = '2008-01-01'
    # end_date = '2008-12-31'
    # df = prp.get_sample_data(df, start_date, end_date)

    # Step 5: Crime Count and add new column
    df = prp.crime_total_count(df)

    # Step 6: Adding some new features to Dataframe and Scaling Longitude and Latitude.
    df = prp.create_new_features(df)

    # Step 7: Data Split and Added Prediction Column with Zero Value.
    seen_df, unseen_df = prp.train_val_test_df_split(df, train_size=0.8)
    seen_df = quad.set_pred_zero(seen_df)
    unseen_df = quad.set_pred_zero(unseen_df)

    # Save CSV file.
    version = "v1"
    seen_df.to_csv(f"{preprocess_dir_csv}/pre_pro_seen_{data_file_name}", index=False)
    unseen_df.to_csv(f"{preprocess_dir_csv}/pre_pro_unseen_{data_file_name}", index=False)
    print(f"Preprocessed data saved at: {preprocess_dir_csv}/pre_pro_{data_file_name}")
    logging.basicConfig(level=logging.INFO)
    logging.info("Data preprocessing completed successfully.")
else:
    seen_df = pd.read_csv(f"{preprocess_dir_csv}/pre_pro_seen{data_file_name}", parse_dates=['CMPLNT_FR_DT'])
    unseen_df = pd.read_csv(f"{preprocess_dir_csv}/pre_pro_unseen{data_file_name}", parse_dates=['CMPLNT_FR_DT'])

print(f"Seen Data: \n {seen_df} \n")
print(f"Unseen Data: \n {unseen_df}")

''' ################# CREATING QUADTREE AND DISTRIBUTING DATA POINTS INTO LIST OF DATA FRAMES ################# '''

# # Step 8: Create Quadtree.
# quadtree = quad.init_quadtree(seen_df)

# # Step 9: Train the Models
# quadtree.traverse_quadtree_modelling()
#
# # Step 10: Prediction on Unseen Data.
# df_len = len(seen_df)  # Number of observation.
# print(f"Length of seen_df is: {df_len} \n")
# p_predictors = 14  # Number of predictors in the model
# print(f"Columns of p_predictors are: {p_predictors} \n")
#
# quadtree.traverse_quadtree_evaluation(unseen_df, df_len, p_predictors)








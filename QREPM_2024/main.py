import pandas as pd
from preprocess import Preprocess
from quadtree import InitialQuadtree
import os
import time

# output_dir_csv = f"E:/Quadtree Ensemble Zero Pred Refined Model Increased MinMax Scaler/unseen_output_csv"
output_dir_csv = 'unseen_output_csv'
if not os.path.exists(output_dir_csv):
    os.makedirs(output_dir_csv)

# Directory for saving CSV files
# dcr_dir_csv = f"E:/Quadtree Ensemble Zero Pred Refined Model Increased MinMax Scaler/unseen_node_pred_dir_csv"
dcr_dir_csv = 'unseen_node_pred_dir_csv'
if not os.path.exists(dcr_dir_csv):
    os.makedirs(dcr_dir_csv)

''' ###################################### DATA PREPROCESSING ###################################### '''
# Optional: Record the start time
start_time = time.time()

# Created an object of classes.
prp = Preprocess()
quad = InitialQuadtree()

# Step 1: Load crime data from csv file.
data_path = 'C:/Users/goika/Quadtree/data/USA_Crime_2008_to_2009.csv'
data = prp.data_import(data_path)

# Step 2: Extract important columns.
df = data[['CMPLNT_FR_DT', 'Longitude', 'Latitude']]

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

print(f"Seen Data: \n {seen_df} \n")
print(f"Unseen Data: \n {unseen_df}")


''' ################# CREATING QUADTREE AND DISTRIBUTING DATA POINTS INTO LIST OF DATA FRAMES ################# '''

# Step 8: Create Quadtree.
quadtree = quad.init_quadtree(seen_df)

# Step 9: Train the Models
quadtree.traverse_quadtree_modelling()

# Step 10: Prediction on Unseen Data.
df_len = len(seen_df)  # Number of observation.
print(f"Length of seen_df is: {df_len} \n")
p_predictors = 14  # Number of predictors in the model
print(f"Columns of p_predictors are: {p_predictors} \n")

quadtree.traverse_quadtree_evaluation(unseen_df, df_len, p_predictors)








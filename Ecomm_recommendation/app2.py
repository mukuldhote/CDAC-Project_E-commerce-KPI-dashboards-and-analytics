import streamlit as st
import pandas as pd
import numpy as np
import surprise
import pickle

# Function to load the pre-built model
def load_model():
    with open('recommendation_model.pkl', 'rb') as model_file:
        algo = pickle.load(model_file)
    with open('id_mapping.pkl', 'rb') as id_mapping_file:
        id_mapping = pickle.load(id_mapping_file)
    return algo, id_mapping

# Function to get recommendations for a given customer ID
def get_recommendations(algo, df, selected_customer_id):
    # List of items rated by the user
    u_pid = df[df['customer_id'] == selected_customer_id]['product_name'].unique()

    # List of all items
    product_ids = df['product_name'].unique()

    # List of items not rated by the user
    pids_to_predict = np.setdiff1d(product_ids, u_pid)

    testset = [[selected_customer_id, product_name, 0.] for product_name in pids_to_predict]
    predictions = algo.test(testset)

    pred_ratings = np.array([pred.est for pred in predictions])

    # Sort predictions to get top recommendations
    sorted_indices = np.argsort(pred_ratings)[::-1]
    top_recs = pids_to_predict[sorted_indices][:5]
    return top_recs
def convert_product_name(row):
    category = row['product_category_name_english']
    product_id = str(row['product_id'])  # Convert product_id to string
    # Create a generic product name by concatenating category abbreviation and a numerical index
    category_abbreviation = ''.join(word[:].upper() for word in category.split())
    index = int(''.join(filter(str.isdigit, product_id)))  # Extract numerical part from product_id
    product_name = f'{category_abbreviation}_{index}'
    return product_name



# Main function to run the Streamlit app
def main():
    st.title("Recommendation System")

    # Load data
    file_path = r'C:\Users\prati\OneDrive\Desktop\Project\joined_data_set.csv'
    df = pd.read_csv(file_path)

    # Convert product IDs to product names
    df['product_name'] = df.apply(convert_product_name, axis=1)

    # Load pre-built model
    algo, _ = load_model()

    # # Convert customer IDs to numeric form
    # df['numeric_customer_id'] = df['customer_id'].apply(lambda x: int(x.split('_')[-1]))
    # Create a dictionary to map unique customer IDs to sequential numbers
    id_mapping = {id_: idx + 1 for idx, id_ in enumerate(df['customer_id'].unique())}

    # Convert customer IDs to simple generic unique numbers
    df['customer_id'] = df['customer_id'].map(id_mapping)

    # Dropdown to select customer ID (numeric)
    selected_customer_id = st.selectbox("Select Customer ID", df['customer_id'].unique())

    # Get recommendations for selected customer ID
    if st.button("Get Recommendations"):
        recommendations = get_recommendations(algo, df, selected_customer_id)
        st.write("Top 5 Recommended Products:")
        for i, rec in enumerate(recommendations, 1):
            st.write(f"{i}. {rec}")

if __name__ == "__main__":
    main()


import streamlit as st
import pandas as pd
import plotly.express as px
from streamlit_option_menu import option_menu
from numerize.numerize import numerize
import time
from streamlit_extras.metric_cards import style_metric_cards
# st.set_option('deprecation.showPyplotGlobalUse', False)
import plotly.graph_objs as go
import matplotlib.pyplot as plt
#from kafka import KafkaConsumer
import json
import time
#import surprise
import seaborn as sns
import numpy as np
import pickle
import joblib
import streamlit.components.v1 as components
#import gdown



st.set_page_config(page_title="Dashboard",page_icon="🌍",layout="wide")
theme_plotly = None 
st.set_option('deprecation.showPyplotGlobalUse', False)

def delay_indicate(delay_value):
    if delay_value<0:
        return "Delay"
    else:
        return "No Delay"

def labels_diff(td):
    if td<0:
        return "Before Delivery"
    if td==0:
        return "Within 1 day"
    if td<=7:
        return "Within 1 Week"
    if td<=31:
        return "Within 1 Month"
    else:
        return "After 1 Month"

def answer_diff(ad):
    if ad==0:
        return "Within 1 day"
    if ad<=2:
        return "Within 2 days"
    if ad<=7:
        return "Within 1 Week"
    if ad<=31:
        return "Within 1 Month"
    else:
        return "After 1 Month"

@st.cache_resource(show_spinner=False)
def data():     
    df=pd.read_csv('joined_data_set.csv')
    files={'order_items':['order_id',    'order_item_id',    'product_id',    'seller_id',    'shipping_limit_date',    'price',    'freight_value'],
    'order_payments':['order_id',    'payment_sequential',    'payment_type',    'payment_installments',    'payment_value'],
    'order_reviews':['review_id',    'order_id',    'review_score',    'review_creation_date',    'review_answer_timestamp',    'review_answer_time_difference',   'review_answer_time_difference_label'],
    'orders':['order_id',    'customer_id',    'order_status',    'order_purchase_timestamp',    'order_approved_at',    'order_delivered_carrier_date',    'order_delivered_customer_date',    'order_estimated_delivery_date',    'order_month_year',    'weekday',    'order_purchase_hour',    'difference_purchased_delivered',    'estimated_delivered_difference',    'delay_or_no_delay'],
    'customers':['customer_id',    'customer_unique_id',    'customer_zip_code_prefix',    'customer_city',    'customer_state'],
    'seller':['seller_id',    'seller_zip_code_prefix',     'seller_city',     'seller_state'],
    'product':['product_id',    'product_category_name_english',    'product_name_lenght',    'product_description_lenght',    'product_photos_qty',    'product_weight_g',    'product_length_cm',    'product_height_cm',    'product_width_cm']}
    
    
    # Transformations

    customers_df=files['customers']
    order_items_df=files['order_items']
    order_payments_df=files['order_payments']
    order_reviews_df=files['order_reviews']
    orders_df=files['orders']
    seller_df=files['seller']
    product_df=files['product']
    return transform(df),customers_df,order_items_df,order_payments_df,order_reviews_df,orders_df,seller_df,product_df




def transform(df):
    df=df[df['order_status']=='delivered']
    df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
    df['order_month_year'] = df['order_purchase_timestamp'].dt.to_period('M')

    # Incomplete data post 2018-08
    df=df[df['order_month_year']<'2018-09']

    df['weekday']=df['order_purchase_timestamp'].dt.day_name()
    df['order_purchase_hour']=df['order_purchase_timestamp'].dt.hour

    df['order_estimated_delivery_date']=pd.to_datetime(df['order_estimated_delivery_date'])
    df['order_delivered_customer_date']=pd.to_datetime(df['order_delivered_customer_date'])
    df['difference_purchased_delivered']=(df['order_delivered_customer_date']-df['order_purchase_timestamp']).dt.days
    df['estimated_delivered_difference']=(df['order_estimated_delivery_date']-df['order_delivered_customer_date']).dt.days
    df['delay_or_no_delay']=df['estimated_delivered_difference'].apply(delay_indicate)

    df['review_creation_date']=pd.to_datetime(df['review_creation_date'])
    df['review_answer_timestamp']=pd.to_datetime(df['review_answer_timestamp'])
    df['review_answer_time_difference']=(df['review_answer_timestamp'] - df['review_creation_date']).dt.days
    df['review_answer_time_difference_label']=df['review_answer_time_difference'].apply(answer_diff)
    return df




#with open('style.css')as f:
    #st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html = True)




# Function to load the pre-built model
@st.cache_resource(show_spinner=False)
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

@st.cache_data(show_spinner=False)
def churn_data():
    df = pd.read_csv('https://github.com/littlebear27/E-commerce-KPI-dashboards-and-analytics/blob/main/streamlit_webapp/extra/clv_data.csv')
    rfm_df = pd.read_csv('https://github.com/littlebear27/E-commerce-KPI-dashboards-and-analytics/blob/main/streamlit_webapp/extra/rfm_df.csv')
    return df,rfm_df




def predict_churn(data):
    relevant_features = ['freight_value', 'price', 'monetary_value','review_score','payment_installments','customer_state','frequency']
    input_data = data[relevant_features]
    prediction=None
    # Make prediction
    # with open('decisionTree.pkl', 'rb') as model_file:
    #     churn_model = pickle.load(model_file)
    churn_model=joblib.load('decisionTree.joblib')

    prediction = churn_model.predict(input_data)
    return prediction

def convert_product_name(row):
    category = row['product_category_name_english']
    product_id = str(row['product_id'])  # Convert product_id to string
    # Create a generic product name by concatenating category abbreviation and a numerical index
    category_abbreviation = ''.join(word[:].upper() for word in category.split())
    index = int(''.join(filter(str.isdigit, product_id)))  # Extract numerical part from product_id
    product_name = f'{category_abbreviation}_{index}'
    return product_name




#---------------------------------------------------dash boards
#"Ratings","Geo-loc","Categories","CRM","Time"

def dash_board_Geo():
   Geo(data())

def dash_board_Review():
   Review(data())

def dash_board_Preferance():
   Preferance(data())
   
def dash_board_Trend():
   Trend(data())

def dash_board_Home():
    pass


def dash_board_RFM():
    RFM()

def dash_board_Churn():
    Churn()

def dash_board_Recommend():
    Recommend()

def RFM():
    df,rfm_df=churn_data()
    st.subheader("Customer Segmentation")

    # Sidebar for segmentation options
    selected_segmentation = st.sidebar.selectbox("Select Segmentation", ["Recency", "Monetary", "Review Score", "Service Satisfaction", "RFM Score Segmentation"])

    # Segmentation based on Recency
    if selected_segmentation == "Recency":
        bins = [-1, 60, 120, 220, float('inf')]
        labels = ['Recent', 'Moderately Recent', 'Not So Recent', 'Inactive']
        df['RecencySegment'] = pd.cut(df['recency'], bins=bins, labels=labels, right=False)

        # Visualization
        st.markdown("Recency Segmentation")
        plt.figure(figsize=(10, 5))
        sns.countplot(x='RecencySegment', data=df, order=labels, palette='viridis')
        plt.title('Recency Segmentation')
        plt.xlabel('Recency Segments')
        plt.ylabel('Number of Customers')
        st.pyplot()

        # Show insights button
        with st.expander("Show Insights"):
            st.subheader("Interpretation and Business Insights")
            st.markdown("""
1. The average recency is approximately 239.54 days. This indicates that, on average, customers made their last purchase or engagement about 240 days ago.
2. The minimum recency value is 0.These customers represent very recent activity.
3. Customers with low recency values may be more receptive to promotions or re-engagement campaigns.
4. Higher recency values represents customers who have not made a purchase for an extended period. Strategies to re-engage or retain these      customers may be needed.
*Strategies to reduce customer's recencies:
1. Promotional Campaigns: Launch targeted promotional campaigns, such as discounts, special offers, or limited-time promotions
2. Loyalty Programs: Rewards customers for frequent purchases. Offer points, discounts, or exclusive access to encourage repeat business.
3. Reactivation Campaigns: Identify dormant customers (those with high recency values) and launch targeted reactivation campaigns. Provide special incentives or discounts to encourage them to return.
                    """)

    # Segmentation based on Monetary
    elif selected_segmentation == "Monetary":
        # Calculate the first and third quartiles (Q1 and Q3)
        Q1 = df['monetary_value'].quantile(0.25)
        Q3 = df['monetary_value'].quantile(0.75)

        # Calculate the IQR (Interquartile Range)
        IQR = Q3 - Q1

        # Define the lower and upper bounds for identifying outliers
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Filter out values outside the bounds to remove outliers
        monetary_data_no_outliers = df['monetary_value'][(df['monetary_value'] >= lower_bound) & (df['monetary_value'] <= upper_bound)]
        monetary_data_no_outliers['MonetarySegment'] = pd.cut(monetary_data_no_outliers, bins=[0, 50, 150, 300, 600], labels=['Low', 'Medium', 'High', 'Very High'])
        st.markdown("Monetary Segmentation")
        # Visualize segmentation
        print(type(monetary_data_no_outliers))
        plt.figure(figsize=(10, 5))
        sns.countplot(x=monetary_data_no_outliers['MonetarySegment'], palette='viridis')
        plt.title('Customer Segmentation based on Monetary Values (Without Outliers)')
        plt.xlabel('Monetary Segment')
        plt.ylabel('Number of Customers')
        st.pyplot()

        # Show insights button
        with st.expander("Show Insights"):
            st.subheader("Interpretation and Business Insights")
            st.markdown("""
1. The average monetary value per customer is 376.04. This represents the central tendency of spending across the customer base.
2. The standard deviation is relatively high at 1,671.11, indicating a wide dispersion in spending. 
   This suggests that customer spending varies significantly, and there may be outliers with exceptionally high monetary values.
3. The wide range in monetary values highlights the heterogeneity in customer spending habits.
4. The concentration of values in the lower quartiles (Q1 and Q2) suggests that a considerable portion of customers has relatively moderate     spending.
5. Identification and focus on high-value customers may lead to strategies that enhance customer retention and increase overall revenue.

* To gain more value from customers:
1. Personalized Marketing
2. Customer Segmentation: Segment customers based on their purchasing behavior, demographics, or preferences.
   Design targeted marketing strategies for each segment to address their specific needs and preferences.
3. Exclusive Offers for High-Value Customers
                        """)

    # Segmentation based on Review Score
    elif selected_segmentation == "Review Score":
        reviewScore = df['review_score'].value_counts(normalize=True) * 100
        
        # Visualization
        st.markdown("Review Score Segmentation")
        plt.figure(figsize=(8, 10))
        plt.pie(reviewScore, labels=reviewScore.index, autopct='%1.1f%%', colors=['#66b3ff', '#99ff99', 'lightpink'], startangle=90)
        #plt.title('Review Score Segmentation')
        st.pyplot()

        # Show insights button
        with st.expander("Show Insights"):
            st.subheader("Interpretation and Business Insights")
            st.markdown("""
1. The majority of customers are highly satisfied, as indicated by the high percentage in the 5.0 category.
2. A notable portion of customers gave scores indicating moderate to high satisfaction (4.0).
3. Around 23% of customers are not satified.
                        """)

    # Segmentation based on Service Satisfaction
    elif selected_segmentation == "Service Satisfaction":
        ontimeDelivery_review = df[df.delivered_estimated > 0][['review_score', 'approved_carrier', 'carrier_delivered', 'delivered_estimated', 'purchased_delivered']].mean()
        lateDelivery_review = df[df.delivered_estimated < 0][['review_score', 'approved_carrier', 'carrier_delivered', 'delivered_estimated', 'purchased_delivered']].mean()
        comparison_review = pd.DataFrame([ontimeDelivery_review, lateDelivery_review]).T
        comparison_review.rename(columns={0: 'on time delivery', 1: 'late delivery'}, inplace=True)

        # Visualization
        st.markdown("Comparison of On-time vs. Late Delivery Orders")
        plt.figure(figsize=(10, 5))
        comparison_review.plot(kind='barh', colormap='viridis', ax=plt.gca())
        plt.title('Comparison of On-time Vs. Late Delivery Orders', fontweight='bold', fontsize=20)
        plt.xlabel('Average Value')
        plt.ylabel('Metrics')
        st.pyplot()

        # Show insights button
        with st.expander("Show Insights"):
            st.subheader("Interpretation and Business Insights")
            st.markdown("""
1. On-Time Delivery Average Review Score: 4.038101
   Customers who received their deliveries on time gave an average review score of approximately 4.04. 
   This indicates a relatively high level of satisfaction among customers who experienced on-time deliveries.

2. Late Delivery Average Review Score: 2.583333
   Customers who experienced late deliveries gave an average review score of approximately 2.58. 
   This suggests a lower level of satisfaction among customers who faced delays in their deliveries.
3. Timely delivery appears to be positively correlated with higher customer satisfaction.
4. Late deliveries may have a negative impact on the overall customer experience, leading to lower review scores.
5. It took 25 days on an average for the orders to get delivered to the customers for late delivery orders, 
   while it took just 9 days for on-time delivery orders.
6. On an average, orders were delivered 14 days before the estimated date of delivery for on-time delivered orders.


#### Business insights:

1. Carrier Selection: Emphasize the importance of selecting and using approved carriers, as they seem to contribute to better on-time performance and higher customer satisfaction.

2. Operational Efficiency: Focus on improving operational efficiency for carriers involved in the delivery process to reduce the time gap between the estimated and actual delivery times.

3. Customer Satisfaction: Timely deliveries positively impact customer satisfaction. Implement strategies to enhance on-time delivery performance, as it correlates with higher review scores.

4. Communication and Expectations: Ensure accurate estimation of delivery times and communicate them clearly to customers. This can help manage customer expectations and reduce dissatisfaction associated with late deliveries.
                        """)

    # Segmentation based on RFM Score
    else:
        # RFM Score Calculation
        rfm_df['recency_score'] = pd.qcut(rfm_df['recency'], q=4, labels=False, duplicates='drop')
        rfm_df['frequency_score'] = pd.qcut(rfm_df['frequency'], q=4, labels=False, duplicates='drop')
        rfm_df['monetary_score'] = pd.qcut(rfm_df['monetary_value'], q=4, labels=False, duplicates='drop')
        rfm_df['rfm_score'] = rfm_df['recency_score'] + rfm_df['frequency_score'] + rfm_df['monetary_score']
        correlation_matrix = rfm_df[['recency_score', 'frequency_score', 'monetary_score', 'rfm_score']].corr()

        # Visualization
        st.markdown("Correlation Heatmap of RFM Scores")
        plt.figure(figsize=(10, 5))
        sns.heatmap(correlation_matrix, annot=True, cmap='viridis', fmt=".2f", linewidths=.5)
        plt.title('Correlation Heatmap of RFM Scores', fontsize=16)
        st.pyplot()

        with st.expander("Show Insights"):
            st.subheader("Interpretation")
            st.markdown("""
1. The RFM score is strongly influenced by monetary value, with a notable positive correlation.
2. Recency has a moderate positive impact on the overall RFM score, but the relationship is not as strong as monetary value.
3. Frequency has a moderate positive impact on both monetary value and the overall RFM score.
4. Therefore, we will use rfm_score to decide churn                        
                        """)

        # Define customer segments based on RFM score
        segment_labels = ['Inactive', 'Regular', 'Engaged']
        rfm_df['segment'] = pd.cut(rfm_df['rfm_score'], bins=[0, 3, 6, 9], labels=segment_labels)

        # Create a container to group the plots
        st.container()
        st.markdown("Based on RFM score")

        # Create three columns for the plots
        col1, col2, col3 = st.columns(3)

        # Bar Plot
        with col1:
            st.markdown("Customer Segmentation")
            plt.figure(figsize=(8, 6))
            sns.countplot(x='segment', data=rfm_df, palette='viridis')
            plt.title('Customer Segmentation')
            st.pyplot() 

        # Pie Chart
        with col2:
            st.markdown("Segment Distribution")
            segment_counts = rfm_df['segment'].value_counts()
            colors = sns.color_palette('viridis')
            explode = (0.01, 0.01, 0.1)
            plt.figure(figsize=(8, 6))
            plt.pie(segment_counts, labels=segment_counts.index, autopct='%1.2f%%', colors=colors, startangle=90, explode=explode)
            st.pyplot()

        # Count Plot
        with col3:
            st.markdown("No. of Customers in RFM Scores")
            plt.figure(figsize=(8, 6))
            sns.countplot(x='rfm_score', hue='segment', data=rfm_df, palette='viridis')
            plt.title('No. of Customers in RFM Scores')
            st.pyplot()
            
            #Show insights button
        
        with st.expander("Segment Interpretation"):
            st.markdown("""
            I. rfm score(0-3): Inactive:
            1. These customers have low recency, frequency, and monetary contributions.
            2. They might be considered less engaged or potentially at risk of churning.

            II. rfm score(4-6): Regular:
            1. This group represents customers with moderate recency, frequency, and monetary values.
            2. They are considered reasonably engaged and contribute to the business consistently.

            III. rfm score(7-9): Engaged:
            1. This segment includes customers with high recency, frequency, and monetary contributions
            2. These are the most valuable and engaged customers, likely to be loyal and provide significant revenue.
            """)


def Churn():
    st.title("Churn Prediction")

    # Input form for the user to provide feature values
    freight_value = st.number_input("Enter freight value:",placeholder="Type freight value...", value=None)
    price = st.number_input("Enter price:",placeholder="Type price...", value=None)
    monetary_value = st.number_input("Enter monetary value:",placeholder="Type monetary value...", value=None)
    review_score =st.slider("Enter review score: Range: 1-5", min_value=1, max_value=5, value=5)
    payment_installments = st.number_input("Enter payment installments:",placeholder="Type payment installments...", value=None)
    customer_state = st.number_input("Enter customer state:  Centralwestern: 0, Northeastern: 1, Northern: 2, Southeastern: 3, Southern: 4",placeholder="Type a number of customer state...", value=None)
    frequency = st.number_input("Enter frequency:",placeholder="Type a frequency...", value=None)


    if st.button('find if churn'):
    # Create a DataFrame with the user input
        input_data = pd.DataFrame([[freight_value, price, monetary_value,review_score,payment_installments,customer_state,frequency]],
                                columns=['freight_value', 'price', 'monetary_value','review_score','payment_installments','customer_state','frequency'])

        # Display the input data
        st.subheader("Input Data:")
        st.write(input_data)

        # Make the churn prediction using the model
        prediction = predict_churn(input_data)

        # Display the prediction
        st.subheader("Churn Prediction:")
        if prediction[0] == 1:
            st.write("The customer is predicted to churn.")
        else:
            st.write("The customer is predicted to not churn.")

        with st.expander("Customer Retention Suggestions"):
            explain()


def Recommend():
    st.title("Recommendation System")

    # Load data
    file_path = r'joined_data_set.csv'
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

    # df['customer_id'].unique().sort_value()
    # Dropdown to select customer ID (numeric)
    selected_customer_id = st.selectbox("Select Customer ID",df['customer_id'].unique())

    # Get recommendations for selected customer ID
    if st.button("Get Recommendations"):
        recommendations = get_recommendations(algo, df, selected_customer_id)
        st.write("Top 5 Recommended Products:")
        for i, rec in enumerate(recommendations, 1):
            st.write(f"{i}. {rec}")


def Geo(df): #customer,sellers,order_customer
    #1 customers_df,
    #2 order_items_df,
    #3 order_payments_df,
    #4 order_reviews_df,
    #5 orders_df,
    #6 seller_df,
    #7 product_df
    df1=df[0]
    customers_df=df1[df[1]]
    sellers_df=df1[df[6]]
    orders_and_customers=df1[list(set(df[1]).union(set(df[5])))]
    orders_df=df1[df[5]]

#--------------------------------------metics

    number_cities_serviced=customers_df['customer_city'].unique().size

    avg_days_to_deliver=int(round(orders_df['difference_purchased_delivered'].mean()))
    
    delay_classes_count=orders_df['delay_or_no_delay'].value_counts()
    delay_classes_count_perc= round((delay_classes_count.div(delay_classes_count.sum())*100),2)
    on_time_perc=delay_classes_count_perc['No Delay']




    # freight_value = float(pd.Series(df['freight_value']).mean())
    # payment_value = float(pd.Series(df['payment_value']).sum())
    # review_score = float(pd.Series(df['review_score']).mean())
    # Tfreight_value= float(pd.Series(df['freight_value']).sum())

#------------------------------------------------------empty graph 
    empty=px.scatter(x=None,y=None)
    empty.update_layout(showlegend=False)
    empty.update_xaxes(visible=False)
    empty.update_yaxes(visible=False)
    empty.add_annotation(
    text="Low Data for Analysis",
    xanchor="center",
    yanchor="middle",
    showarrow=False,
    font=dict(
        size=32,
        color="black"
    ),
    opacity=0.5
    )    
#---------------------------------------------------------metrics    
    total1,total2,total3=st.columns(3,gap='small')
    with total1:
        st.metric(label="Number of Cities Serviced",value=f"{number_cities_serviced}")

    with total2:
        st.metric(label="Average Number of Days to Deliver",value=f"{numerize(avg_days_to_deliver)}")

    with total3:
        st.metric(label="Percentage of Orders Delivered on-time",value=f"{on_time_perc}")

#---------------- graphs
    state_number_of_orders_map=orders_and_customers.groupby('customer_state')['order_id'].nunique().reset_index(name='number_of_orders_for_state')

    dash3_fig3=empty
    if not state_number_of_orders_map.empty:
        dash3_fig3=px.treemap(state_number_of_orders_map,path=['customer_state'],values='number_of_orders_for_state',color_discrete_sequence=px.colors.qualitative.Pastel)
        dash3_fig3.update_layout(title_text="Order Concentration by States (States by Number of Orders)",
                                template="simple_white")



    city_delivery_days_mean=orders_and_customers.groupby('customer_city')['difference_purchased_delivered'].mean().sort_values()
    city_delivery_days_mean_lowest_5=city_delivery_days_mean.head(5)

    dash3_fig4=empty
    if not city_delivery_days_mean_lowest_5.empty:
        dash3_fig4=px.bar(x=city_delivery_days_mean_lowest_5.index, 
                    y=city_delivery_days_mean_lowest_5.values, 
                    title="Top 5 Cities with Lowest Average Delivery Time",
                    labels={"x": "City", "y": "Average Number of Days to Deliver"},
                    color_discrete_sequence=px.colors.qualitative.Pastel, 
                    template="simple_white")
    
    cities_mean_delay=orders_and_customers.groupby('customer_city')['estimated_delivered_difference'].mean().sort_values()
    cities_mean_delay_top5_delay=cities_mean_delay.head(5)

    dash3_fig5=empty
    if not cities_mean_delay_top5_delay.empty:
        dash3_fig5=px.bar(x=cities_mean_delay_top5_delay.index, 
                y=cities_mean_delay_top5_delay.values, 
                title="Top 5 Cities with Highest Average Delivery Delay compared to Estimated Delivery Date",
                labels={"x": "City", "y": "Average delivery delay (in days)"},
                color_discrete_sequence=px.colors.qualitative.Pastel, 
                template="simple_white")

    seller_state_city=sellers_df.groupby(['seller_state','seller_city'])['seller_id'].nunique().sort_values(ascending=False).reset_index(name='number_of_sellers')
    seller_state_city_20plus=seller_state_city[seller_state_city['number_of_sellers']>20]

    dash3_fig2=empty
    if not seller_state_city_20plus.empty:
        dash3_fig2=px.sunburst(seller_state_city_20plus,path=['seller_state','seller_city'],values='number_of_sellers',color_discrete_sequence=px.colors.qualitative.Pastel)
        dash3_fig2.add_annotation(
            text="SP:  São Paulo<br>RJ: Rio de Janeiro<br>MG: Minas Gerais<br>DF: Distrito Federal<br>RS: Rio Grande do Sul<br>PR: Paraná",
            xanchor="left",
            x=0.7,
            yanchor="middle",
            y=0.8,
            showarrow=False,
            font=dict(
                size=12,
                color="black"
            ),
            align="center",
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="#636363",
            ax=20,
            ay=-30,
            bordercolor="#c7c7c7",
            borderwidth=2,
            borderpad=4,
            bgcolor="#ff7f0e",
            opacity=0.8
        )

        dash3_fig2.update_layout(title_text="Seller Concentration by States and Cities (States and Cities by Number of Sellers)",
                                    template="simple_white")
        

    state_city_customer_count=customers_df.groupby(['customer_state','customer_city'])['customer_unique_id'].count().sort_values(ascending=False).reset_index(name='number_of_customers')
    state_city_customer_count_more_than_500=state_city_customer_count[state_city_customer_count['number_of_customers']>=500]

    dash3_fig1=empty
    if not state_city_customer_count_more_than_500.empty:
        dash3_fig1=px.sunburst(state_city_customer_count_more_than_500,path=['customer_state','customer_city'],values='number_of_customers',color_discrete_sequence=px.colors.qualitative.Pastel)

        dash3_fig1.add_annotation(
            text="SP:  São Paulo<br>RJ: Rio de Janeiro<br>MG: Minas Gerais<br>DF: Distrito Federal<br>RS: Rio Grande do Sul<br>PR: Paraná",
            xanchor="left",
            x=0.7,
            yanchor="middle",
            y=0.8,
            showarrow=False,
            font=dict(
                size=12,
                color="black"
            ),
            align="center",
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="#636363",
            ax=20,
            ay=-30,
            bordercolor="#c7c7c7",
            borderwidth=2,
            borderpad=4,
            bgcolor="#ff7f0e",
            opacity=0.8
        )

        dash3_fig1.update_layout(title_text="Customer Concentration by States and Cities (States and Cities by Number of Customers)",
                            template="simple_white")

    left,right=st.columns(2)
    left.plotly_chart(dash3_fig4,use_container_width=True)
    right.plotly_chart(dash3_fig5,use_container_width=True)
    st.plotly_chart(dash3_fig3,use_container_width=True)
    left2,right2=st.columns(2)
    left2.plotly_chart(dash3_fig1,use_container_width=True)
    right2.plotly_chart(dash3_fig2,use_container_width=True)



    ##------------------------------------------------------uncomment to rerunning data
    # time.sleep(15) 
    # st.experimental_rerun()

def Trend(df):#
    #1 customers_df,
    #2 order_items_df,
    #3 order_payments_df,
    #4 order_reviews_df,
    #5 orders_df,
    #6 seller_df,
    #7 product_df
    df1=df[0] 
    customers_df=df1[df[1]]
    orders_df=df1[df[5]]
    orders_and_order_items=df1[list(set(df[2]).union(set(df[5])))]
    orders_and_customers=df1[list(set(df[1]).union(set(df[5])))]

#--------------------------------------metics
    
    total_customers = customers_df['customer_unique_id'].nunique()

    order_sales_time_m=orders_and_order_items.groupby('order_month_year')['price'].sum().sort_index()
    total_revenue = sum(order_sales_time_m)
    revenue_per_customer = round (total_revenue/total_customers, 2)

    total_orders=orders_df['order_id'].nunique()
    avg_num_of_order_per_customer= round(total_orders/total_customers,2)

    # payment_value = float(pd.Series(df['payment_value']).sum())
    # review_score = float(pd.Series(df['review_score']).mean())
    # Tfreight_value= float(pd.Series(df['freight_value']).sum())

#------------------------------------------------------empty graph 
    empty=px.scatter(x=None,y=None)
    empty.update_layout(showlegend=False)
    empty.update_xaxes(visible=False)
    empty.update_yaxes(visible=False)
    empty.add_annotation(
    text="Select Some Things in filters",
    xanchor="center",
    yanchor="middle",
    showarrow=False,
    font=dict(
        size=32,
        color="black"
    ),
    opacity=0.5
    )    
#---------------------------------------------------------metrics    
    total1,total2,total3=st.columns(3,gap='small')
    with total1:
        st.metric(label="Total Number of Customer",value=f"{numerize(total_customers)}")

    with total2:
        st.metric(label="Average Revenue per Customer",value=f"{revenue_per_customer}")

    with total3:
        st.metric(label="Average # of Orders per Customer",value=f"{avg_num_of_order_per_customer}")

#---------------- graphs
    order_volume_time = orders_df['order_month_year'].value_counts().sort_index()

    dash1_fig1=empty
    if not order_volume_time.empty:
        dash1_fig1=px.line(x=order_volume_time.index.to_timestamp(),
                   y=order_volume_time.values,
                   title="Order Volume by Time",
                   labels={"x": "Month-Year", "y": "Number of Orders"},
                   color_discrete_sequence=px.colors.sequential.Plasma, 
                   template="simple_white")

    if orders_and_order_items['order_month_year'].nunique() != 1:
        order_sales_time=orders_and_order_items.groupby('order_month_year')['price'].sum().sort_index()
    dash1_fig2=empty
    if not order_sales_time.empty :
        dash1_fig2=px.line(x=order_sales_time.index.to_timestamp(),
                   y=order_sales_time.values,
                   title='Total Sales by Time',
                   labels={'x':'Month-Year','y':'Total Sales'},
                   color_discrete_sequence=px.colors.sequential.Plasma,
                   template="simple_white")
    
    num_of_customer_by_time=orders_and_customers.groupby('order_month_year')['customer_unique_id'].nunique().sort_index()
    dash1_fig3=empty
    if not num_of_customer_by_time.empty:
        dash1_fig3=px.line(x=num_of_customer_by_time.index.to_timestamp(),
                   y=num_of_customer_by_time.values,
                   title='Number of Customers by Time',
                   labels={'x':'Month-Year','y':'Number of Distinct Customers'},
                   color_discrete_sequence=px.colors.sequential.Plasma,
                   template="simple_white")
        

    number_orders_delayed=orders_df.groupby('delay_or_no_delay')['delay_or_no_delay'].count()
    number_orders_delayed_perc=number_orders_delayed.div(number_orders_delayed.sum())*100
    number_orders_delayed_perc=number_orders_delayed_perc.round(2)
    dash1_fig4=empty
    if not number_orders_delayed_perc.empty:
        dash1_fig4 = px.pie(number_orders_delayed_perc, 
              title="Delivery Performance (Percentage of Orders by Delivery Status)",
              names=number_orders_delayed_perc.index,
              values=number_orders_delayed_perc.values,
              color_discrete_sequence=px.colors.qualitative.Pastel, 
              template="simple_white")

    l,r=st.columns(2)
    l.plotly_chart(dash1_fig1,use_container_width=True)
    r.plotly_chart(dash1_fig2,use_container_width=True)
    l1,r1=st.columns(2)
    l1.plotly_chart(dash1_fig3,use_container_width=True)
    r1.plotly_chart(dash1_fig4,use_container_width=True)

    

def Preferance(df):#
    #1 customers_df,
    #2 order_items_df,
    #3 order_payments_df,
    #4 order_reviews_df,
    #5 orders_df,
    #6 seller_df,
    #7 product_df
    df1=df[0]
    orders_and_order_payments_df=df1[list(set(df[3]).union(set(df[5])))]
    orders_and_order_items_and_products_dataset_translated=df1[list(set(df[2]).union(set(df[5])).union(set(df[7])))]
    orders_df=df1[df[5]]
    order_items_df=df1[df[2]]
    product_category_name_translation_df=df1[df[7]]

#--------------------------------------metics

    number_of_product_categories=product_category_name_translation_df['product_category_name_english'].nunique()


    highest_rev_product_cat=orders_and_order_items_and_products_dataset_translated.groupby('product_category_name_english')['price'].sum().sort_values(ascending=False)
    highest_rev_product_cat_perc=round((highest_rev_product_cat.div(highest_rev_product_cat.sum())*100),2)
    highest_rev_product_cat_perc_top=highest_rev_product_cat_perc.head(1)
    highest_rev_product_cat_perc_top_prod=highest_rev_product_cat_perc_top.index.tolist()[0]


    highest_rev_product_cat_perc_top_revenue=highest_rev_product_cat_perc_top.values.tolist()[0]


    # freight_value = float(pd.Series(df['freight_value']).mean())
    # payment_value = float(pd.Series(df['payment_value']).sum())
    # review_score = float(pd.Series(df['review_score']).mean())
    # Tfreight_value= float(pd.Series(df['freight_value']).sum())

#------------------------------------------------------empty graph 
    empty=px.scatter(x=None,y=None)
    empty.update_layout(showlegend=False)
    empty.update_xaxes(visible=False)
    empty.update_yaxes(visible=False)
    empty.add_annotation(
    text="Select Some Things in filters",
    xanchor="center",
    yanchor="middle",
    showarrow=False,
    font=dict(
        size=32,
        color="black"
    ),
    opacity=0.5
    )    
#---------------------------------------------------------metrics    
    total1,total2,total3=st.columns(3,gap='small')
    with total1:
        st.metric(label="Total Number of Product Categories",value=f"{number_of_product_categories}")

    with total2:
        st.metric(label="Product Category Contributing Highest to Revenue",value=f"{highest_rev_product_cat_perc_top_prod}")

    with total3:
        st.metric(label="Revenue Contribution (in %) of top Product Category",value=f"{highest_rev_product_cat_perc_top_revenue:,.0f}")

#---------------- graphs
    monetary_total_payment_type_by_time=orders_and_order_payments_df.groupby(['order_month_year','payment_type'])['payment_value'].sum().sort_index().unstack(fill_value=0)
    monetary_total_payment_type_by_time_melted = monetary_total_payment_type_by_time.reset_index().melt(id_vars="order_month_year", var_name="payment_type", value_name="total")
    monetary_total_payment_type_by_time_melted['order_month_year'] = monetary_total_payment_type_by_time_melted['order_month_year'].astype(str)

    dash2_fig1=empty
    if not monetary_total_payment_type_by_time_melted.empty:
        dash2_fig1 = px.bar(monetary_total_payment_type_by_time_melted, x="order_month_year", y="total", color="payment_type", title="Preferred Payment Method across Time [Total Payment (in $) by Payment Mode and Time]",
             labels={"total": "Total Payment (in $)", "order_month_year": "Month-Year", "payment_type": "Payment Type"},
             color_discrete_sequence=px.colors.qualitative.Pastel, template="simple_white")





    number_of_products_ordered_by_product_type=orders_and_order_items_and_products_dataset_translated.groupby('product_category_name_english')['product_category_name_english'].count().sort_values(ascending=False)
    number_of_products_ordered_by_product_type_percentage=number_of_products_ordered_by_product_type.div(number_of_products_ordered_by_product_type.sum())*100
    number_of_products_ordered_by_product_type_percentage_top5=number_of_products_ordered_by_product_type_percentage.head(5).round(2)
    dash2_fig2=empty
    if not number_of_products_ordered_by_product_type_percentage_top5.empty:
        dash2_fig2 = px.bar(x=number_of_products_ordered_by_product_type_percentage_top5.index, 
              y=number_of_products_ordered_by_product_type_percentage_top5.values, 
              title="Top 5 Product Categories based on Number of Orders (in %)",
              labels={"x": "Product Category", "y": "Percentage of Orders"},
              color_discrete_sequence=px.colors.qualitative.Pastel, 
              template="simple_white")
    
    number_of_orders_by_weekday=orders_df['weekday'].value_counts()
    sorter=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    number_of_orders_by_weekday=number_of_orders_by_weekday.reindex(sorter)

    percentage_of_orders_by_weekday=number_of_orders_by_weekday.div(number_of_orders_by_weekday.sum())*100
    percentage_of_orders_by_weekday=percentage_of_orders_by_weekday.round(2)
    dash2_fig3=empty
    if not percentage_of_orders_by_weekday.empty:
        dash2_fig3 = px.bar(x=percentage_of_orders_by_weekday.index, 
              y=percentage_of_orders_by_weekday.values, 
              title="Preferred Day of Week for Ordering (Number of Orders (in %) by Day of Week)",
              labels={"x": "Day of Week", "y": "Percentage of orders"},
              color_discrete_sequence=px.colors.qualitative.Pastel, 
              template="simple_white")
        
    number_of_orders_by_hour=orders_df.groupby('order_purchase_hour')['order_id'].count().sort_index()
    perc_of_orders_by_hour=number_of_orders_by_hour.div(number_of_orders_by_hour.sum())*100
    perc_of_orders_by_hour=perc_of_orders_by_hour.round(2)

    dash2_fig4=empty
    if not perc_of_orders_by_hour.empty:
        dash2_fig4 = px.bar(x=perc_of_orders_by_hour.index, 
                    y=perc_of_orders_by_hour.values, 
                    title="Preferred Time of Day for Ordering (Number of Orders (in %) by Time of Day)",
                    labels={"x": "Hour", "y": "Percentage of Orders"},
                    color_discrete_sequence=px.colors.qualitative.Pastel, 
                    template="simple_white",color=perc_of_orders_by_hour.values)
    

    dash2_fig5=empty
    if not order_items_df.empty:
        dash2_fig5=px.histogram(order_items_df,
                            x='price',
                            nbins=1000,
                            title='Product Price Preferrence (Number of Products Ordered by Price) - Histogram)',
                            color_discrete_sequence=px.colors.qualitative.Pastel,
                            template="simple_white")
        dash2_fig5.update_xaxes(range=[0, 750],title_text='Price')
        dash2_fig5.update_yaxes(title_text='Number of Products Ordered')

    
    order_basket=order_items_df.groupby('order_id')['order_id'].count().reset_index(name='basket_size')
    dash2_fig6=empty
    if not order_basket.empty:
        dash2_fig6=px.histogram(order_basket,
                                x='basket_size',
                                title='Preferred Basket Size (Number of Orders per Basket Size) - Histogram',
                                color_discrete_sequence=px.colors.qualitative.Pastel,
                                template="simple_white")
        dash2_fig6.update_xaxes(range=[0, 5], title_text='Basket Size (Number of Products in a Single Order)')
        dash2_fig6.update_yaxes(title_text='Number of Orders')

    st.plotly_chart(dash2_fig1,use_container_width=True)
    left,right=st.columns(2)
    left.plotly_chart(dash2_fig2,use_container_width=True)
    right.plotly_chart(dash2_fig3,use_container_width=True)
    st.plotly_chart(dash2_fig4,use_container_width=True)
    left1,right1=st.columns(2)
    left1.plotly_chart(dash2_fig5,use_container_width=True)
    right1.plotly_chart(dash2_fig6,use_container_width=True)


    

def Review(df):#
    #1 customers_df,
    #2 order_items_df,
    #3 order_payments_df,
    #4 order_reviews_df,
    #5 orders_df,
    #6 seller_df,
    #7 product_df
    df1=df[0]
    orders_and_order_reviews=df1[list(set(df[5]).union(set(df[4])))]
    orders_df=df1[df[5]]
    orders_and_order_reviews['delivery_review_time_difference']=(orders_and_order_reviews['review_creation_date'] - orders_and_order_reviews['order_delivered_customer_date']).dt.days
    orders_and_order_reviews['delivery_review_time_difference_label']=orders_and_order_reviews['delivery_review_time_difference'].apply(labels_diff)

#--------------------------------------metics

    metric_all_order_ids=orders_df['order_id']
    metric_non_duplicated_orders_and_order_reviews=orders_and_order_reviews[~orders_and_order_reviews.duplicated('order_id',keep='last')]
    metric_reviewed_order_ids=metric_non_duplicated_orders_and_order_reviews['order_id']
    metric_non_reviewed_orders=metric_all_order_ids[~metric_all_order_ids.isin(metric_reviewed_order_ids)]
    metric_count_non_reviewed_orders=metric_non_reviewed_orders.count()
    metric_count_reviewed_orders=metric_reviewed_order_ids.count()
    metric_count_reviewed_orders_perc=round((metric_count_reviewed_orders/(metric_count_non_reviewed_orders+metric_count_reviewed_orders)*100),2)

    # freight_value = float(pd.Series(df['freight_value']).mean())
    # payment_value = float(pd.Series(df['payment_value']).sum())
    # review_score = float(pd.Series(df['review_score']).mean())
    # Tfreight_value= float(pd.Series(df['freight_value']).sum())

#------------------------------------------------------empty graph 
    empty=px.scatter(x=None,y=None)
    empty.update_layout(showlegend=False)
    empty.update_xaxes(visible=False)
    empty.update_yaxes(visible=False)
    empty.add_annotation(
    text="Select Some Things in filters",
    xanchor="center",
    yanchor="middle",
    showarrow=False,
    font=dict(
        size=32,
        color="black"
    ),
    opacity=0.5
    )    
#---------------------------------------------------------metrics    

    st.metric(label="Percentage of Orders Reviewed",value=f"{numerize(metric_count_reviewed_orders_perc)}")

#---------------- graphs
    keep_first_orders_and_order_reviews=orders_and_order_reviews[~orders_and_order_reviews.duplicated('order_id',keep='first')]
    time_difference_label_count=keep_first_orders_and_order_reviews['delivery_review_time_difference_label'].value_counts()
    time_difference_label_count_perc=time_difference_label_count.div(time_difference_label_count.sum())*100
    diff_sort=["Within 1 day","Within 1 Week","Within 1 Month","After 1 Month","Before Delivery"]
    time_difference_label_count_perc=time_difference_label_count_perc.reindex(diff_sort)

    
    dash4_fig1=empty
    if not time_difference_label_count_perc.empty:
        dash4_fig1=px.bar(x=time_difference_label_count_perc.index, 
              y=time_difference_label_count_perc.values, 
              title="Swiftness of Review Creation after Order Delivery",
              labels={"x": "Average Time Difference between Order Delivery and Review Creation", "y": "Percentage of Reviewed Orders"},
              color_discrete_sequence=px.colors.qualitative.Pastel, 
              template="simple_white")
        
    answer_difference_label_count=orders_and_order_reviews['review_answer_time_difference_label'].value_counts()
    answer_difference_label_count_perc=answer_difference_label_count.div(answer_difference_label_count.sum())*100
    answer_diff_sort=["Within 1 day","Within 2 days","Within 1 Week","Within 1 Month","After 1 Month"]
    answer_difference_label_count_perc=answer_difference_label_count_perc.reindex(answer_diff_sort)

    dash4_fig2=empty
    if not answer_difference_label_count_perc.empty:
        dash4_fig2=px.bar(x=answer_difference_label_count_perc.index, 
              y=answer_difference_label_count_perc.values, 
              title="Swiftness of Review Reply after Customer Review Creation",
              labels={"x": "Average Time Difference between Review Creation and Review Reply", "y": "Percentage of Reviewed Orders"},
              color_discrete_sequence=px.colors.qualitative.Pastel, 
              template="simple_white")
        

    mean_review_score=orders_and_order_reviews['review_score'].mean()
    median_review_score=orders_and_order_reviews['review_score'].median()
    mode_review_score=int(orders_and_order_reviews['review_score'].mode().values)
    mmm = pd.DataFrame({
        'Measure': ['Mean Review Score', 'Median Review Score', 'Mode Review Score'],
        'Review_Score': [mean_review_score, median_review_score, mode_review_score]
    })
    
    dash4_fig3=empty
    if not mmm.empty:
        dash4_fig3 = px.bar(mmm, 
                    x='Measure', 
                    y='Review_Score', 
                    title="Mean/Median/Mode Review Score", 
                    color_discrete_sequence=px.colors.qualitative.Pastel,
                    template="simple_white")

        dash4_fig3.update_yaxes(title_text='Review Score')



    all_order_ids=orders_df['order_id']
    non_duplicated_orders_and_order_reviews=orders_and_order_reviews[~orders_and_order_reviews.duplicated('order_id',keep='last')]
    reviewed_order_ids=non_duplicated_orders_and_order_reviews['order_id']
    non_reviewed_orders=all_order_ids[~all_order_ids.isin(reviewed_order_ids)]
    count_non_reviewed_orders=non_reviewed_orders.count()
    review_score_count=non_duplicated_orders_and_order_reviews['review_score'].value_counts().sort_index()
    review_score_count.index = review_score_count.index.astype(str)
    review_score_count = pd.concat([review_score_count, pd.Series(count_non_reviewed_orders, index=['No review'])])
    review_score_count_perc=review_score_count.div(review_score_count.sum())*100

    dash4_fig4=empty
    if not review_score_count_perc.empty:
        dash4_fig4=px.bar(x=review_score_count_perc.index, 
              y=review_score_count_perc.values, 
              title="Percentage of Orders by Review Score",
              labels={"x": "Review Score", "y": "Percentage of Orders"},
              color_discrete_sequence=px.colors.qualitative.Pastel, 
              template="simple_white")

    mean_review_on_delay=orders_and_order_reviews.groupby('delay_or_no_delay')['review_score'].mean()

    dash4_fig5=empty
    if not mean_review_on_delay.empty:
        dash4_fig5=px.bar(x=mean_review_on_delay.index, 
              y=mean_review_on_delay.values, 
              title="Effect of Delivery Delay on Review (Average Review Score by Order Delayed/Not Delayed)",
              labels={"x": "Order Delayed/Not Delayed", "y": "Average Review Score"},
              color_discrete_sequence=px.colors.qualitative.Pastel, 
              template="simple_white")


    left,right=st.columns(2)
    left.plotly_chart(dash4_fig1,use_container_width=True)
    right.plotly_chart(dash4_fig2,use_container_width=True)
    st.plotly_chart(dash4_fig4,use_container_width=True)
    left1,right1=st.columns(2)
    left1.plotly_chart(dash4_fig3,use_container_width=True)
    right1.plotly_chart(dash4_fig5,use_container_width=True)

    
    

def sideBar():
    with st.sidebar:
        selected=option_menu(
            # orientation="horizontal",
            menu_title="Main Menu",
            options=["Preferance","Trend","Geo","Review","Recommend","RFM","Churn"],
            icons=["clipboard-data","graph-up-arrow","globe","hand-thumbs-up-fill","gift","cash-coin","box-arrow-left"],
            menu_icon="cast",
            default_index=0
        )
    if selected=="Trend":
        dash_board_Trend()
    elif selected=="Churn":
        dash_board_Churn()
    elif selected=="Preferance":
        dash_board_Preferance()
    elif selected=="Geo":
        dash_board_Geo()
    elif selected=="Recommend":
        dash_board_Recommend()
    elif selected=="RFM":
        dash_board_RFM()
    elif selected=="Review":
        dash_board_Review()



sideBar()
# rerun()


#theme
hide_st_style=""" 

<style>
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}
</style>
"""

# st.session_state['time_reload_data']=


# get_data - KafkaConsumer
# data - define and divide / append the data to state
# caller_fun = use session state as df 

# rerun but with not data load


def explain():
    st.write("""
### 1. Segmentation Understanding:

#### Regular Customers (Segment: Regular):
   - **Insight:** Moderate recency, frequency, and monetary value.
   - **Strategy:** Implement personalized promotions for related products, maintaining engagement and increasing average order value.

#### Inactive Customers (Segment: Inactive):
   - **Insight:** Low recency, frequency, and monetary value.
   - **Strategy:** Launch reactivation campaigns with special offers and targeted messaging to reignite interest.

### 2. Churn Analysis Insights:

   - **Churned Customers:**
      - **Insight:** Churn associated with late deliveries and review scores.
      - **Strategy:** Address late deliveries promptly, gather feedback, and showcase resolutions to improve customer satisfaction.

### 3. Recency-Frequency-Monetary (RFM) Score Utilization:

   - **Recency Score:**
      - **Insight:** Win-back potential in higher recency scores.
      - **Strategy:** Launch time-sensitive promotions for recent customers, emphasizing exclusive benefits.

   - **Frequency Score:**
      - **Insight:** Low engagement in customers with low frequency scores.
      - **Strategy:** Introduce a loyalty program with tiered benefits to increase engagement.

   - **Monetary Score:**
      - **Insight:** Create VIP programs for high monetary score customers.
      - **Strategy:** Design premium VIP programs with exclusive benefits, aiming to increase the monetary value of moderate-scoring customers.

### 4. RFM Score (Overall) Strategy:

   - **High RFM Score (5-7):**
      - **Insight:** Indicates highly engaged and valuable customers.
      - **Strategy:** Implement an exclusive loyalty program, offering personalized perks, early access to promotions, and special events to maintain their loyalty.

   - **Moderate RFM Score (3-4):**
      - **Insight:** Indicates room for improvement and potential for increased engagement.
      - **Strategy:** Launch targeted promotions and personalized communication to elevate their RFM scores, encouraging more frequent and higher-value transactions.

   - **Low RFM Score (0-2):**
      - **Insight:** Indicates less engaged and lower-value customers.
      - **Strategy:** Implement win-back campaigns, offering special incentives to encourage repeat purchases and regain their interest.

### 5. Customer Experience Enhancement:

   - **Late Deliveries:**
      - **Insight:** Optimize the supply chain to reduce late deliveries.
      - **Strategy:** Implement real-time tracking and proactive communication for better customer satisfaction.

   - **Review Scores:**
      - **Insight:** Positive reviews correlate with customer retention.
      - **Strategy:** Encourage customers to leave positive reviews and address negative ones promptly.

### 6. Communication Strategies:

   - **Personalized Messaging:**
      - **Insight:** Tailor communications based on customer segments and RFM scores.
      - **Strategy:** Use dynamic content in emails showcasing relevant products and promotions based on each customer's past behavior.

   - **Reactivation Campaigns:**
      - **Insight:** Inactive customers show potential for reactivation.
      - **Strategy:** Employ multi-channel communication with personalized incentives to win back inactive customers.

### 7. Retention KPIs and Measurement:

   - **Key Performance Indicators (KPIs):**
      - **Insight:** Monitor customer retention rate, repeat purchase rate, and customer lifetime value.
      - **Strategy:** Regularly track and analyze KPIs to measure the success of campaigns, adjusting strategies based on performance metrics.

   - **Feedback and Surveys:**
      - **Insight:** Customer feedback provides valuable insights.
      - **Strategy:** Conduct regular surveys to gather feedback and make data-driven improvements.

### 8. Continuous Improvement:

   - **Data Analysis:**
      - **Insight:** Regularly analyze customer behavior data for trends.
      - **Strategy:** Adapt strategies based on evolving customer preferences and market dynamics.

   - **Adaptability:**
      - **Insight:** Stay informed about industry changes.
      - **Strategy:** Adapt the retention strategy to align with emerging market trends, maintaining a competitive edge.

### 9. Employee Training and Customer Service:

   - **Training Programs:**
      - **Insight:** Well-trained staff contributes to positive interactions.
      - **Strategy:** Provide ongoing training to customer service teams, empowering them to handle concerns proactively.

   - **Customer Service Excellence:**
      - **Insight:** Customer service excellence positively impacts brand perception.
      - **Strategy:** Implement a customer-centric culture within the organization to ensure excellence in every customer interaction.

### Conclusion:

This retention strategy, enriched with specific insights from RFM and churn analyses, includes a tailored approach for customers based on their overall RFM scores. The goal is to maximize engagement, increase loyalty, and address specific needs of each customer segment. Regular evaluation and adaptability are crucial to sustaining a successful retention strategy in the dynamic landscape of customer preferences and market trends.

    """)


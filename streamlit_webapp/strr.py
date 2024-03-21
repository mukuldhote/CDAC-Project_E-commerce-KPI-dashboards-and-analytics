import streamlit as st
import pandas as pd
import plotly.express as px
from streamlit_option_menu import option_menu
from numerize.numerize import numerize
import time

# from streamlit_extras.metric_cards import style_metric_cards
# st.set_option('deprecation.showPyplotGlobalUse', False)
import plotly.graph_objs as go
from kafka import KafkaConsumer
import json
import time
from datetime import datetime

#uncomment this line if you use mysql
#from query import *
st.set_page_config(page_title="Dashboard",page_icon="🌍",layout="wide")

theme_plotly = None 

@st.experimental_memo(ttl=20, max_entries=3000,show_spinner=False)# <--- to edit after full integrate
def get_data(mode=1):
    consumer = KafkaConsumer(
        'newer',  # Replace with your Kafka topic <--- edit the name
        bootstrap_servers=['localhost:9092'],  # Replace with your Kafka server
        value_deserializer=lambda x: json.loads(x.decode('utf-8'))
    )

    messages = []

    # Consume messages for 10 seconds
    end_time = time.time() + 1
    for message in consumer:
        messages.append(message.value)
        if time.time() > end_time:
            break
    # Create DataFrame from consumed messages
    df = pd.DataFrame(messages)
    if mode == 1:
        return df
    elif mode ==2:
        return df
    else:
        return df


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

def dash_board_Ratings():
    pass

def dash_board_Geo_loc():
   Geo(get_data())

def dash_board_Categories():
   pass

def dash_board_CRM():
   pass
   
def dash_board_Time():
   pass

def dash_board_Home():
   pass


def delay_indicate(delay_value):
    if delay_value<0:
        return "Delay"
    else:
        return "No Delay"

def Geo(df):
    seller_state=st.sidebar.multiselect(
        "SELECT seller_state",
        options=df["seller_state"].unique(),
        default=df["seller_state"].unique()
    )
    customer_state=st.sidebar.multiselect(
        "SELECT customer_state",
        options=df["customer_state"].unique(),
        default=df["customer_state"].unique()
    )
    df_filter=df.query(
        "seller_state==@seller_state & customer_state==@customer_state"
    )
    df_filter=df_filter[df_filter['order_status']=='delivered']

    df_filter['order_purchase_timestamp'] = pd.to_datetime(df_filter['order_purchase_timestamp'])
    df_filter['order_month_year'] = df_filter['order_purchase_timestamp'].dt.to_period('M')

    # Incomplete data post 2018-08
    df_filter=df_filter[df_filter['order_month_year']<'2018-09']
    df_filter['weekday']=df_filter['order_purchase_timestamp'].dt.day_name()
    df_filter['order_purchase_hour']=df_filter['order_purchase_timestamp'].dt.hour
    df_filter['order_estimated_delivery_date']=pd.to_datetime(df_filter['order_estimated_delivery_date'])
    df_filter['order_delivered_customer_date']=pd.to_datetime(df_filter['order_delivered_customer_date'])
    df_filter['difference_purchased_delivered']=(df_filter['order_delivered_customer_date']-df_filter['order_purchase_timestamp']).dt.days
    df_filter['estimated_delivered_difference']=(df_filter['order_estimated_delivery_date']-df_filter['order_delivered_customer_date']).dt.days
    df_filter['delay_or_no_delay']=df_filter['estimated_delivered_difference'].apply(delay_indicate)
    df_filter['review_creation_date']=pd.to_datetime(df_filter['review_creation_date'])
    df_filter['review_answer_timestamp']=pd.to_datetime(df_filter['review_answer_timestamp'])
    df_filter['review_answer_time_difference']=(df_filter['review_answer_timestamp'] - df_filter['review_creation_date']).dt.days
    df_filter['review_answer_time_difference_label']=df_filter['review_answer_time_difference'].apply(answer_diff)


    freight_value = float(pd.Series(df['freight_value']).mean())
    payment_value = float(pd.Series(df['payment_value']).sum())
    review_score = float(pd.Series(df['review_score']).mean())
    Tfreight_value= float(pd.Series(df['freight_value']).sum())

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
    total1,total2,total3,total4=st.columns(4)
    with total1:
        st.metric(label="Avg Freight.val",value=f"{numerize(freight_value)}",delta=-100,delta_color='normal'  
                #   help="Help"
                  )

    with total2:
        st.metric(label="Total Payment.val",value=f"{numerize(payment_value)}",delta=100,delta_color='inverse'
                #    help="HELP"
                  )

    with total3:
        st.metric(label="Average Review.score",value=f"{review_score:,.0f}",delta=300,delta_color='inverse' 
                #   help="HELP"
                  )

    with total4:
        st.metric(label="Total Freight.val",value=f"{Tfreight_value:,.0f}",delta=-50,delta_color='inverse'
                #   help="HELP"
                  )

    #---------------- graphs
    state_number_of_orders_map=df_filter.groupby('customer_state')['order_id'].nunique().reset_index(name='number_of_orders_for_state')

    dash3_fig3=empty
    if not state_number_of_orders_map.empty:
        dash3_fig3=px.treemap(state_number_of_orders_map,path=['customer_state'],values='number_of_orders_for_state',color_discrete_sequence=px.colors.qualitative.Pastel)
        dash3_fig3.update_layout(title_text="Order Concentration by States (States by Number of Orders)",
                                template="simple_white")



    city_delivery_days_mean=df_filter.groupby('customer_city')['difference_purchased_delivered'].mean().sort_values()
    city_delivery_days_mean_lowest_5=city_delivery_days_mean.head(5)

    dash3_fig4=empty
    if city_delivery_days_mean_lowest_5.shape[0] > 0:
        dash3_fig4=px.bar(x=city_delivery_days_mean_lowest_5.index, 
                    y=city_delivery_days_mean_lowest_5.values, 
                    title="Top 5 Cities with Lowest Average Delivery Time",
                    labels={"x": "City", "y": "Average Number of Days to Deliver"},
                    color_discrete_sequence=px.colors.qualitative.Pastel, 
                    template="simple_white")
    
    cities_mean_delay=df_filter.groupby('customer_city')['estimated_delivered_difference'].mean().sort_values()
    cities_mean_delay_top5_delay=cities_mean_delay.head(5)

    dash3_fig5=empty
    if not city_delivery_days_mean_lowest_5.empty:
        dash3_fig5=px.bar(x=cities_mean_delay_top5_delay.index, 
                y=cities_mean_delay_top5_delay.values, 
                title="Top 5 Cities with Highest Average Delivery Delay compared to Estimated Delivery Date",
                labels={"x": "City", "y": "Average delivery delay (in days)"},
                color_discrete_sequence=px.colors.qualitative.Pastel, 
                template="simple_white")


    left,right=st.columns(2)
    left.plotly_chart(dash3_fig4,use_container_width=True)
    right.plotly_chart(dash3_fig5,use_container_width=True)
    st.plotly_chart(dash3_fig3,use_container_width=True)



def sideBar():
    with st.sidebar:
        selected=option_menu(
            orientation="horizontal",
            menu_title="Main Menu",
            options=["Home","Ratings","Geo-loc","Categories","CRM","Time"],
            icons=["house","clipboard-data","globe","house","people","graph-up-arrow"],
            menu_icon="cast",
            default_index=0
        )
    if selected=="Time":
        dash_board_Time()
    elif selected=="Ratings":
        dash_board_Ratings()
    elif selected=="CRM":
        dash_board_CRM()
    elif selected=="Categories":
        dash_board_Categories()
    elif selected=="Geo-loc":
        dash_board_Geo_loc()
    elif selected=="Home":
        dash_board_Home()
    

# if st.sessionstate[''] sideBar()
Geo(get_data())
# time.sleep(20)

# end_time = time.time() + 19
# while time.time() < end_time:
#     time.sleep(1)  # Adjust the sleep time as needed
# else:
#     st.experimental_rerun()

end_time = time.time() + 20
while time.time() < end_time:
    # Update specific parts of your app here
    time.sleep(20)
    st.experimental_rerun()
    # Add a slight sleep to avoid too frequent refreshes
    # time.sleep(0.1)  # Adjust the sleep time as needed





#df=get_data()

#def main():
#    st.title("Streamlit Kafka Consumer")

    #data_stream = get_data()

    #for data in data_stream:
        # Update your Streamlit UI with the received data
#    while True:
#    	st.write("Received data:", pd.read_json(get_data()))

#if __name__ == "__main__":
#    main()

#while True:
#    get_data()
#    st.table(df)
#    time.sleep(6)

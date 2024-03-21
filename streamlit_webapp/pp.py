import asyncio
import streamlit as st

async def periodic():
    while True:
        st.write("Hello world")
        try:
            await asyncio.sleep(1)  # Use `try-except` to handle potential errors
        except asyncio.CancelledError:
            break  # Exit the loop gracefully if cancelled

async def main():
    async with st.container():  # Use `st.container` to manage layout
        task = asyncio.create_task(periodic())

        # Create a mechanism to stop the task
        stop_button = st.button("Stop")
        if stop_button:
            task.cancel()
            try:
                await task  # Wait for task to finish gracefully
            except asyncio.CancelledError:
                pass

asyncio.run(main())  # Run the `main` coroutine


number_cities_serviced
avg_days_to_deliver
on_time_perc

total_customers
revenue_per_customer
avg_num_of_order_per_customer


number_of_product_categories
highest_rev_product_cat_perc_top_prod
highest_rev_product_cat_perc_top_revenue


metric_count_reviewed_orders_perc


column=['order_id',
    'order_item_id',
    'product_id',
    'seller_id',
    'shipping_limit_date',
    'price',
    'freight_value','order_id',
    'payment_sequential',
    'payment_type',
    'payment_installments',
    'payment_value','review_id',
    'order_id',
    'review_score',
    'review_creation_date',
    'review_answer_timestamp',
    'review_answer_time_difference',
    'review_answer_time_difference_label','order_id',
    'customer_id',
    'order_status',
    'order_purchase_timestamp',
    'order_approved_at',
    'order_delivered_carrier_date',
    'order_delivered_customer_date',
    'order_estimated_delivery_date',
    'order_month_year',
    'weekday',
    'order_purchase_hour',
    'difference_purchased_delivered',
    'estimated_delivered_difference',
    'delay_or_no_delay','customer_id',
    'customer_unique_id',
    'customer_zip_code_prefix',
    'customer_city',
    'customer_state','seller_id',
    'seller_zip_code_prefix', 
    'seller_city', 
    'seller_state','product_id',
    'product_category_name_english',
    'product_name_lenght',
    'product_description_lenght',
    'product_photos_qty',
    'product_weight_g',
    'product_length_cm',
    'product_height_cm',
    'product_width_cm']

















import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Load datasets
orders = pd.read_csv('data/olist_orders_dataset.csv')
payments = pd.read_csv('data/olist_order_payments_dataset.csv')
reviews = pd.read_csv('data/olist_order_reviews_dataset.csv')

# Merge and preprocess
df = orders.merge(payments, on='order_id').merge(reviews, on='order_id')
df = df.dropna(subset=['order_delivered_customer_date', 'order_purchase_timestamp'])
df['order_delivered_customer_date'] = pd.to_datetime(df['order_delivered_customer_date'])
df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
df['delivery_days'] = (df['order_delivered_customer_date'] - df['order_purchase_timestamp']).dt.days

# Feature engineering
df['purchase_dayofweek'] = df['order_purchase_timestamp'].dt.dayofweek
df['purchase_hour'] = df['order_purchase_timestamp'].dt.hour

features = ['payment_value', 'purchase_dayofweek', 'purchase_hour']
X = df[features]
y = df['delivery_days']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Save model and features
with open('models/delivery_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('models/feature_columns.pkl', 'wb') as f:
    pickle.dump(features, f)

import pandas as pd
from dateutil import parser
import numpy as np
import re
from geopy.geocoders import Nominatim
import time

# Question 1

# Function to parse dates
def parse_date(date_str):
    try:
        # Use dateutil parser with dayfirst=True for mixed formats
        date_time = parser.parse(date_str, dayfirst=True)
        return date_time.date()
    except Exception:
        return pd.NaT

# Apply parsing function
# df['clean_date'] = df['order_date'].apply(parse_date)

# Convert to standardized format 'YYYY-MM-DD'
# df['clean_date'] = pd.to_datetime(df['clean_date']).dt.strftime('%Y-%m-%d')

# If desired, replace NaT formatted strings with None or leave as-is
# df.loc[df['clean_date'] == 'NaT', 'clean_date'] = None




# Question 2

def clean_price(value):
    if isinstance(value, (int, float)):  # Already numeric
        return value
    if isinstance(value, str):
        # Remove currency symbols and commas, keep digits and dot
        cleaned = re.sub(r'[^\d.]', '', value)
        try:
            return float(cleaned) if '.' in cleaned else int(cleaned)
        except ValueError:
            return np.nan
    return np.nan  # For other types (e.g., None)

# Example usage
# df['original_price_inr_cleaned'] = df['original_price_inr'].apply(clean_price)




# Question 3

def round_up_scale(value):
    """Round value up to nearest 5 or 10, whichever is appropriate."""
    if value <= 5:
        return 5
    elif value <= 10:
        return 10
    else:
        # For values above 10, round up to nearest multiple of 5
        return int(math.ceil(value / 5.0)) * 5

def standardize_rating(rating):
    if rating is None:
        return np.nan

    rating_str = str(rating).strip().lower()

    # Fraction parsing
    if '/' in rating_str:
        try:
            numerator, denominator = rating_str.split('/')
            numerator = float(numerator)
            denominator = float(denominator)
            if denominator == 0:
                return np.nan
            normalized = (numerator / denominator) * 5
            if 1.0 <= normalized <= 5.0:
                return round(normalized, 2)
            else:
                return np.nan
        except:
            return np.nan

    # Extract numeric value
    match = re.search(r'(\d+(\.\d+)?)', rating_str)
    if match:
        try:
            value = float(match.group(1))

            scale = round_up_scale(value)
            normalized = (value / scale) * 5

            # Clamp between 1 and 5
            normalized = max(1.0, min(normalized, 5.0))
            return round(normalized, 2)

        except:
            return np.nan

    return np.nan

# df["customer_rating_cleaned"] = df["customer_rating"].apply(standardize_rating)






# Question 4

# Sample city names (including variants and misspellings)
# cities = ['Bangalore', 'bengaluru', 'Mumbai', 'Bombay', 'delhi', 'New Delhi', 'mumbay', 'banaglore', 'DELHI']

# df = pd.DataFrame({'customer_city': cities})

# Initialize Nominatim geocoder with a user_agent
geolocator = Nominatim(user_agent="osm_city_standardizer")

def geocode_city_osm(city):
    try:
        # Limit search to India for better accuracy
        location = geolocator.geocode(f"{city}, India", language='en', exactly_one=True, addressdetails=True)
        if location and 'address' in location.raw:
            address = location.raw['address']
            # Extract city or town from the address details
            city_name = address.get('city') or address.get('town') or address.get('village')
            
            # Sometimes the city might be under 'state_district' or 'county' if city not found
            if not city_name:
                city_name = address.get('state_district') or address.get('county')
            
            if city_name:
                return city_name
            else:
                # Fallback: use first part of display_name
                return location.address.split(',')[0]
        else:
            return None
    except Exception as e:
        print(f"Error geocoding '{city}': {e}")
        return None

# Step 1: Get unique city names
unique_cities = df['customer_city'].unique()
# print(f"the len of unique cities: {len(unique_cities)}")

# Step 2: Build a mapping dictionary
city_mapping = {}

for city in unique_cities:
    print(f"input city is: {city}")
    standardized_city = geocode_city_osm(city)
    print(f"output city is: {standardized_city}")
    city_mapping[city] = standardized_city
    time.sleep(1)  # Pause to respect Nominatim's usage policy (1 sec per request)

# Step 3: Map standardized values back to the original DataFrame
# df['customer_city_standardized'] = df['customer_city'].map(city_mapping)

# to handle
"""
input city is: mumba
output city is: Navi Mumbai
input city is: chenai
output city is: None
input city is: Delhi NCR
output city is: Gurugram
"""




# Question 5

# Define a function to normalize boolean-like values
def normalize_boolean(val):
    if pd.isna(val):
        return False  # or pd.NA if you want to keep missing values
    if isinstance(val, str):
        val = val.strip().lower()
        return val in ['yes', 'y', 'true', '1']
    return bool(val)

# Apply normalization to each of the boolean columns
bool_cols = ['is_prime_member', 'is_prime_eligible', 'is_festival_sale']
new_cols = ['is_prime_member_cleaned', 'is_prime_eligible_cleaned', 'is_festival_sale_cleaned']

for index, col in enumerate(bool_cols):
    df[new_cols[index]] = df[col].apply(normalize_boolean)





# Question 6

df['Category_Standardized'] = 'Electronics'





# Question 7

def clean_day(value):
    value = value.strip().lower()

    # Special case: 'same day'
    if value == "same day":
        return 0

    # Check for two numbers separated by a dash (e.g., "1-2 days")
    range_match = re.findall(r"\d+(?:\.\d+)?", value)
    if '-' in value:
        if len(range_match) == 2:
            try:
                return max(float(range_match[0]), float(range_match[1]))
            except:
                return np.nan
        else:
            return np.nan

    # Check if it's a single valid number (possibly with 'days' word)
    if len(range_match) == 1:
        try:
            num = float(range_match[0])
            if num < 0:
                return np.nan
            return num
        except:
            return np.nan

    # If none of the above matched, it's invalid
    return np.nan






# Question 8

# Step 1: Find groups with potential duplicates
grouped = df.groupby(['customer_id', 'product_id', 'order_date', 'final_amount_inr'])

def analyze_group(group):
    unique_tx_ids = group['transaction_id'].nunique()
    total_records = len(group)
    
    if unique_tx_ids == total_records:
        # All transactions have unique IDs -> likely genuine bulk order
        return 'genuine_bulk_order'
    elif unique_tx_ids < total_records:
        # Some transaction_ids repeat -> data error duplicates
        return 'data_error_duplicates'
    else:
        return 'ambiguous'

# Apply the function to each group and create a Series with the result repeated for each row in the group
duplicate_types = grouped.apply(analyze_group).reset_index(name='duplicate_type')

# Map the group keys back to the original dataframe by merging or joining
df = df.merge(duplicate_types.reset_index(), on=['customer_id', 'product_id', 'order_date', 'final_amount_inr'], how='left')







# Question 9

def correct_prices(group):
    median_price = group['original_price_inr_cleaned'].median()
    threshold = 3 * median_price
    
    # Create new column 'corrected_price' starting as copy of original 'price'
    group['corrected_price'] = group['original_price_inr_cleaned']
    
    outliers = group['original_price_inr_cleaned'] > threshold
    correction_candidates = (group['original_price_inr_cleaned'] / 100) < threshold
    
    # Only update 'corrected_price' column; original 'price' unchanged
    group.loc[outliers & correction_candidates, 'corrected_price'] = group.loc[outliers & correction_candidates, 'original_price_inr_cleaned'] / 100
    
    return group

# df = df.groupby('product_id').apply(correct_prices).reset_index(drop=True)

# Apply function per product
df = df.groupby('product_id').apply(correct_prices).reset_index(drop=True)






# Question 10

def standardize_payment_method(payment_method: str) -> str:
    payment_method = str(payment_method).lower()  # Convert to string and lowercase

    payment_map = {
        'UPI': ['upi', 'phonepe', 'googlepay'],
        'Credit Card': ['credit card', 'credit_card', 'cc'],
        'Debit Card': ['debit card', 'debit_card', 'dc'],
        'Cash on Delivery': ['cash on delivery', 'cod', 'c.o.d'],
        # Add more mappings if needed
    }

    for category, keywords in payment_map.items():
        if any(keyword in payment_method for keyword in keywords):
            return category
    return 'Other'

# Apply the function to the payment_method column
df['standard_payment_method'] = df['payment_method'].apply(standardize_payment_method)
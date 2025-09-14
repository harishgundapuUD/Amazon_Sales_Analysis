import pandas as pd
from dateutil import parser
import numpy as np
import re
from geopy.geocoders import Nominatim
import time
import math


class DataCleaning:
    def __init__(self):
        self.city_names_corrections = {
            "chenai": "chennai",
            "delhi ncr": "delhi",
            "navi mumbai": "mumbai",
            "new delhi": "delhi",
            "delhi ncr": "delhi"
        }
        self.payment_map = {
            'UPI': ['upi', 'phonepe', 'googlepay'],
            'Credit Card': ['credit card', 'credit_card', 'cc'],
            'Debit Card': ['debit card', 'debit_card', 'dc'],
            'Cash on Delivery': ['cash on delivery', 'cod', 'c.o.d'],
            "Net Banking": ['net banking', 'net_banking', 'nb'],
            "Wallet": ['wallet'],
            "Buy Now Pay Later": ['buy now pay later', 'bnpl'],
            # Add more mappings if needed
        }
        self.max_delivery_days = 15  # Set a maximum threshold for delivery days

    # Question 1: Date Cleaning
    def date_cleaning(self, date_str):
        try:
            # Use dateutil parser with dayfirst=True for mixed formats
            date_time = parser.parse(date_str, dayfirst=True)
            return date_time.date()
        except Exception:
            return None
    
    # Question 2: Price Cleaning
    def price_cleaning(self, value):
        if isinstance(value, (int, float)):  # Already numeric
            return value
        if isinstance(value, str):
            # Remove currency symbols and commas, keep digits and dot
            cleaned = re.sub(r'[^\d.]', '', value)
            try:
                return float(cleaned) if '.' in cleaned else int(cleaned)
            except ValueError:
                return np.nan
        return np.nan # For other types (e.g., None)
    
    # Question 3: Rating Standardization
    def round_up_scale(self, value):
        """Round value up to nearest 5 or 10, whichever is appropriate."""
        if value <= 5:
            return 5
        elif value <= 10:
            return 10
        else:
            # For values above 10, round up to nearest multiple of 5
            return int(math.ceil(value / 5.0)) * 5
        
    def standardize_rating(self, rating):
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

                scale = self.round_up_scale(value)
                normalized = (value / scale) * 5

                # Clamp between 1 and 5
                normalized = max(1.0, min(normalized, 5.0))
                return round(normalized, 2)

            except:
                return np.nan
        return np.nan
    
    # Question 4: City Name Standardization
    def geocode_city_osm(self, city, geolocator):
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
    
    def city_standardization(self, unique_cities):
        geolocator = Nominatim(user_agent="osm_city_standardizer")

        # Step 1: Get unique city names
        for i in range(len(unique_cities)):
            if unique_cities[i].lower() in self.city_names_corrections:
                unique_cities[i] = self.city_names_corrections[unique_cities[i].lower()]

        # Step 2: Build a mapping dictionary
        city_mapping = {}

        for city in unique_cities:
            print(f"input city is: {city}")
            standardized_city = self.geocode_city_osm(city=city, geolocator=geolocator)
            print(f"output city is: {standardized_city}")
            city_mapping[city] = standardized_city
            time.sleep(1)  # Pause to respect Nominatim's usage policy (1 sec per request)
        
        for key, value in city_mapping.items():
            city_mapping[key] = self.city_names_corrections.get(value.lower(), value).title()
            print(f"after correction city is: {key, city_mapping[key]}")
        return city_mapping
    
    # Question 5: Boolean Normalization
    def normalize_boolean(self, val):
        if pd.isna(val):
            return False  # or pd.NA if you want to keep missing values
        if isinstance(val, str):
            val = val.strip().lower()
            if val in ['yes', 'y', 'true', '1']:
                return True
            elif val in ['no', 'n', 'false', '0']:
                return False
            else:
                return pd.NA  # unrecognized string
        return bool(val)
    
    # Question 6: Category Standardization
    def standardize_category(self, categories):
        output = {}
        for i in categories:
            output[i] = "Electronics" if "electronic" in i.lower() else i
        return output
    
    # Question 7: Delivery Days Cleaning    
    def clean_delivery_day(self, value, max_valid_days=None):
        """
        Cleans delivery day input and returns a numeric value representing delivery days.
        Returns np.nan for invalid or unrealistic values.
        """
        if not max_valid_days:
            max_valid_days = self.max_delivery_days
        if not isinstance(value, str):
            return np.nan

        value = value.strip().lower()

        # Special case: 'same day'
        if value == "same day":
            return 0

        # Extract all numeric parts (integers or decimals)
        range_match = re.findall(r"\d+(?:\.\d+)?", value)

        if '-' in value or 'to' in value:
            # Handle range like "1-2 days" or "1 to 3 business days"
            if len(range_match) == 2:
                try:
                    high = max(float(range_match[0]), float(range_match[1]))
                    return high if high <= max_valid_days else np.nan
                except:
                    return np.nan
            else:
                return np.nan

        # Single number case (e.g., '2 days', '3 business days')
        if len(range_match) == 1:
            try:
                num = float(range_match[0])
                if num < 0 or num > max_valid_days:
                    return np.nan
                return num
            except:
                return np.nan

        # Unrecognized format
        return np.nan
    
    # Question 8: Duplicate Transaction Identification
    def analyze_group(self, group):
        unique_tx_ids = group['transaction_id'].nunique()
        total_records = len(group)
        
        # All unique transaction IDs
        if unique_tx_ids == total_records:
            return 'genuine_bulk_order'
        
        # Repeated transaction IDs with identical other features
        tx_id_counts = group['transaction_id'].value_counts()
        repeated_tx_ids = tx_id_counts[tx_id_counts > 1].index

        for tx_id in repeated_tx_ids:
            repeated_tx_group = group[group['transaction_id'] == tx_id]
            # Check if all rows for this transaction ID are exactly the same
            if repeated_tx_group.drop(columns='transaction_id').duplicated().all():
                return 'data_error_duplicates'
        
        # Could be mix of genuine and error
        return 'ambiguous'
    
    # Question 9: Price Correction
    def mad_based_threshold(self, group, column_name, threshold_multiplier=3):
        # median absolute deviation method
        median_price = group[column_name].median()
        mad = (group[column_name] - median_price).abs().median()
        threshold = median_price + threshold_multiplier * mad
        return threshold


    def correct_prices(self, group, threshold_multiplier=3):
        # median_price = group['clean_original_price_inr'].median()
        # threshold = threshold_multiplier * median_price
        threshold = self.mad_based_threshold(group, column_name='clean_original_price_inr', threshold_multiplier=threshold_multiplier)
        
        # Start corrected_price as a copy to avoid SettingWithCopyWarning
        group = group.copy()
        group['corrected_price'] = group['clean_original_price_inr']
        
        # Identify outliers beyond threshold
        outliers = group['clean_original_price_inr'] > threshold
        
        # Check if dividing by 100 brings price below threshold
        correction_candidates = (group['clean_original_price_inr'] / 100) < threshold
        
        # Apply correction where both conditions hold
        condition = outliers & correction_candidates
        group.loc[condition, 'corrected_price'] = group.loc[condition, 'clean_original_price_inr'] / 100

        # Optional: print or log number of corrections made
        num_corrected = condition.sum()
        # if num_corrected > 0:
        #     print(f"Corrected {num_corrected} prices in product_id {group.name}")
        
        return group
    
    # Question 10: Payment Method Standardization
    def standardize_payment_method(self, payment_method: str) -> str:
        payment_method = str(payment_method).lower()  # Convert to string and lowercase

        for category, keywords in self.payment_map.items():
            if any(keyword in payment_method for keyword in keywords):
                return category
        return 'Other'

# [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]
years = [2025]
for i in years:
    input_path = fr"C:\Users\haris\OneDrive\Desktop\Guvi\Projects\Amazon_Sales_Analysis\dataset\amazon_india_{i}.csv"
    output_path = fr"C:\Users\haris\OneDrive\Desktop\Guvi\Projects\Amazon_Sales_Analysis\dataset\amazon_india_{i}_cleaned.csv"
    df = pd.read_csv(input_path)
    data_cleaner = DataCleaning()

    # Question 1
    df["clean_order_date"] = df["order_date"].apply(data_cleaner.date_cleaning)

    # Question 2
    columns = ["original_price_inr", "discount_percent", "final_amount_inr", "delivery_charges"]
    df[["clean_original_price_inr", "clean_discount_percent", "clean_final_amount_inr", "clean_delivery_charges"]] = df[columns].map(data_cleaner.price_cleaning)

    # Question 3
    columns = ['customer_rating', 'product_rating']
    df[["cleaned_customer_rating", "cleaned_product_rating"]] = df[columns].map(data_cleaner.standardize_rating)

    # Question 4
    cleaned_citys = data_cleaner.city_standardization(df['customer_city'].astype(str).str.lower().unique())
    df["cleaned_customer_city"] = df["customer_city"].astype(str).str.lower().map(cleaned_citys)

    # Question 5
    columns = ['is_prime_member', 'is_prime_eligible', 'is_festival_sale']
    df[['cleaned_is_prime_member', 'cleaned_is_prime_eligible', 'cleaned_is_festival_sale']] = df[columns].map(data_cleaner.normalize_boolean)

    # Question 6
    category_map = data_cleaner.standardize_category(df["category"].astype(str).unique())
    df["cleaned_category"] = df["category"].map(category_map)

    # Question 7
    df["cleaned_delivery_days"] = df["delivery_days"].apply(data_cleaner.clean_delivery_day)

    # Question 8
    grouped = df.groupby(['customer_id', 'product_id', 'order_date', 'final_amount_inr'])
    # Apply the function to each group and create a Series with the result repeated for each row in the group
    duplicate_types = grouped.apply(data_cleaner.analyze_group, include_groups=False).reset_index(name='duplicate_type')

    # Map the group keys back to the original dataframe by merging or joining
    df = df.merge(duplicate_types.reset_index(), on=['customer_id', 'product_id', 'order_date', 'final_amount_inr'], how='left')
    if 'index' in df.columns:
        df.drop(columns='index', inplace=True)

    # Question 9
    df = df.groupby('product_id').apply(data_cleaner.correct_prices).reset_index(drop=True)

    # Question 10
    df['standard_payment_method'] = df['payment_method'].apply(data_cleaner.standardize_payment_method)

    # save the output to a csv file
    df.to_csv(output_path, index=False)
    print(f"cleaned data saved to {output_path}")
    print("\n\n")
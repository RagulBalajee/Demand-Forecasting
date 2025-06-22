# Sample Data Directory

This directory contains sample data files for testing the retail demand forecasting system.

## Data Format

### Sales Data (sales_sample.csv)
The sales data should contain the following columns:
- `Date`: Date of the sale (YYYY-MM-DD format)
- `Store`: Store ID (integer)
- `Product`: Product ID (integer)
- `Sales`: Number of units sold (integer)
- `Customers`: Number of customers (integer, optional)
- `Promo`: Whether promotion was active (0 or 1, optional)
- `SchoolHoliday`: Whether it was a school holiday (0 or 1, optional)
- `StateHoliday`: Whether it was a state holiday (0 or 1, optional)

### Store Data (stores_sample.csv)
The store data should contain the following columns:
- `Store`: Store ID (integer)
- `StoreType`: Type of store (a, b, c, d)
- `Assortment`: Assortment type (a, b, c)
- `CompetitionDistance`: Distance to nearest competitor (float)
- `CompetitionOpenSinceMonth`: Month when competition opened (integer)
- `CompetitionOpenSinceYear`: Year when competition opened (integer)
- `Promo2`: Whether Promo2 is active (0 or 1)
- `Promo2SinceWeek`: Week when Promo2 started (integer)
- `Promo2SinceYear`: Year when Promo2 started (integer)
- `PromoInterval`: Promo2 intervals (string)

## Usage

1. The system will automatically generate sample data when you run the demo or web application
2. You can also upload your own CSV files in the same format
3. Make sure your data follows the expected column names and data types

## Example Data Generation

The system includes a `DataLoader` class that can generate realistic sample data similar to the Rossmann Store Sales dataset. This includes:
- Seasonal patterns
- Weekend effects
- Store-specific variations
- Product-specific trends
- Promotional effects
- Holiday impacts 
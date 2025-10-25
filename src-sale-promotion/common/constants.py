"""
Constants specific to sales promotion documents.
"""

# Campaign Types
CAMPAIGN_TYPES = [
    "email",
    "social_media",
    "display_ads",
    "search_ads",
    "content_marketing",
    "influencer",
    "affiliate",
    "direct_mail",
    "events",
    "partnerships"
]

# Product Categories
PRODUCT_CATEGORIES = [
    "electronics",
    "fashion",
    "home_and_garden",
    "beauty_and_personal_care",
    "sports_and_outdoors",
    "toys_and_games",
    "books_and_media",
    "food_and_beverage",
    "health_and_wellness",
    "automotive"
]

# Seasons
SEASONS = ["spring", "summer", "fall", "winter", "holiday", "back_to_school"]

# Target Demographics
AGE_GROUPS = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
INCOME_LEVELS = ["<30k", "30k-50k", "50k-75k", "75k-100k", "100k-150k", "150k+"]

# Sales Channels
SALES_CHANNELS = [
    "online",
    "retail_store",
    "mobile_app",
    "marketplace",
    "social_commerce",
    "direct_sales",
    "wholesale"
]

# Promotion Types
PROMOTION_TYPES = [
    "percentage_discount",
    "fixed_amount_discount",
    "buy_one_get_one",
    "bundle_deal",
    "free_shipping",
    "loyalty_reward",
    "flash_sale",
    "clearance",
    "seasonal_sale"
]

# Performance Metrics Fields
PERFORMANCE_FIELDS = [
    "impressions",
    "clicks",
    "ctr",
    "conversions",
    "conversion_rate",
    "revenue",
    "roi",
    "cost_per_click",
    "cost_per_acquisition",
    "engagement_rate",
    "bounce_rate",
    "average_order_value"
]

# Metadata Field Categories
METADATA_CATEGORIES = {
    "campaign": ["campaign_id", "campaign_name", "campaign_type", "start_date", "end_date", "status", "budget", "channels"],
    "product": ["product_id", "product_name", "category", "subcategory", "brand", "sku", "price", "discount_percentage"],
    "geographic": ["region", "country", "city", "store_location", "distribution_channel"],
    "demographic": ["target_age_group", "target_income", "target_gender", "customer_segment"],
    "performance": PERFORMANCE_FIELDS
}

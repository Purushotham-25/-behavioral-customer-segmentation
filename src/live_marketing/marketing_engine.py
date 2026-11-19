
import pandas as pd


MARKETING_RULES = [
    {
        "name": "High-value customer",
        "condition": lambda r: (
            r.get("Monetary", 0) >= 1000
            and r.get("Frequency", 0) >= 5
            and r.get("Recency", 9999) <= 60
        ),
        "campaign": "VIP Loyalty",
        "message": (
            "Thank you for being a valued customer! "
            "Enjoy exclusive benefits and early access to selected products."
        ),
        "action": "Offer loyalty rewards and complementary products.",
        "priority": "High",
    },
    {
        "name": "At-risk customer",
        "condition": lambda r: (
            r.get("Recency", 0) > 90
            and r.get("Frequency", 0) >= 2
        ),
        "campaign": "Customer Win-Back",
        "message": (
            "We miss you! Discover products selected based "
            "on your previous purchases."
        ),
        "action": "Send a personalized re-engagement offer.",
        "priority": "High",
    },
    {
        "name": "Repeat customer",
        "condition": lambda r: r.get("Frequency", 0) >= 2,
        "campaign": "Personalized Cross-Sell",
        "message": (
            "Discover products that complement your previous purchases."
        ),
        "action": "Recommend related products based on purchase history.",
        "priority": "Medium",
    },
]


def generate_recommendation(customer: dict | pd.Series) -> dict:
    """Generate a marketing recommendation for one customer."""

    customer = dict(customer)

    for rule in MARKETING_RULES:
        if rule["condition"](customer):
            return {
                "CampaignType": rule["campaign"],
                "MarketingMessage": rule["message"],
                "RecommendedAction": rule["action"],
                "MarketingPriority": rule["priority"],
                "MarketingRule": rule["name"],
            }

    return {
        "CampaignType": "New Customer Engagement",
        "MarketingMessage": (
            "Welcome! Explore products selected to help "
            "you discover items you may enjoy."
        ),
        "RecommendedAction": (
            "Show relevant products and encourage a follow-up purchase."
        ),
        "MarketingPriority": "Low",
        "MarketingRule": "Default engagement",
    }


def generate_marketing_recommendations(
    customers: pd.DataFrame,
) -> pd.DataFrame:
    """Generate recommendations for all customers."""

    if customers.empty:
        return customers.copy()

    recommendations = customers.apply(
        generate_recommendation,
        axis=1,
    )

    recommendation_df = pd.DataFrame(
        recommendations.tolist(),
        index=customers.index,
    )

    result = customers.copy()

    for column in recommendation_df.columns:
        result[column] = recommendation_df[column]

    return result

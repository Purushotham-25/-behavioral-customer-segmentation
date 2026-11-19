import unittest

from src.live_marketing.marketing_engine import (
    generate_recommendation,
)


class TestMarketingEngine(unittest.TestCase):

    def test_high_value_customer(self):
        customer = {
            "Recency": 5,
            "Frequency": 8,
            "Monetary": 2500,
        }

        result = generate_recommendation(customer)

        self.assertEqual(result["CampaignType"], "VIP Loyalty")
        self.assertEqual(result["MarketingPriority"], "High")

    def test_at_risk_customer(self):
        customer = {
            "Recency": 150,
            "Frequency": 2,
            "Monetary": 200,
        }

        result = generate_recommendation(customer)

        self.assertEqual(result["CampaignType"], "Customer Win-Back")

    def test_new_customer_default(self):
        customer = {
            "Recency": 0,
            "Frequency": 1,
            "Monetary": 50,
        }

        result = generate_recommendation(customer)

        self.assertEqual(
            result["CampaignType"],
            "New Customer Engagement",
        )


if __name__ == "__main__":
    unittest.main()
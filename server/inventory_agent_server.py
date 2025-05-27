import os
import sys
import pandas as pd
from mcp.server.fastmcp import FastMCP

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

mcp = FastMCP("InventoryAgent")

df = pd.read_csv("data/DataCoSupplyChainDataset.csv", encoding="ISO-8859-1")

@mcp.tool()
def low_stock_products(threshold: int = 10) -> list:
    """
    List products with stock below a threshold.
    """
    # Fake inventory: use quantity sold as proxy
    stock_df = df.groupby("Product Name")["Order Item Quantity"].sum().reset_index()
    stock_df.columns = ["Product", "QuantitySold"]
    low_stock = stock_df[stock_df["QuantitySold"] < threshold]
    return low_stock.to_dict(orient="records")

@mcp.tool()
def restock_suggestion(region: str) -> list:
    """
    Suggest top products to restock in a region.
    """
    regional_orders = df[df["Order Region"].str.lower() == region.lower()]
    top_products = (
        regional_orders["Product Name"]
        .value_counts()
        .head(5)
        .reset_index()
    )
    top_products.columns = ["Product", "TimesOrdered"]
    return top_products.to_dict(orient="records")

@mcp.tool()
def products_at_risk_of_stockout(min_orders: int = 5) -> list:
    """
    Identify products that are frequently ordered but have stock issues.

    Args:
        min_orders: Minimum number of orders in past month to be considered.

    Returns:
        List of product names with high demand and low availability.
    """
    # Use Product Status: 1 = Not Available, 0 = Available
    stock_status = df[df["Product Status"] == 1]
    
    recent_orders = df[
        pd.to_datetime(df["order date (DateOrders)"], errors="coerce") >= pd.Timestamp.now() - pd.Timedelta(days=30)
    ]

    high_demand = (
        recent_orders.groupby("Product Name")
        .size()
        .reset_index(name="recent_orders")
        .query("recent_orders >= @min_orders")
    )

    at_risk = stock_status[stock_status["Product Name"].isin(high_demand["Product Name"])]
    return at_risk["Product Name"].drop_duplicates().tolist()

@mcp.tool()
def demand_supply_gap(top_n: int = 5) -> list:
    """
    Show products with the biggest demand/supply mismatch.

    Returns:
        List of products sorted by descending demand-supply gap.
    """
    demand = df.groupby("Product Name")["Order Item Quantity"].sum()
    availability = (
        df[df["Product Status"] == 0]
        .groupby("Product Name")["Order Item Quantity"]
        .sum()
    )

    mismatch = (demand - availability).dropna().sort_values(ascending=False).head(top_n)
    
    return [{"Product": name, "Gap": int(gap)} for name, gap in mismatch.items()]

@mcp.tool()
def product_status_overview() -> dict:
    """
    Returns total available vs. unavailable products in inventory.
    """
    counts = df["Product Status"].value_counts().to_dict()
    return {
        "Available Products": counts.get(0, 0),
        "Unavailable Products": counts.get(1, 0)
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")

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

if __name__ == "__main__":
    mcp.run(transport="stdio")

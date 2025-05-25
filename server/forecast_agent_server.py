import os
import sys
import pandas as pd
from mcp.server.fastmcp import FastMCP

# So it can find your project modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Create a new MCP server
mcp = FastMCP("ForecastAgent")

# Load your dataset
df = pd.read_csv("data/DataCoSupplyChainDataset.csv", encoding="ISO-8859-1")
df["Order_Date"] = pd.to_datetime(df["order date (DateOrders)"], errors="coerce")

@mcp.tool()
def total_sales_by_region(region: str) -> str:
    filtered = df[df["Order Region"].str.lower() == region.lower()]
    total = filtered["Sales"].sum()
    return f"Total sales in {region}: ${total:,.2f}"

# Define a tool that forecasts demand
@mcp.tool()
def forecast_demand(regions: list) -> dict:
    filtered = df[df["Order Region"].isin(regions)]
    if filtered.empty:
        return f"No data found for regions: {regions}"
    forecast = filtered.groupby("Order Region")["Sales"].sum().sort_values(ascending=False)
    return forecast.round(2).to_dict()


# Run as an MCP stdio server
if __name__ == "__main__":
    mcp.run(transport="stdio")

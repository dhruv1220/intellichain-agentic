import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mcp.server.fastmcp import FastMCP
import pandas as pd

mcp = FastMCP("SupplyChainServer")

# Load data
df = pd.read_csv("data/DataCoSupplyChainDataset.csv", encoding="ISO-8859-1")
df['Order_Date'] = pd.to_datetime(df['order date (DateOrders)'], errors='coerce')
df['Ship_Date'] = pd.to_datetime(df['shipping date (DateOrders)'], errors='coerce')
df['Scheduled_Ship_Date'] = df['Order_Date'] + pd.to_timedelta(df['Days for shipment (scheduled)'], unit='D')
df['Delivery_Delay_Days'] = (df['Ship_Date'] - df['Scheduled_Ship_Date']).dt.days

# Tool 1: Delay stats
@mcp.tool()
def get_delay_stats() -> dict:
    return df['Delivery_Delay_Days'].describe().to_dict()

# Tool 2: Orders by region
@mcp.tool()
def query_orders_by_region(region: str) -> list:
    results = df[df['Order Region'].str.lower() == region.lower()]
    return results[['Order Id', 'Order Region', 'Sales', 'Shipping Mode']].head(5).to_dict(orient='records')

if __name__ == "__main__":
    mcp.run(transport="stdio")

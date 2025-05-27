import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mcp.server.fastmcp import FastMCP
import pandas as pd

mcp = FastMCP("SupplyChainServer")

df = pd.read_csv("data/DataCoSupplyChainDataset.csv", encoding="ISO-8859-1")
df['Order_Date'] = pd.to_datetime(df['order date (DateOrders)'], errors='coerce')
df['Ship_Date'] = pd.to_datetime(df['shipping date (DateOrders)'], errors='coerce')
df['Scheduled_Ship_Date'] = df['Order_Date'] + pd.to_timedelta(df['Days for shipment (scheduled)'], unit='D')
df['Delivery_Delay_Days'] = (df['Ship_Date'] - df['Scheduled_Ship_Date']).dt.days

@mcp.tool()
def get_delay_stats() -> dict:
    return df['Delivery_Delay_Days'].describe().to_dict()

@mcp.tool()
def query_orders_by_region(region: str) -> list:
    filtered = df[df['Order Region'].str.lower() == region.lower()]
    return filtered[['Order Id', 'Order Region', 'Sales', 'Shipping Mode']].head(5).to_dict(orient='records')

@mcp.tool()
def get_shipping_mode_breakdown() -> dict:
    return df['Shipping Mode'].value_counts().to_dict()

@mcp.tool()
def top_delayed_products(n: int = 5) -> list:
    grouped = (
        df.groupby('Product Name')['Delivery_Delay_Days']
        .mean()
        .sort_values(ascending=False)
        .head(n)
    )
    return grouped.to_dict()

@mcp.tool()
def avg_delay_by_shipping_mode() -> dict:
    return df.groupby("Shipping Mode")["Delivery_Delay_Days"].mean().round(2).to_dict()

@mcp.tool()
def recommend_shipping_method(region: str) -> str:
    """
    Based on average delivery delays per shipping mode in `region`,
    recommend the fastest / most reliable mode.
    """
    df_region = df[df["Order Region"].str.lower()==region.lower()]
    if df_region.empty:
        return f"No data for region: {region}"

    # compute average delay by shipping mode
    stats = (
        df_region
        .groupby("Shipping Mode")["Delivery_Delay_Days"]
        .mean()
        .sort_values()
    )
    best_mode = stats.index[0]
    avg_delay = stats.iloc[0]
    return (
        f"In {region}, the fastest shipping mode on average is '{best_mode}' "
        f"with mean delay {avg_delay:.2f} days."
    )

if __name__ == "__main__":
    mcp.run(transport="stdio")

from typing import Any
import httpx
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("openmeteo")
OPEN_METEO = "https://api.open-meteo.com/v1/forecast"


@mcp.tool()
async def get_weather(latitude: float, longitude: float) -> str:
    """Give a brief weather forecast based on coordinates.
    Args:
        latitude: latitude
        longitude: longitude
    """
    data = await fetch_weather(latitude, longitude)
    if not data or "hourly" not in data:
        return "Could not get forecast."
    temps = data["hourly"]["temperature_2m"][:4]
    times = data["hourly"]["time"][:4]
    lines = [f"{t}: {temp}°C" for t, temp in zip(times, temps)]
    return "Brief forecast (nearest hours):\n" + "\n".join(lines)


async def fetch_weather(lat: float, lon: float) -> dict[str, Any] | None:
    params = {"latitude": lat, "longitude": lon, "hourly": "temperature_2m", "forecast_days": 1}
    async with httpx.AsyncClient(timeout=20.0) as client:
        try:
            r = await client.get(OPEN_METEO, params=params)
            r.raise_for_status()
            return r.json()
        except Exception:
            return None


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()

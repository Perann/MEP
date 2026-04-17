import os
import requests
from datetime import datetime


class FlightDataCollector:
    """
    A service class to fetch, filter, and format flight data from the AviationStack API
    specifically for Indian domestic routes and standardized time categories.
    """

    def __init__(self, api_key):
        """
        Initialize the collector with API credentials and categorical mappings.

        Args:
            api_key (str): Authentication key for AviationStack API.
        """
        self.api_key = api_key
        self.base_url = "http://api.aviationstack.com/v1/flights"
        self.allowed_airlines = [
            "SpiceJet",
            "AirAsia",
            "Vistara",
            "GO_FIRST",
            "Indigo",
            "Air_India",
        ]
        self.city_to_iata = {
            "Delhi": "DEL",
            "Mumbai": "BOM",
            "Bangalore": "BLR",
            "Kolkata": "CCU",
            "Hyderabad": "HYD",
            "Chennai": "MAA",
        }
        self.iata_to_city = {v: k for k, v in self.city_to_iata.items()}

    def get_time_category(self, dt_str):
        """
        Categorizes a timestamp into specific day periods.

        Args:
            dt_str (str): ISO 8601 formatted date string.

        Returns:
            str: Time category from ['Early_Morning', 'Morning', 'Afternoon', 'Evening', 'Night', 'Late_Night'].
        """
        dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
        hour = dt.hour

        if 0 <= hour < 4:
            return "Late_Night"
        elif 4 <= hour < 8:
            return "Early_Morning"
        elif 8 <= hour < 12:
            return "Morning"
        elif 12 <= hour < 17:
            return "Afternoon"
        elif 17 <= hour < 21:
            return "Evening"
        else:
            return "Night"

    def calculate_duration(self, dep_str, arr_str):
        """
        Calculates the flight duration between departure and arrival.

        Args:
            dep_str (str): Scheduled departure time (ISO 8601).
            arr_str (str): Scheduled arrival time (ISO 8601).

        Returns:
            float: Duration in decimal hours, rounded to 2 decimal places.
        """
        dep = datetime.fromisoformat(dep_str.replace("Z", "+00:00"))
        arr = datetime.fromisoformat(arr_str.replace("Z", "+00:00"))
        duration = arr - dep
        return round(duration.total_seconds() / 3600, 2)

    def fetch_and_format(self, source_city, destination_city, ticket_class="Economy"):
        """
        Retrieves flight data from the API, filters by airline, and maps IATA codes back to city names.

        Args:
            source_city (str): Name of the origin city.
            destination_city (str): Name of the destination city.
            ticket_class (str): Travel class (default: "Economy").

        Returns:
            dict: Formatted flight record or error message.
        """
        origin_iata = self.city_to_iata.get(source_city)
        dest_iata = self.city_to_iata.get(destination_city)

        if not origin_iata or not dest_iata:
            return {
                "error": f"Invalid city. Supported: {list(self.city_to_iata.keys())}"
            }

        params = {
            "access_key": self.api_key,
            "dep_iata": origin_iata,
            "arr_iata": dest_iata,
            "limit": 20,
        }

        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            res_json = response.json()

            flights = res_json.get("data", [])
            if not flights:
                return {"error": "No flights found for this route."}

            selected_flight = None
            for f in flights:
                api_airline = f["airline"]["name"].replace(" ", "_")
                if api_airline in self.allowed_airlines:
                    selected_flight = f
                    selected_flight["airline_formatted"] = api_airline
                    break

            if not selected_flight:
                return {
                    "error": f"No flights found for allowed airlines: {self.allowed_airlines}"
                }

            return {
                "airline": selected_flight["airline_formatted"],
                "flight": selected_flight["flight"]["iata"],
                "source_city": self.iata_to_city.get(
                    selected_flight["departure"]["iata"], source_city
                ),
                "departure_time": self.get_time_category(
                    selected_flight["departure"]["scheduled"]
                ),
                "stops": "zero",
                "arrival_time": self.get_time_category(
                    selected_flight["arrival"]["scheduled"]
                ),
                "destination_city": self.iata_to_city.get(
                    selected_flight["arrival"]["iata"], destination_city
                ),
                "class": ticket_class,
                "duration": self.calculate_duration(
                    selected_flight["departure"]["scheduled"],
                    selected_flight["arrival"]["scheduled"],
                ),
            }

        except Exception as e:
            return {"error": f"API Connection Error: {str(e)}"}


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    API_KEY = os.getenv("AVIATION_STACK_API_KEY")
    collector = FlightDataCollector(API_KEY)
    result = collector.fetch_and_format("Delhi", "Mumbai", ticket_class="Economy")
    print(result)

"""
Real-time Air Quality Data Pipeline for Delhi
Fetches live NO2 and CO concentrations from World Air Quality Index API
"""

import requests
import json
import time
from datetime import datetime
from typing import Dict, Optional, Any
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class AirQualityPipeline:
    """Pipeline to fetch and process real-time air quality data for Delhi"""
    
    def __init__(self, api_token: Optional[str] = None, city: str = "delhi"):
        """
        Initialize the Air Quality Pipeline
        
        Args:
            api_token: WAQI API token (if None, will try to get from environment)
            city: City name or station ID (default: "delhi")
        """
        self.api_token = api_token or os.getenv("WAQI_API_TOKEN", "demo")
        self.city = city
        self.base_url = "https://api.waqi.info/feed"
        self.session = requests.Session()
        
    def fetch_air_quality_data(self) -> Dict[str, Any]:
        """
        Fetch real-time air quality data for Delhi
        
        Returns:
            Dictionary containing the API response
        """
        url = f"{self.base_url}/{self.city}/"
        params = {"token": self.api_token}
        
        try:
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if data.get("status") == "error":
                error_msg = data.get("message", "Unknown error")
                raise Exception(f"API Error: {error_msg}")
            
            return data
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Network error: {str(e)}")
        except json.JSONDecodeError as e:
            raise Exception(f"Invalid JSON response: {str(e)}")
    
    def extract_no2_co_concentrations(self, api_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract NO2 and CO concentrations from API response
        
        Args:
            api_data: Raw API response dictionary
            
        Returns:
            Dictionary with extracted NO2 and CO data
        """
        if api_data.get("status") != "ok":
            return {"error": "Invalid API response status"}
        
        data = api_data.get("data", {})
        iaqi = data.get("iaqi", {})
        
        # Extract NO2 concentration
        no2_data = iaqi.get("no2", {})
        no2_value = no2_data.get("v") if no2_data else None
        
        # Extract CO concentration
        co_data = iaqi.get("co", {})
        co_value = co_data.get("v") if co_data else None
        
        # Extract additional metadata
        result = {
            "timestamp": datetime.now().isoformat(),
            "station_id": data.get("idx"),
            "station_name": data.get("city", {}).get("name"),
            "location": {
                "latitude": data.get("city", {}).get("geo", [None, None])[0],
                "longitude": data.get("city", {}).get("geo", [None, None])[1]
            },
            "overall_aqi": data.get("aqi"),
            "measurement_time": data.get("time", {}).get("s"),
            "timezone": data.get("time", {}).get("tz"),
            "concentrations": {
                "no2": {
                    "value": no2_value,
                    "unit": "µg/m³",
                    "available": no2_value is not None
                },
                "co": {
                    "value": co_value,
                    "unit": "µg/m³",
                    "available": co_value is not None
                }
            },
            "all_iaqi": iaqi  # Include all individual AQI values for reference
        }
        
        return result
    
    def get_delhi_air_quality(self) -> Dict[str, Any]:
        """
        Main method to fetch and extract NO2 and CO concentrations for Delhi
        
        Returns:
            Dictionary with processed air quality data
        """
        try:
            # Fetch raw data from API
            raw_data = self.fetch_air_quality_data()
            
            # Extract NO2 and CO concentrations
            processed_data = self.extract_no2_co_concentrations(raw_data)
            
            return {
                "status": "success",
                "data": processed_data,
                "raw_response": raw_data  # Include raw response for debugging
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def save_to_file(self, data: Dict[str, Any], filename: str = "delhi_air_quality.json"):
        """
        Save air quality data to a JSON file
        
        Args:
            data: Data dictionary to save
            filename: Output filename
        """
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Data saved to {filename}")
    
    def continuous_monitoring(self, interval_seconds: int = 3600, output_file: Optional[str] = None):
        """
        Continuously monitor air quality at specified intervals
        
        Args:
            interval_seconds: Time between fetches in seconds (default: 1 hour)
            output_file: Optional file to append data to
        """
        print(f"Starting continuous monitoring for {self.city}")
        print(f"Fetch interval: {interval_seconds} seconds ({interval_seconds/60:.1f} minutes)")
        print("Press Ctrl+C to stop\n")
        
        all_data = []
        
        try:
            while True:
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Fetching data...")
                
                result = self.get_delhi_air_quality()
                
                if result["status"] == "success":
                    data = result["data"]
                    no2 = data["concentrations"]["no2"]["value"]
                    co = data["concentrations"]["co"]["value"]
                    
                    print(f"  Station: {data.get('station_name', 'N/A')}")
                    print(f"  Overall AQI: {data.get('overall_aqi', 'N/A')}")
                    print(f"  NO2: {no2 if no2 else 'N/A'} µg/m³")
                    print(f"  CO: {co if co else 'N/A'} µg/m³")
                    print()
                    
                    all_data.append(result)
                    
                    # Save to file if specified
                    if output_file:
                        self.save_to_file(all_data, output_file)
                else:
                    print(f"  Error: {result.get('error', 'Unknown error')}\n")
                
                # Wait for next interval
                time.sleep(interval_seconds)
                
        except KeyboardInterrupt:
            print("\n\nMonitoring stopped by user")
            if all_data and output_file:
                print(f"Final data saved to {output_file}")
                self.save_to_file(all_data, output_file)


def main():
    """Main function to run the pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Air Quality Data Pipeline for Delhi")
    parser.add_argument("--token", type=str, help="WAQI API token (or set WAQI_API_TOKEN env var)")
    parser.add_argument("--city", type=str, default="delhi", help="City name (default: delhi)")
    parser.add_argument("--output", type=str, help="Output JSON file path")
    parser.add_argument("--continuous", action="store_true", help="Run continuous monitoring")
    parser.add_argument("--interval", type=int, default=3600, help="Interval in seconds for continuous mode (default: 3600)")
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = AirQualityPipeline(api_token=args.token, city=args.city)
    
    if args.continuous:
        # Run continuous monitoring
        pipeline.continuous_monitoring(
            interval_seconds=args.interval,
            output_file=args.output or "delhi_air_quality_continuous.json"
        )
    else:
        # Single fetch
        print(f"Fetching air quality data for {args.city}...\n")
        result = pipeline.get_delhi_air_quality()
        
        if result["status"] == "success":
            data = result["data"]
            
            print("=" * 60)
            print("AIR QUALITY DATA FOR DELHI")
            print("=" * 60)
            print(f"Station ID: {data.get('station_id', 'N/A')}")
            print(f"Station Name: {data.get('station_name', 'N/A')}")
            print(f"Location: {data.get('location', {}).get('latitude', 'N/A')}, {data.get('location', {}).get('longitude', 'N/A')}")
            print(f"Overall AQI: {data.get('overall_aqi', 'N/A')}")
            print(f"Measurement Time: {data.get('measurement_time', 'N/A')}")
            print(f"Timezone: {data.get('timezone', 'N/A')}")
            print("\n" + "-" * 60)
            print("CONCENTRATIONS:")
            print("-" * 60)
            
            no2 = data["concentrations"]["no2"]
            co = data["concentrations"]["co"]
            
            print(f"NO2: {no2['value'] if no2['value'] else 'N/A'} {no2['unit']} {'✓' if no2['available'] else '✗ Not available'}")
            print(f"CO:  {co['value'] if co['value'] else 'N/A'} {co['unit']} {'✓' if co['available'] else '✗ Not available'}")
            print("=" * 60)
            
            # Print all available IAQI values
            if data.get("all_iaqi"):
                print("\nAll Available Individual AQI Values:")
                for pollutant, value_obj in data["all_iaqi"].items():
                    if isinstance(value_obj, dict) and "v" in value_obj:
                        print(f"  {pollutant.upper()}: {value_obj['v']}")
            
        else:
            print(f"Error: {result.get('error', 'Unknown error')}")
        
        # Save to file if specified
        if args.output:
            pipeline.save_to_file(result, args.output)
        else:
            # Save to default file
            pipeline.save_to_file(result, "delhi_air_quality.json")


if __name__ == "__main__":
    main()


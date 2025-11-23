"""
Example usage of the Air Quality Data Pipeline
"""

from air_quality_pipeline import AirQualityPipeline

# Example 1: Basic usage with default settings
print("Example 1: Fetching Delhi air quality data...")
pipeline = AirQualityPipeline(city="delhi")
result = pipeline.get_delhi_air_quality()

if result["status"] == "success":
    data = result["data"]
    print(f"Station: {data.get('station_name')}")
    print(f"NO2: {data['concentrations']['no2']['value']} {data['concentrations']['no2']['unit']}")
    print(f"O3: {data['concentrations']['o3']['value']} {data['concentrations']['o3']['unit']}")
else:
    print(f"Error: {result.get('error')}")

# Example 2: Save to file
print("\nExample 2: Saving data to file...")
pipeline.save_to_file(result, "example_output.json")

# Example 3: Using custom API token
# pipeline = AirQualityPipeline(api_token="your_token_here", city="delhi")
# result = pipeline.get_delhi_air_quality()

print("\nDone! Check example_output.json for the full data.")


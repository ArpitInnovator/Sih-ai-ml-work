"""
Script to fetch air quality data for all 7 Delhi monitoring sites
"""

from air_quality_pipeline import AirQualityPipeline
import json

# Define all 7 Delhi sites with their coordinates
DELHI_SITES = [
    {"site": 1, "latitude": 28.69536, "longitude": 77.18168},
    {"site": 2, "latitude": 28.5718, "longitude": 77.07125},
    {"site": 3, "latitude": 28.58278, "longitude": 77.23441},
    {"site": 4, "latitude": 28.82286, "longitude": 77.10197},
    {"site": 5, "latitude": 28.53077, "longitude": 77.27123},
    {"site": 6, "latitude": 28.72954, "longitude": 77.09601},
    {"site": 7, "latitude": 28.71052, "longitude": 77.24951}
]

def main():
    """Fetch air quality data for all Delhi sites"""
    print("=" * 70)
    print("DELHI AIR QUALITY DATA PIPELINE - ALL SITES")
    print("=" * 70)
    print(f"Fetching data for {len(DELHI_SITES)} monitoring sites in Delhi")
    print()
    
    # Initialize pipeline
    pipeline = AirQualityPipeline(city="delhi")
    
    # Fetch data for all sites
    results = pipeline.fetch_multiple_locations(DELHI_SITES)
    
    # Display summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total Sites: {results['total_locations']}")
    print(f"Successful: {results['summary']['successful']}")
    print(f"Errors: {results['summary']['errors']}")
    print("=" * 70)
    
    # Display detailed results table
    print("\n" + "=" * 70)
    print("DETAILED RESULTS")
    print("=" * 70)
    print(f"{'Site':<6} {'Status':<8} {'Station Name':<30} {'AQI':<6} {'NO2 (µg/m³)':<12} {'O3 (µg/m³)':<12}")
    print("-" * 70)
    
    for loc in results["locations"]:
        site = loc.get("site")
        status = loc.get("status", "unknown")
        
        if status == "success":
            data = loc.get("data", {})
            station = data.get("station_name", "N/A")
            aqi = data.get("overall_aqi", "N/A")
            no2 = data.get("concentrations", {}).get("no2", {}).get("value")
            o3 = data.get("concentrations", {}).get("o3", {}).get("value")
            
            # Truncate station name if too long
            if len(station) > 28:
                station = station[:25] + "..."
            
            print(f"{site:<6} {'✓':<8} {station:<30} {aqi if aqi else 'N/A':<6} {no2 if no2 else 'N/A':<12} {o3 if o3 else 'N/A':<12}")
        else:
            error = loc.get("error", "Unknown error")
            if len(error) > 50:
                error = error[:47] + "..."
            print(f"{site:<6} {'✗':<8} {error:<30} {'-':<6} {'-':<12} {'-':<12}")
    
    print("=" * 70)
    
    # Save to file
    output_file = "delhi_all_sites_air_quality.json"
    pipeline.save_to_file(results, output_file)
    
    print(f"\n✓ Data saved to {output_file}")
    print("\nYou can also run: python air_quality_pipeline.py --multiple")

if __name__ == "__main__":
    main()


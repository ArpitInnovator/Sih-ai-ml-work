# Real-time Air Quality Data Pipeline for Delhi

A Python data pipeline to fetch live air quality data for Delhi, specifically extracting NO2 (Nitrogen Dioxide) and CO (Carbon Monoxide) concentrations from the World Air Quality Index (WAQI) API.

## Features

- ✅ Real-time air quality data fetching for Delhi
- ✅ Extraction of NO2 and CO concentrations
- ✅ Support for single fetch or continuous monitoring
- ✅ JSON output for data storage
- ✅ Error handling and validation
- ✅ Configurable API token management

## Prerequisites

- Python 3.7 or higher
- WAQI API token (get one from [aqicn.org/data-platform/token/](https://aqicn.org/data-platform/token/))

## Installation

1. Clone or navigate to this directory
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set up your API token:

   - Copy `.env.example` to `.env`:
     ```bash
     copy .env.example .env
     ```
   
   - Edit `.env` and add your API token:
     ```
     WAQI_API_TOKEN=your_actual_token_here
     ```

## Usage

### Single Data Fetch

Fetch air quality data once:

```bash
python air_quality_pipeline.py
```

This will:
- Fetch current air quality data for Delhi
- Display NO2 and CO concentrations
- Save data to `delhi_air_quality.json`

### Custom Output File

```bash
python air_quality_pipeline.py --output my_data.json
```

### Continuous Monitoring

Monitor air quality at regular intervals (default: 1 hour):

```bash
python air_quality_pipeline.py --continuous
```

With custom interval (e.g., every 30 minutes):

```bash
python air_quality_pipeline.py --continuous --interval 1800
```

### Different City

Fetch data for a different city:

```bash
python air_quality_pipeline.py --city shanghai
```

### Using API Token as Command Line Argument

```bash
python air_quality_pipeline.py --token your_token_here
```

## API Response Structure

The pipeline extracts the following information:

- **Station Information**: ID, name, location (latitude/longitude)
- **Overall AQI**: Air Quality Index value
- **NO2 Concentration**: Nitrogen Dioxide in µg/m³
- **CO Concentration**: Carbon Monoxide in µg/m³
- **Measurement Time**: Timestamp and timezone
- **All IAQI Values**: All available individual AQI measurements

## Output Format

The pipeline outputs JSON data with the following structure:

```json
{
  "status": "success",
  "data": {
    "timestamp": "2024-01-15T10:30:00",
    "station_id": 7397,
    "station_name": "Delhi, India",
    "location": {
      "latitude": "28.6139",
      "longitude": "77.2090"
    },
    "overall_aqi": 150,
    "measurement_time": "2024-01-15 10:00:00",
    "timezone": "+05:30",
    "concentrations": {
      "no2": {
        "value": 45,
        "unit": "µg/m³",
        "available": true
      },
      "co": {
        "value": 1200,
        "unit": "µg/m³",
        "available": true
      }
    }
  }
}
```

## Error Handling

The pipeline handles various error scenarios:

- **Network errors**: Connection timeouts, network failures
- **API errors**: Invalid token, over quota, unknown city
- **Data availability**: Missing NO2/CO data (some stations may not have all pollutants)

## API Documentation

For more information about the WAQI API, visit:
- [API Overview](https://aqicn.org/api/)
- [City Feed Documentation](https://aqicn.org/api/)

## Example API Endpoint

```
https://api.waqi.info/feed/delhi/?token=your_token
```

## Notes

- The API uses "demo" token by default (limited requests)
- For production use, get your own token from WAQI
- Some monitoring stations may not have NO2 or CO data available
- The API may have rate limits depending on your token type

## License

This pipeline is provided as-is for data collection purposes. Please refer to WAQI's terms of service for API usage.


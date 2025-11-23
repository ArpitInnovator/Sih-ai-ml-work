"""
Quick test script to verify admin endpoints are accessible
"""
import requests
import json

BASE_URL = "http://localhost:8000"

def test_endpoints():
    """Test admin endpoints"""
    endpoints = [
        "/",
        "/api/v1/admin/",
        "/api/v1/admin/models",
        "/api/v1/admin/system/info",
        "/api/v1/admin/config",
    ]
    
    print("Testing Admin Endpoints...")
    print("=" * 50)
    
    for endpoint in endpoints:
        url = f"{BASE_URL}{endpoint}"
        try:
            response = requests.get(url, timeout=5)
            print(f"\n✓ {endpoint}")
            print(f"  Status: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"  Response keys: {list(data.keys())[:5]}...")
            else:
                print(f"  Response: {response.text[:100]}")
        except requests.exceptions.ConnectionError:
            print(f"\n✗ {endpoint}")
            print("  Error: Server not running or not accessible")
        except Exception as e:
            print(f"\n✗ {endpoint}")
            print(f"  Error: {str(e)}")
    
    print("\n" + "=" * 50)
    print("Test complete!")
    print(f"\nAPI Docs: {BASE_URL}/docs")

if __name__ == "__main__":
    test_endpoints()


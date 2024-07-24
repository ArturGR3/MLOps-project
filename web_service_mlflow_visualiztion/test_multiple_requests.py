import requests
import json
import time
import click

test_example = {
    "id": 0,
    "store_sales_in_millions_": 8.609999656677246,
    "unit_sales_in_millions_": 3,
    "total_children": 2,
    "num_children_at_home": 2,
    "avg_cars_at_home_approx__1": 2,
    "gross_weight": 10.300000190734863,
    "recyclable_package": 1,
    "low_fat": 0,
    "units_per_case": 32,
    "store_sqft": 36509,
    "coffee_bar": 0,
    "video_store": 0,
    "salad_bar": 0,
    "prepared_food": 0,
    "florist": 0,
    "cost": 62.09000015258789,
    "autoFE_f_0": 6.140243153628883e-05,
    "autoFE_f_1": 0.0,
    "autoFE_f_2": 0.9808552748753283,
    "autoFE_f_3": 36511.0,
    "autoFE_f_4": 2992.0,
}

url = "http://localhost:9696/predict"

def send_request():
    response = requests.post(url, json=test_example)
    if response.status_code == 200:
        try:
            json_response = response.json()
            print(json_response)
        except json.JSONDecodeError as e:
            print("Failed to decode JSON response:", e)
    else:
        print("Request failed with status:", response.status_code)
        print("Response text:", response.text)
        
# Number of requests to send
@click.command()
@click.option("--num_requests", default=1000, help="Number of requests to send")
def main(num_requests):
    # Send 100 requests with a 1-second delay between each
    for _ in range(num_requests):
        send_request()
        time.sleep(0.5)  # Wait for 1 second before sending the next request

if __name__ == "__main__":
    main()
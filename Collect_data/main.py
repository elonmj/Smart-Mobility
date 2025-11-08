from datetime import datetime
from traffic_collector import TrafficDataCollector
import time
import schedule

def collect_data():
    collector = TrafficDataCollector()
    data = collector.get_traffic_data()
    collector.save_data(data)
    print(f"Data collection completed at {datetime.now()}")

def main():
    # Schedule data collection every hour
    schedule.every().hour.at(":00").do(collect_data)
    
    # Run initial collection
    collect_data()
    
    # Keep the script running
    while True:
        schedule.run_pending()
        time.sleep(60)

if __name__ == "__main__":
    main()

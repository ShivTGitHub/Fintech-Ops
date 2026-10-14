import pandas as pd
import random
from faker import Faker
from datetime import datetime, timedelta

fake = Faker()

def generate_switch_logs(num_normal=50, num_outage=15):
    logs = []
    base_time = datetime.now() - timedelta(hours=1)

    # 1. Generate normal background noise (localized errors)
    for _ in range(num_normal):
        logs.append({
            "timestamp": (base_time + timedelta(minutes=random.randint(1, 59))).isoformat(),
            "bank_id": f"BANK_{random.randint(100, 999)}",
            "terminal_id": fake.bban(),
            "error_msg": random.choice([
                "Pin pad disconnected at terminal",
                "Biometric scanner timeout on AePS device",
                "Insufficient funds in customer account",
                "Daily transaction limit exceeded"
            ]),
            "response_code": random.choice(["68", "51", "61"])
        })

    # 2. Inject a systemic sponsor-bank outage (The anomaly cluster)
    target_bank = "BANK_500_SPONSOR"
    for _ in range(num_outage):
        logs.append({
            "timestamp": (base_time + timedelta(minutes=random.randint(40, 45))).isoformat(),
            "bank_id": target_bank,
            "terminal_id": fake.bban(),
            "error_msg": random.choice([
                "Connection timeout to upstream authorization endpoint",
                "Sponsor bank host failed to respond within 30000ms",
                "TCP socket reset by peer during ISO 8583 transmission"
            ]),
            "response_code": "91"
        })

    df = pd.DataFrame(logs)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df.sort_values('timestamp').reset_index(drop=True)
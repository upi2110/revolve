import datetime

def check_weekly_reminder(log_path="logs/last_reminder.txt"):
    today = datetime.date.today()
    try:
        with open(log_path, 'r') as f:
            last = datetime.datetime.strptime(f.read().strip(), '%Y-%m-%d').date()
    except:
        last = today - datetime.timedelta(days=8)

    if (today - last).days >= 7:
        print("🔔 Time to upload a new 500-number dataset!")
        with open(log_path, 'w') as f:
            f.write(str(today))
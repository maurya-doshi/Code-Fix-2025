import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from datetime import datetime

# ---------------------------
# Argument Parser
# ---------------------------
parser = argparse.ArgumentParser(description="SQLite DB Analyzer & Email Report")
parser.add_argument("--db", required=True, help="Path to SQLite database")
parser.add_argument("--email", required=True, help="Recipient email")
args = parser.parse_args()

db_path = args.db
recipient_email = args.email

print(f"Database Path: {db_path}")
print(f"Recipient Email: {recipient_email}")

# ---------------------------
# Connect to DB
# ---------------------------
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# ---------------------------
# Get Tables
# ---------------------------
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = [t[0] for t in cursor.fetchall()]

if not tables:
    print("📌 No tables found in database!")
    exit()
else:
    print(f"📌 Tables Found in Database: {tables}")

# ---------------------------
# Analyze Tables
# ---------------------------
stats_summary = ""
charts = []

for table in tables:
    print(f"\nAnalyzing table: {table}")
    df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
    
    # Basic info
    print(df.info())
    print(df.describe())
    
    stats_summary += f"<h3>Table: {table}</h3>"
    stats_summary += df.describe().to_html()
    
    # Categorical columns visualization
    cat_cols = df.select_dtypes(include="object").columns
    if len(cat_cols) > 0:
        plt.figure(figsize=(6,4))
        sns.countplot(x=cat_cols[0], data=df, palette="Set2")
        plt.title(f"{table} - {cat_cols[0]} Distribution")
        chart_path = f"output/{table}_{cat_cols[0]}_count.png"
        plt.tight_layout()
        plt.savefig(chart_path, dpi=300)
        charts.append(chart_path)
        plt.close()
    
    # Numerical correlation heatmap
    num_cols = df.select_dtypes(include="number").columns
    if len(num_cols) > 1:
        plt.figure(figsize=(6,4))
        sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm")
        plt.title(f"{table} - Numeric Correlation")
        chart_path = f"output/{table}_correlation.png"
        plt.tight_layout()
        plt.savefig(chart_path, dpi=300)
        charts.append(chart_path)
        plt.close()

# ---------------------------
# Generate Report (HTML)
# ---------------------------
report_file = "output/report.html"
with open(report_file, "w") as f:
    f.write(f"<h1>Database Analysis Report</h1>")
    f.write(f"<p>Database: {db_path}<br>Analysis Date: {datetime.now()}</p>")
    f.write(stats_summary)
    for c in charts:
        f.write(f'<img src="{c}" width="600"><br>')

print(f"\n✅ Report generated: {report_file}")

# ---------------------------
# Send Email
# ---------------------------
sender_email = os.environ.get("SENDER_EMAIL")
sender_password = os.environ.get("SENDER_PASSWORD")

if not sender_email or not sender_password:
    print("❌ Missing email credentials. Set SENDER_EMAIL and SENDER_PASSWORD as environment variables.")
    exit()

msg = MIMEMultipart()
msg['From'] = sender_email
msg['To'] = recipient_email
msg['Subject'] = "Database Analysis Report - AI CODEFIX 2025"

body = f"""
Dear Recipient,<br><br>
Please find the automated database analysis report attached.<br><br>
Total Tables: {len(tables)}<br>
Analysis Date: {datetime.now()}<br><br>
Best regards,<br>
AI CODEFIX 2025
"""
msg.attach(MIMEText(body, "html"))

# Attach report
with open(report_file, "rb") as f:
    part = MIMEBase('application', 'octet-stream')
    part.set_payload(f.read())
encoders.encode_base64(part)
part.add_header('Content-Disposition', f'attachment; filename=report.html')
msg.attach(part)

# Attach charts
for c in charts:
    with open(c, "rb") as f:
        part = MIMEBase('application', 'octet-stream')
        part.set_payload(f.read())
    encoders.encode_base64(part)
    part.add_header('Content-Disposition', f'attachment; filename={os.path.basename(c)}')
    msg.attach(part)

# Send via SMTP
try:
    server = smtplib.SMTP("smtp.gmail.com", 587)
    server.starttls()
    server.login(sender_email, sender_password)
    server.send_message(msg)
    server.quit()
    print("✅ Email sent successfully!")
except Exception as e:
    print(f"❌ Failed to send email: {e}")

# Close DB connection
conn.close()

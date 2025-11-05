# Accessing Audit Logs on Render

This guide shows you how to access audit logs from your Render deployment.

## Method 1: Render Shell (Easiest)

### Step 1: Open Render Shell
1. Go to https://dashboard.render.com
2. Click on your web service (bnb-chat-with-data)
3. Click "Shell" in the left sidebar

### Step 2: Navigate to logs
```bash
cd backend/audit_logs
ls -lh
```

### Step 3: View logs
```bash
# View today's log file name
echo audit_$(date +%Y-%m-%d).jsonl

# Count total queries today
wc -l audit_$(date +%Y-%m-%d).jsonl

# View last 5 entries
tail -5 audit_$(date +%Y-%m-%d).jsonl

# View specific entry (formatted)
head -1 audit_$(date +%Y-%m-%d).jsonl | python -m json.tool

# Search for specific question
grep -i "oncology" audit_$(date +%Y-%m-%d).jsonl

# View all questions asked today
cat audit_$(date +%Y-%m-%d).jsonl | grep -o '"question":"[^"]*"' | cut -d'"' -f4
```

### Step 4: Download log file
```bash
# Copy content to clipboard (in shell)
cat audit_$(date +%Y-%m-%d).jsonl

# Then paste into a local file
```

## Method 2: API Endpoint (Programmatic Access)

### Get today's logs
```bash
curl -H "Authorization: Bearer oAzfiqXIYiHiEkX" \
  https://your-app.onrender.com/api/admin/audit-logs
```

### Get logs for specific date
```bash
curl -H "Authorization: Bearer oAzfiqXIYiHiEkX" \
  "https://your-app.onrender.com/api/admin/audit-logs?date=2025-11-05"
```

### Get logs for specific user
```bash
# First, find user hash (see Method 3)
curl -H "Authorization: Bearer oAzfiqXIYiHiEkX" \
  "https://your-app.onrender.com/api/admin/audit-logs?user_hash=1ac4c4fb37c2"
```

### Save to file
```bash
curl -H "Authorization: Bearer oAzfiqXIYiHiEkX" \
  https://your-app.onrender.com/api/admin/audit-logs > audit-logs-today.json
```

### Parse with jq
```bash
curl -H "Authorization: Bearer oAzfiqXIYiHiEkX" \
  https://your-app.onrender.com/api/admin/audit-logs | jq '.queries[] | .question'
```

## Method 3: Python Script (Best for Analysis)

Create a script `download_logs.py`:

```python
import requests
import json
from datetime import datetime

# Your access token
TOKEN = "oAzfiqXIYiHiEkX"
API_URL = "https://your-app.onrender.com/api/admin/audit-logs"

# Get today's logs
headers = {"Authorization": f"Bearer {TOKEN}"}
response = requests.get(API_URL, headers=headers)

if response.status_code == 200:
    data = response.json()
    print(f"Total queries: {data['total_queries']}")
    print(f"Date: {data['date']}")

    # Save to file
    with open(f"audit_logs_{data['date']}.json", "w") as f:
        json.dump(data, f, indent=2)

    # Print summary
    for query in data['queries']:
        print(f"\n{query['timestamp']}")
        print(f"  User: {query['user_hash']}")
        print(f"  Question: {query['question'][:60]}...")
        print(f"  Time: {query['execution_time_ms']:.1f}ms")
        if query['error']:
            print(f"  Error: {query['error'][:50]}...")
else:
    print(f"Error: {response.status_code}")
    print(response.text)
```

Run it:
```bash
python download_logs.py
```

## Method 4: Find User Hash from Token

If you need to find which user_hash corresponds to which token:

```python
import hashlib

# Token from USER_ACCESS_TOKENS.md
token = "oAzfiqXIYiHiEkX"
user_hash = hashlib.sha256(token.encode()).hexdigest()[:12]
print(f"Token: {token}")
print(f"Hash in logs: {user_hash}")
```

Result:
```
Token: oAzfiqXIYiHiEkX
Hash in logs: 1ac4c4fb37c2
```

Now you can filter logs by this hash:
```bash
curl -H "Authorization: Bearer oAzfiqXIYiHiEkX" \
  "https://your-app.onrender.com/api/admin/audit-logs?user_hash=1ac4c4fb37c2"
```

## Method 5: Persistent Disk (For Production)

**Important**: Render's filesystem is ephemeral. Logs will be lost on restart unless you add a persistent disk.

### Add Persistent Disk:
1. Go to Render Dashboard → Your Service → Disks
2. Click "Add Disk"
3. Configure:
   - **Name**: `audit-logs`
   - **Mount Path**: `/opt/render/project/src/backend/audit_logs`
   - **Size**: 10GB (adjust based on needs)
4. Click "Create Disk"
5. Redeploy your service

After adding persistent disk:
- Logs survive deployments and restarts
- Access using any method above
- Logs accumulate over time (implement retention policy)

## Quick Reference

### View recent activity
```bash
# In Render Shell
tail -10 backend/audit_logs/audit_$(date +%Y-%m-%d).jsonl | python -m json.tool
```

### Count queries per user
```bash
# In Render Shell
cat backend/audit_logs/audit_$(date +%Y-%m-%d).jsonl | \
  grep -o '"user_hash":"[^"]*"' | \
  sort | uniq -c
```

### Find errors
```bash
# In Render Shell
grep '"error":' backend/audit_logs/audit_$(date +%Y-%m-%d).jsonl | \
  grep -v '"error": null'
```

### Export all logs for date range
```bash
# Using API endpoint (multiple dates)
for date in 2025-11-{01..05}; do
  curl -H "Authorization: Bearer oAzfiqXIYiHiEkX" \
    "https://your-app.onrender.com/api/admin/audit-logs?date=$date" \
    > audit_$date.json
done
```

## Troubleshooting

### "No such file or directory"
- Logs haven't been created yet (no queries made)
- Or using wrong date format
- Check: `ls backend/audit_logs/`

### "Permission denied"
- Check Render Shell has access
- Verify audit_logs directory was created
- Check logs in Render logs tab for audit_logger errors

### Empty response from API
- No queries made on that date
- Check date format is YYYY-MM-DD
- Verify AUTH_TOKEN is set correctly

### Logs disappeared after deployment
- Render filesystem is ephemeral
- Add persistent disk (see Method 5)
- Or use external storage (S3, GCS, etc.)

## Next Steps

- Set up automated daily log downloads
- Configure log rotation/archival
- Set up alerts for error rates
- Create dashboard for usage metrics

See [AUDIT_LOGGING_GUIDE.md](AUDIT_LOGGING_GUIDE.md) for more analysis examples.

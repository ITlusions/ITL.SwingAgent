# Deployment Guide

This guide covers production deployment strategies for the SwingAgent system.

## Deployment Architecture Options

### 1. Single Server Deployment

Simple deployment for individual traders or small teams.

```
┌─────────────────────────────────┐
│         Server                  │
│  ┌─────────────────────────────┐│
│  │     SwingAgent              ││
│  │                             ││
│  │  ┌─────────┐ ┌─────────────┐││
│  │  │Scripts  │ │  Databases  │││
│  │  │         │ │             │││
│  │  │run_*    │ │signals.db   │││
│  │  │eval_*   │ │vectors.db   │││
│  │  │analyze_*│ │             │││
│  │  └─────────┘ └─────────────┘││
│  └─────────────────────────────┘│
│                                 │
│  External APIs:                 │
│  - Yahoo Finance                │
│  - OpenAI                       │
└─────────────────────────────────┘
```

### 2. Microservices Deployment

Scalable deployment with separated concerns.

```
┌─────────────────────────────────────────────────────────┐
│                   Load Balancer                         │
└─────────────────────┬───────────────────────────────────┘
                      │
        ┌─────────────┼─────────────┐
        │             │             │
┌───────▼──────┐ ┌────▼─────┐ ┌─────▼──────┐
│Signal Service│ │Eval      │ │Analytics   │
│              │ │Service   │ │Service     │
│- Live signals│ │          │ │            │
│- Backtesting │ │- Trade   │ │- Performance│
│              │ │  eval    │ │- Reporting │
└──────┬───────┘ └────┬─────┘ └─────┬──────┘
       │              │             │
┌──────▼──────────────▼─────────────▼──────┐
│            Shared Data Layer             │
│                                          │
│  ┌─────────────┐    ┌─────────────────┐  │
│  │PostgreSQL   │    │Redis Cache      │  │
│  │- Signals    │    │- Market data    │  │
│  │- Vectors    │    │- Session data   │  │
│  └─────────────┘    └─────────────────┘  │
└──────────────────────────────────────────┘
```

### 3. Cloud-Native Deployment

Enterprise deployment with auto-scaling and monitoring.

```
┌──────────────────────── AWS/GCP/Azure ─────────────────────────┐
│                                                                │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────┐ │
│  │API Gateway      │    │Container        │    │Monitoring   │ │
│  │                 │    │Orchestration    │    │             │ │
│  │- Rate limiting  │    │                 │    │- Logs       │ │
│  │- Authentication │    │- Auto-scaling   │    │- Metrics    │ │
│  │- Load balancing │    │- Health checks  │    │- Alerts     │ │
│  └─────────────────┘    └─────────────────┘    └─────────────┘ │
│                                                                │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                Container Cluster                        │   │
│  │                                                         │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐  │   │
│  │  │Signal Pods  │  │Eval Pods    │  │Analytics Pods   │  │   │
│  │  │             │  │             │  │                 │  │   │
│  │  │- SwingAgent │  │- Trade eval │  │- Performance    │  │   │
│  │  │- Market data│  │- Vector     │  │- Reporting      │  │   │
│  │  │- LLM calls  │  │  updates    │  │- API endpoints  │  │   │
│  │  └─────────────┘  └─────────────┘  └─────────────────┘  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                Managed Services                         │   │
│  │                                                         │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐  │   │
│  │  │RDS/CloudSQL │  │ElastiCache/ │  │Secret Manager   │  │   │
│  │  │             │  │Memorystore  │  │                 │  │   │
│  │  │- PostgreSQL │  │- Redis      │  │- API keys       │  │   │
│  │  │- Backups    │  │- Sessions   │  │- DB credentials │  │   │
│  │  └─────────────┘  └─────────────┘  └─────────────────┘  │   │
│  └─────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────┘
```

## Single Server Deployment

### Prerequisites

```bash
# System requirements
- Ubuntu 20.04+ / CentOS 8+ / RHEL 8+
- Python 3.10+
- 4GB RAM minimum, 8GB recommended
- 50GB storage for databases
- Internet connectivity for APIs
```

### Installation Steps

#### 1. System Setup

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python and dependencies
sudo apt install python3.10 python3.10-venv python3.10-dev \
                 git curl sqlite3 supervisor nginx -y

# Create application user
sudo useradd -m -s /bin/bash swingagent
sudo usermod -aG sudo swingagent
```

#### 2. Application Setup

```bash
# Switch to application user
sudo su - swingagent

# Clone repository
git clone https://github.com/ITlusions/ITL.SwingAgent.git
cd ITL.SwingAgent

# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate

# Install application
pip install --upgrade pip
pip install -e .
```

#### 3. Configuration

```bash
# Create configuration directory
mkdir -p /home/swingagent/config
mkdir -p /home/swingagent/data
mkdir -p /home/swingagent/logs

# Environment configuration
cat > /home/swingagent/config/env << 'EOF'
export OPENAI_API_KEY="sk-your-key-here"
export SWING_LLM_MODEL="gpt-4o-mini"
export SWING_SIGNALS_DB="/home/swingagent/data/signals.sqlite"
export SWING_VECTOR_DB="/home/swingagent/data/vec_store.sqlite"
export PYTHONPATH="/home/swingagent/ITL.SwingAgent/src"
EOF

# Load environment
source /home/swingagent/config/env
echo 'source /home/swingagent/config/env' >> ~/.bashrc
```

#### 4. Database Initialization

```bash
# Test database creation
cd /home/swingagent/ITL.SwingAgent
source venv/bin/activate
python -c "
from swing_agent.storage import record_signal
from swing_agent.vectorstore import add_vector
import numpy as np

# Test signal database
try:
    print('Signal database: OK')
except Exception as e:
    print(f'Signal database error: {e}')

# Test vector database  
try:
    add_vector(
        '/home/swingagent/data/vec_store.sqlite',
        vid='test', ts_utc='2024-01-01T00:00:00Z',
        symbol='TEST', timeframe='30m',
        vec=np.array([0.1, 0.2, 0.3]),
        realized_r=None, exit_reason=None, payload=None
    )
    print('Vector database: OK')
except Exception as e:
    print(f'Vector database error: {e}')
"
```

#### 5. Service Configuration

```bash
# Create supervisor configuration
sudo tee /etc/supervisor/conf.d/swingagent.conf << 'EOF'
[group:swingagent]
programs=signal-generator,signal-evaluator

[program:signal-generator]
command=/home/swingagent/ITL.SwingAgent/venv/bin/python scripts/run_swing_agent.py --symbol %(ENV_SYMBOL)s --interval 30m --db /home/swingagent/data/signals.sqlite --vec-db /home/swingagent/data/vec_store.sqlite
directory=/home/swingagent/ITL.SwingAgent
user=swingagent
environment=OPENAI_API_KEY="%(ENV_OPENAI_API_KEY)s",SWING_LLM_MODEL="gpt-4o-mini",SYMBOL="AAPL"
autostart=false
autorestart=true
stderr_logfile=/home/swingagent/logs/signal-generator.err.log
stdout_logfile=/home/swingagent/logs/signal-generator.out.log
startretries=3

[program:signal-evaluator]
command=/home/swingagent/ITL.SwingAgent/venv/bin/python scripts/eval_signals.py --db /home/swingagent/data/signals.sqlite --max-hold-days 2.0
directory=/home/swingagent/ITL.SwingAgent
user=swingagent
autostart=true
autorestart=true
stderr_logfile=/home/swingagent/logs/signal-evaluator.err.log
stdout_logfile=/home/swingagent/logs/signal-evaluator.out.log
startretries=3
startsecs=30
EOF

# Reload supervisor
sudo supervisorctl reread
sudo supervisorctl update
```

#### 6. Nginx Setup (Optional)

```bash
# Create nginx configuration for web interface
sudo tee /etc/nginx/sites-available/swingagent << 'EOF'
server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        return 301 https://$server_name$request_uri;
    }
}

server {
    listen 443 ssl;
    server_name your-domain.com;
    
    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;
    
    location /api/ {
        proxy_pass http://127.0.0.1:8000/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
    
    location / {
        root /home/swingagent/web;
        index index.html;
    }
}
EOF

# Enable site
sudo ln -s /etc/nginx/sites-available/swingagent /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

### Backup Strategy

```bash
# Create backup script
cat > /home/swingagent/scripts/backup.sh << 'EOF'
#!/bin/bash
BACKUP_DIR="/home/swingagent/backups"
DATE=$(date +%Y%m%d_%H%M%S)

mkdir -p $BACKUP_DIR

# Backup databases
sqlite3 /home/swingagent/data/signals.sqlite ".backup $BACKUP_DIR/signals_$DATE.sqlite"
sqlite3 /home/swingagent/data/vec_store.sqlite ".backup $BACKUP_DIR/vectors_$DATE.sqlite"

# Backup configuration
tar -czf $BACKUP_DIR/config_$DATE.tar.gz /home/swingagent/config/

# Cleanup old backups (keep 30 days)
find $BACKUP_DIR -type f -mtime +30 -delete

# Upload to cloud storage (optional)
# aws s3 sync $BACKUP_DIR s3://your-backup-bucket/swingagent/
EOF

chmod +x /home/swingagent/scripts/backup.sh

# Schedule backups
crontab -e
# Add: 0 2 * * * /home/swingagent/scripts/backup.sh
```

## Containerized Deployment

### Docker Setup

#### 1. Dockerfile

```dockerfile
# Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY pyproject.toml .
COPY src/ src/
RUN pip install --no-cache-dir -e .

# Create non-root user
RUN useradd -m -u 1000 swingagent
USER swingagent

# Create data directory
RUN mkdir -p /app/data

# Copy scripts
COPY scripts/ scripts/

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import swing_agent; print('OK')" || exit 1

# Default command
CMD ["python", "scripts/run_swing_agent.py", "--help"]
```

#### 2. Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  signal-generator:
    build: .
    container_name: swingagent-signals
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - SWING_LLM_MODEL=gpt-4o-mini
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    command: >
      sh -c "while true; do
        python scripts/run_swing_agent.py 
          --symbol AAPL 
          --interval 30m 
          --db /app/data/signals.sqlite 
          --vec-db /app/data/vec_store.sqlite;
        sleep 1800;
      done"
    restart: unless-stopped
    depends_on:
      - postgres
      
  signal-evaluator:
    build: .
    container_name: swingagent-evaluator
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    command: >
      sh -c "while true; do
        python scripts/eval_signals.py 
          --db /app/data/signals.sqlite 
          --max-hold-days 2.0;
        sleep 300;
      done"
    restart: unless-stopped
    depends_on:
      - postgres

  postgres:
    image: postgres:15
    container_name: swingagent-postgres
    environment:
      - POSTGRES_DB=swingagent
      - POSTGRES_USER=swingagent
      - POSTGRES_PASSWORD=${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql
    ports:
      - "5432:5432"
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    container_name: swingagent-redis
    volumes:
      - redis_data:/data
    ports:
      - "6379:6379"
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    container_name: swingagent-nginx
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - signal-generator
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
```

#### 3. Environment File

```bash
# .env
OPENAI_API_KEY=sk-your-key-here
DB_PASSWORD=your-secure-password
COMPOSE_PROJECT_NAME=swingagent
```

#### 4. Deployment Commands

```bash
# Build and start services
docker-compose up -d --build

# View logs
docker-compose logs -f signal-generator

# Scale services
docker-compose up -d --scale signal-generator=3

# Update services
docker-compose pull
docker-compose up -d

# Backup data
docker exec swingagent-postgres pg_dump -U swingagent swingagent > backup.sql
docker cp swingagent-signals:/app/data ./data-backup/
```

## Kubernetes Deployment

### 1. Namespace and ConfigMap

```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: swingagent
---
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: swingagent-config
  namespace: swingagent
data:
  SWING_LLM_MODEL: "gpt-4o-mini"
  DB_HOST: "postgres-service"
  REDIS_HOST: "redis-service"
```

### 2. Secrets

```yaml
# k8s/secrets.yaml
apiVersion: v1
kind: Secret
metadata:
  name: swingagent-secrets
  namespace: swingagent
type: Opaque
data:
  openai-api-key: <base64-encoded-key>
  db-password: <base64-encoded-password>
```

### 3. Signal Generator Deployment

```yaml
# k8s/signal-generator.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: signal-generator
  namespace: swingagent
spec:
  replicas: 2
  selector:
    matchLabels:
      app: signal-generator
  template:
    metadata:
      labels:
        app: signal-generator
    spec:
      containers:
      - name: signal-generator
        image: swingagent:latest
        command:
        - python
        - scripts/run_swing_agent.py
        - --symbol
        - AAPL
        - --interval
        - 30m
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: swingagent-secrets
              key: openai-api-key
        envFrom:
        - configMapRef:
            name: swingagent-config
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        volumeMounts:
        - name: data-volume
          mountPath: /app/data
      volumes:
      - name: data-volume
        persistentVolumeClaim:
          claimName: swingagent-data-pvc
```

### 4. PostgreSQL Deployment

```yaml
# k8s/postgres.yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: postgres
  namespace: swingagent
spec:
  serviceName: postgres-service
  replicas: 1
  selector:
    matchLabels:
      app: postgres
  template:
    metadata:
      labels:
        app: postgres
    spec:
      containers:
      - name: postgres
        image: postgres:15
        env:
        - name: POSTGRES_DB
          value: swingagent
        - name: POSTGRES_USER
          value: swingagent
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: swingagent-secrets
              key: db-password
        ports:
        - containerPort: 5432
        volumeMounts:
        - name: postgres-storage
          mountPath: /var/lib/postgresql/data
  volumeClaimTemplates:
  - metadata:
      name: postgres-storage
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 50Gi
---
apiVersion: v1
kind: Service
metadata:
  name: postgres-service
  namespace: swingagent
spec:
  selector:
    app: postgres
  ports:
  - port: 5432
    targetPort: 5432
```

### 5. Horizontal Pod Autoscaler

```yaml
# k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: signal-generator-hpa
  namespace: swingagent
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: signal-generator
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

## Monitoring and Observability

### 1. Prometheus Monitoring

```yaml
# k8s/monitoring.yaml
apiVersion: v1
kind: ServiceMonitor
metadata:
  name: swingagent-monitor
  namespace: swingagent
spec:
  selector:
    matchLabels:
      app: signal-generator
  endpoints:
  - port: metrics
    path: /metrics
    interval: 30s
```

### 2. Grafana Dashboard

```json
{
  "dashboard": {
    "title": "SwingAgent Monitoring",
    "panels": [
      {
        "title": "Signals Generated",
        "type": "stat",
        "targets": [
          {
            "expr": "increase(swingagent_signals_total[1h])",
            "legendFormat": "Signals/hour"
          }
        ]
      },
      {
        "title": "LLM API Calls",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(swingagent_llm_calls_total[5m])",
            "legendFormat": "Calls/sec"
          }
        ]
      },
      {
        "title": "Database Performance",
        "type": "graph",
        "targets": [
          {
            "expr": "swingagent_db_query_duration_seconds",
            "legendFormat": "Query time"
          }
        ]
      }
    ]
  }
}
```

### 3. Application Logging

```python
# logging_config.py
import logging
import sys
from pythonjsonlogger import jsonlogger

def setup_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    handler = logging.StreamHandler(sys.stdout)
    formatter = jsonlogger.JsonFormatter(
        '%(asctime)s %(name)s %(levelname)s %(message)s'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger

# In your application
logger = setup_logging()
logger.info("Signal generated", extra={
    "symbol": "AAPL",
    "signal_id": "abc123",
    "confidence": 0.75
})
```

## Security Considerations

### 1. API Key Management

```bash
# Using AWS Secrets Manager
aws secretsmanager create-secret \
    --name swingagent/openai-key \
    --description "OpenAI API key for SwingAgent" \
    --secret-string "sk-your-key-here"

# Using Kubernetes secrets
kubectl create secret generic swingagent-secrets \
    --from-literal=openai-api-key="sk-your-key-here" \
    --namespace=swingagent
```

### 2. Network Security

```yaml
# k8s/network-policy.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: swingagent-network-policy
  namespace: swingagent
spec:
  podSelector: {}
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: swingagent
  egress:
  - to: []
    ports:
    - protocol: TCP
      port: 443  # HTTPS for APIs
    - protocol: TCP
      port: 53   # DNS
```

### 3. Resource Limits

```yaml
# k8s/resource-quotas.yaml
apiVersion: v1
kind: ResourceQuota
metadata:
  name: swingagent-quota
  namespace: swingagent
spec:
  hard:
    requests.cpu: "4"
    requests.memory: 8Gi
    limits.cpu: "8"
    limits.memory: 16Gi
    persistentvolumeclaims: "10"
```

## Disaster Recovery

### 1. Backup Strategy

```bash
#!/bin/bash
# backup-script.sh

# Database backup
kubectl exec -n swingagent postgres-0 -- pg_dump -U swingagent swingagent > "backup-$(date +%Y%m%d).sql"

# Upload to cloud storage
aws s3 cp "backup-$(date +%Y%m%d).sql" s3://your-backup-bucket/

# Cleanup local files older than 7 days
find . -name "backup-*.sql" -mtime +7 -delete
```

### 2. Restore Procedure

```bash
#!/bin/bash
# restore-script.sh

BACKUP_FILE=$1

# Stop services
kubectl scale deployment signal-generator --replicas=0 -n swingagent

# Restore database
kubectl exec -n swingagent postgres-0 -- psql -U swingagent -d swingagent < $BACKUP_FILE

# Restart services
kubectl scale deployment signal-generator --replicas=2 -n swingagent
```

## Performance Optimization

### 1. Database Optimization

```sql
-- Database indexes for better performance
CREATE INDEX CONCURRENTLY idx_signals_symbol_asof ON signals(symbol, asof);
CREATE INDEX CONCURRENTLY idx_vectors_symbol_ts ON vec_store(symbol, ts_utc);

-- Partitioning for large datasets
CREATE TABLE signals_2024 PARTITION OF signals 
FOR VALUES FROM ('2024-01-01') TO ('2025-01-01');
```

### 2. Caching Strategy

```python
# redis_cache.py
import redis
import pickle
from functools import wraps

redis_client = redis.Redis(host='redis-service', port=6379, db=0)

def cache_result(expiration=3600):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"
            
            # Try to get from cache
            cached = redis_client.get(key)
            if cached:
                return pickle.loads(cached)
            
            # Compute and cache result
            result = func(*args, **kwargs)
            redis_client.setex(key, expiration, pickle.dumps(result))
            return result
        return wrapper
    return decorator

# Usage
@cache_result(expiration=1800)  # 30 minutes
def fetch_market_data(symbol, interval):
    return load_ohlcv(symbol, interval, 30)
```

This deployment guide provides comprehensive strategies for deploying SwingAgent in various environments, from simple single-server setups to enterprise-grade cloud-native deployments.
# CNPG (CloudNativePG) Setup Guide

This guide covers setting up SwingAgent with CNPG (CloudNativePG), a PostgreSQL operator for Kubernetes environments.

## Overview

CNPG provides cloud-native PostgreSQL clusters with high availability, automated backups, and seamless integration with Kubernetes. SwingAgent includes native CNPG support with optimized connection handling for Kubernetes environments.

## Prerequisites

- Kubernetes cluster (v1.23+)
- CNPG operator installed
- kubectl configured for your cluster

## Installing CNPG Operator

```bash
# Install CNPG operator
kubectl apply -f https://raw.githubusercontent.com/cloudnative-pg/cloudnative-pg/release-1.21/releases/cnpg-1.21.0.yaml

# Verify installation
kubectl get pods -n cnpg-system
```

## Creating a CNPG Cluster

### Basic CNPG Cluster

Create a basic PostgreSQL cluster for SwingAgent:

```yaml
# cnpg-cluster.yaml
apiVersion: postgresql.cnpg.io/v1
kind: Cluster
metadata:
  name: swing-postgres
  namespace: default
spec:
  instances: 3
  
  postgresql:
    parameters:
      max_connections: "200"
      shared_buffers: "256MB"
      effective_cache_size: "1GB"
      maintenance_work_mem: "64MB"
      checkpoint_completion_target: "0.9"
      wal_buffers: "16MB"
      default_statistics_target: "100"
      random_page_cost: "1.1"
      effective_io_concurrency: "200"
    
  bootstrap:
    initdb:
      database: swing_agent
      owner: swing_user
      secret:
        name: swing-postgres-credentials
  
  storage:
    size: 10Gi
    storageClass: fast-ssd  # Adjust to your storage class
  
  resources:
    requests:
      memory: "512Mi"
      cpu: "500m"
    limits:
      memory: "1Gi"
      cpu: "1000m"
  
  monitoring:
    enabled: true
  
  backup:
    barmanObjectStore:
      destinationPath: "s3://your-backup-bucket/cnpg"
      s3Credentials:
        accessKeyId:
          name: backup-credentials
          key: ACCESS_KEY_ID
        secretAccessKey:
          name: backup-credentials
          key: SECRET_ACCESS_KEY
        region:
          name: backup-credentials
          key: REGION
      wal:
        retention: "7d"
      data:
        retention: "30d"
```

### Create Database Credentials

```yaml
# postgres-credentials.yaml
apiVersion: v1
kind: Secret
metadata:
  name: swing-postgres-credentials
  namespace: default
type: kubernetes.io/basic-auth
data:
  username: c3dpbmdfdXNlcg==  # swing_user (base64)
  password: c3dpbmdfcGFzcw==  # swing_pass (base64)
```

### Create Backup Credentials (Optional)

```yaml
# backup-credentials.yaml
apiVersion: v1
kind: Secret
metadata:
  name: backup-credentials
  namespace: default
data:
  ACCESS_KEY_ID: <base64-encoded-access-key>
  SECRET_ACCESS_KEY: <base64-encoded-secret-key>
  REGION: <base64-encoded-region>
```

### Deploy the Cluster

```bash
# Create secrets
kubectl apply -f postgres-credentials.yaml
kubectl apply -f backup-credentials.yaml  # If using backups

# Create cluster
kubectl apply -f cnpg-cluster.yaml

# Monitor cluster creation
kubectl get cluster swing-postgres -w

# Check pods
kubectl get pods -l cnpg.io/cluster=swing-postgres
```

## SwingAgent Configuration

### Environment Variables

Configure SwingAgent to use the CNPG cluster:

```bash
# CNPG Configuration
export SWING_DB_TYPE=cnpg
export CNPG_CLUSTER_NAME=swing-postgres
export CNPG_NAMESPACE=default
export CNPG_SERVICE_TYPE=rw  # Use 'ro' for read-only

# Database Configuration
export SWING_DB_NAME=swing_agent
export SWING_DB_USER=swing_user
export SWING_DB_PASSWORD=swing_pass

# Optional: SSL Configuration
export CNPG_SSL_MODE=require
export CNPG_CONNECT_TIMEOUT=10
export CNPG_APP_NAME=swing-agent

# Optional: Connection Pooling
export SWING_DB_POOL_SIZE=10
export SWING_DB_MAX_OVERFLOW=20
```

### Kubernetes ConfigMap

For production deployments, use ConfigMaps and Secrets:

```yaml
# swing-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: swing-config
  namespace: default
data:
  SWING_DB_TYPE: "cnpg"
  CNPG_CLUSTER_NAME: "swing-postgres"
  CNPG_NAMESPACE: "default"
  CNPG_SERVICE_TYPE: "rw"
  SWING_DB_NAME: "swing_agent"
  CNPG_SSL_MODE: "require"
  CNPG_CONNECT_TIMEOUT: "10"
  CNPG_APP_NAME: "swing-agent"
  SWING_DB_POOL_SIZE: "10"
  SWING_DB_MAX_OVERFLOW: "20"

---
apiVersion: v1
kind: Secret
metadata:
  name: swing-db-secret
  namespace: default
type: Opaque
data:
  SWING_DB_USER: c3dpbmdfdXNlcg==     # swing_user
  SWING_DB_PASSWORD: c3dpbmdfcGFzcw== # swing_pass
```

### SwingAgent Deployment

```yaml
# swing-agent-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: swing-agent
  namespace: default
spec:
  replicas: 1
  selector:
    matchLabels:
      app: swing-agent
  template:
    metadata:
      labels:
        app: swing-agent
    spec:
      containers:
      - name: swing-agent
        image: swing-agent:latest
        envFrom:
        - configMapRef:
            name: swing-config
        - secretRef:
            name: swing-db-secret
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        # Health checks
        livenessProbe:
          exec:
            command:
            - python
            - -c
            - "from swing_agent.database import get_database_info; print(get_database_info())"
          initialDelaySeconds: 30
          periodSeconds: 60
        readinessProbe:
          exec:
            command:
            - python
            - -c
            - "from swing_agent.database import get_database_config; get_database_config().engine.execute('SELECT 1')"
          initialDelaySeconds: 10
          periodSeconds: 30
```

## Usage Examples

### Basic Usage

```python
# SwingAgent automatically detects CNPG configuration
from swing_agent import SwingAgent

agent = SwingAgent(symbol="AMD")
signal = agent.analyze("AMD")
print(f"Generated signal: {signal}")
```

### Connection Information

```python
from swing_agent.database import get_database_info

info = get_database_info()
print(f"Database type: {info['type']}")
print(f"CNPG cluster: {info.get('cnpg_cluster')}")
print(f"Service type: {info.get('cnpg_service')}")
print(f"Namespace: {info.get('cnpg_namespace')}")
```

### Connection Health Check

```python
from swing_agent.database import get_database_config

def check_cnpg_connection():
    try:
        db_config = get_database_config()
        with db_config.get_session() as session:
            result = session.execute("SELECT version(), current_database(), current_user")
            row = result.fetchone()
            print(f"Connected to: {row[0]}")
            print(f"Database: {row[1]}")
            print(f"User: {row[2]}")
            return True
    except Exception as e:
        print(f"Connection failed: {e}")
        return False

check_cnpg_connection()
```

## SSL/TLS Configuration

### Using Kubernetes Secrets for Certificates

```yaml
# ssl-certificates.yaml
apiVersion: v1
kind: Secret
metadata:
  name: cnpg-ssl-certs
  namespace: default
type: Opaque
data:
  ca.crt: <base64-encoded-ca-certificate>
  client.crt: <base64-encoded-client-certificate>
  client.key: <base64-encoded-client-key>
```

### Mount Certificates in Deployment

```yaml
spec:
  template:
    spec:
      containers:
      - name: swing-agent
        # ... other config
        env:
        - name: CNPG_SSL_CA
          value: "/ssl/ca.crt"
        - name: CNPG_SSL_CERT
          value: "/ssl/client.crt"
        - name: CNPG_SSL_KEY
          value: "/ssl/client.key"
        volumeMounts:
        - name: ssl-certs
          mountPath: /ssl
          readOnly: true
      volumes:
      - name: ssl-certs
        secret:
          secretName: cnpg-ssl-certs
```

## Monitoring and Observability

### Database Connection Metrics

SwingAgent provides connection pool metrics for CNPG:

```python
from swing_agent.database import get_database_config

db_config = get_database_config()
engine = db_config.engine

# Connection pool stats
pool = engine.pool
print(f"Pool size: {pool.size()}")
print(f"Checked in: {pool.checkedin()}")
print(f"Checked out: {pool.checkedout()}")
print(f"Overflow: {pool.overflow()}")
```

### Logging Configuration

Enable detailed logging for troubleshooting:

```bash
export SWING_DB_ECHO=true  # Enable SQL query logging
export CNPG_APP_NAME=swing-agent-debug  # Custom application name
```

## Migration from SQLite

Migrate existing SQLite data to CNPG:

```python
from swing_agent.migrate import migrate_to_external_db
from swing_agent.database import create_cnpg_url

# Ensure CNPG environment variables are set
import os
os.environ["SWING_DB_TYPE"] = "cnpg"
os.environ["CNPG_CLUSTER_NAME"] = "swing-postgres"
# ... other env vars

# Get CNPG URL
cnpg_url = create_cnpg_url()

# Migrate data
migrate_to_external_db(
    sqlite_path="data/swing_agent.sqlite",
    external_url=cnpg_url
)
```

## Troubleshooting

### Common Issues

1. **Connection Timeouts**
   ```bash
   export CNPG_CONNECT_TIMEOUT=30
   export SWING_DB_POOL_TIMEOUT=60
   ```

2. **Service Discovery Issues**
   - Verify CNPG cluster is running: `kubectl get cluster swing-postgres`
   - Check service endpoints: `kubectl get endpoints swing-postgres-rw`
   - Verify DNS resolution from pod

3. **SSL Certificate Issues**
   - Ensure certificates are properly mounted
   - Check certificate validity
   - Verify SSL mode configuration

4. **Permission Issues**
   ```sql
   -- Connect to CNPG cluster directly
   kubectl exec -it swing-postgres-1 -- psql -U postgres
   
   -- Grant permissions
   GRANT ALL PRIVILEGES ON DATABASE swing_agent TO swing_user;
   GRANT ALL ON SCHEMA public TO swing_user;
   ```

### Debug Commands

```bash
# Check cluster status
kubectl describe cluster swing-postgres

# View cluster logs
kubectl logs -l cnpg.io/cluster=swing-postgres

# Connect to primary pod
kubectl exec -it swing-postgres-1 -- psql -U swing_user -d swing_agent

# Test connectivity from SwingAgent pod
kubectl exec -it <swing-agent-pod> -- python -c "
from swing_agent.database import get_database_config
db = get_database_config()
with db.get_session() as s:
    result = s.execute('SELECT 1')
    print('Connection successful!')
"
```

## Performance Tuning

### CNPG Cluster Optimization

```yaml
spec:
  postgresql:
    parameters:
      # Connection settings
      max_connections: "200"
      shared_buffers: "512MB"        # 25% of RAM
      effective_cache_size: "1536MB" # 75% of RAM
      
      # Write performance
      wal_buffers: "16MB"
      checkpoint_completion_target: "0.9"
      checkpoint_timeout: "10min"
      
      # Query performance
      random_page_cost: "1.1"        # For SSD storage
      effective_io_concurrency: "200"
      default_statistics_target: "100"
      
      # Memory settings
      work_mem: "4MB"
      maintenance_work_mem: "128MB"
```

### SwingAgent Connection Pool Tuning

```bash
# Production settings
export SWING_DB_POOL_SIZE=20
export SWING_DB_MAX_OVERFLOW=30
export SWING_DB_POOL_TIMEOUT=60
export SWING_DB_POOL_RECYCLE=1800  # 30 minutes
```

## Best Practices

1. **Use read-only replicas** for read-heavy workloads
2. **Enable connection pooling** with appropriate pool sizes
3. **Monitor connection usage** and adjust pool settings
4. **Use SSL/TLS** for encrypted connections
5. **Implement proper backup strategies** with CNPG's built-in backup features
6. **Set resource limits** for both CNPG cluster and SwingAgent pods
7. **Use dedicated namespaces** for production deployments
8. **Monitor database performance** and optimize queries

## Support

For CNPG-specific issues:
- [CNPG Documentation](https://cloudnative-pg.io/documentation/)
- [CNPG GitHub](https://github.com/cloudnative-pg/cloudnative-pg)

For SwingAgent with CNPG integration:
- Check database connectivity with `scripts/db_info.py`
- Enable debug logging with `SWING_DB_ECHO=true`
- Review connection pool metrics
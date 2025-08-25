# CNPG Kubernetes Manifests

This directory contains Kubernetes manifests for deploying SwingAgent with CNPG (CloudNativePG) PostgreSQL.

## Files

- `cluster.yaml` - CNPG PostgreSQL cluster definition
- `secrets.yaml` - Database credentials and backup configuration
- `config.yaml` - SwingAgent configuration and database connection settings
- `deployment.yaml` - SwingAgent application deployment with health checks
- `monitoring.yaml` - Custom PostgreSQL monitoring queries
- `deploy.sh` - Automated deployment script

## Quick Deployment

1. **Install CNPG operator** (if not already installed):
   ```bash
   kubectl apply -f https://raw.githubusercontent.com/cloudnative-pg/cloudnative-pg/release-1.21/releases/cnpg-1.21.0.yaml
   ```

2. **Configure secrets** (edit `secrets.yaml` with your credentials):
   ```bash
   # Edit database password and backup credentials
   kubectl create secret generic swing-postgres-credentials \
     --from-literal=username=swing_user \
     --from-literal=password=your_secure_password
   ```

3. **Deploy everything**:
   ```bash
   chmod +x deploy.sh
   ./deploy.sh
   ```

## Manual Deployment

Deploy components in order:

```bash
# 1. Secrets and configuration
kubectl apply -f secrets.yaml
kubectl apply -f monitoring.yaml

# 2. PostgreSQL cluster
kubectl apply -f cluster.yaml

# 3. Wait for cluster to be ready
kubectl wait --for=condition=Ready cluster/swing-postgres --timeout=600s

# 4. Application configuration and deployment
kubectl apply -f config.yaml
kubectl apply -f deployment.yaml
```

## Customization

### Storage Class
Edit `cluster.yaml` to use your preferred storage class:
```yaml
storage:
  size: 20Gi
  storageClass: your-fast-storage-class
```

### Backup Configuration
Configure S3 backup in `secrets.yaml` and `cluster.yaml`:
```yaml
# In secrets.yaml - add your S3 credentials
data:
  ACCESS_KEY_ID: <base64-encoded-key>
  SECRET_ACCESS_KEY: <base64-encoded-secret>
  REGION: <base64-encoded-region>

# In cluster.yaml - update backup destination
backup:
  barmanObjectStore:
    destinationPath: "s3://your-backup-bucket/swing-postgres"
```

### Resource Limits
Adjust resources in both `cluster.yaml` and `deployment.yaml` based on your needs:
```yaml
resources:
  requests:
    memory: "1Gi"
    cpu: "500m"
  limits:
    memory: "2Gi"
    cpu: "1000m"
```

## Monitoring

The deployment includes:
- PostgreSQL metrics via CNPG built-in monitoring
- Custom SwingAgent-specific queries in `monitoring.yaml`
- Health checks for both database and application

Access metrics:
```bash
# Port forward to access metrics
kubectl port-forward svc/swing-postgres-metrics 9187:9187

# View custom queries
kubectl get configmap swing-postgres-queries -o yaml
```

## Troubleshooting

### Check cluster status
```bash
kubectl get cluster swing-postgres
kubectl describe cluster swing-postgres
```

### View logs
```bash
# CNPG cluster logs
kubectl logs -l cnpg.io/cluster=swing-postgres

# SwingAgent logs
kubectl logs -l app=swing-agent
```

### Test connectivity
```bash
# From SwingAgent pod
kubectl exec -it deployment/swing-agent -- python -c "
from swing_agent.database import get_database_info
print(get_database_info())
"

# Direct PostgreSQL access
kubectl exec -it swing-postgres-1 -- psql -U swing_user swing_agent
```

### Debug connection issues
```bash
# Check service endpoints
kubectl get endpoints swing-postgres-rw
kubectl get endpoints swing-postgres-ro

# Check DNS resolution
kubectl run debug --image=busybox:1.28 --rm -it --restart=Never -- nslookup swing-postgres-rw.default.svc.cluster.local
```

## Scaling

### Horizontal scaling (read replicas)
Edit `cluster.yaml`:
```yaml
spec:
  instances: 5  # Increase number of instances
```

### Vertical scaling (resources)
Update resources and apply:
```bash
kubectl patch cluster swing-postgres --type='merge' -p='{"spec":{"resources":{"limits":{"memory":"4Gi","cpu":"2000m"}}}}'
```

### Application scaling
```bash
kubectl scale deployment swing-agent --replicas=3
```

## Security

### SSL/TLS
Enable SSL certificates by uncommenting sections in `deployment.yaml` and creating certificate secrets:
```bash
kubectl create secret tls cnpg-ssl-certs \
  --cert=client.crt \
  --key=client.key \
  --dry-run=client -o yaml | kubectl apply -f -
```

### Network Policies
Apply network policies to restrict traffic:
```bash
# Allow only SwingAgent to access PostgreSQL
kubectl apply -f - <<EOF
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: swing-postgres-netpol
spec:
  podSelector:
    matchLabels:
      cnpg.io/cluster: swing-postgres
  policyTypes:
  - Ingress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: swing-agent
    ports:
    - protocol: TCP
      port: 5432
EOF
```
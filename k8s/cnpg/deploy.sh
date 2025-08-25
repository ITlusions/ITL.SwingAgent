#!/bin/bash

# CNPG SwingAgent Deployment Script
# This script deploys SwingAgent with CNPG PostgreSQL on Kubernetes

set -e

NAMESPACE="${NAMESPACE:-default}"
CLUSTER_NAME="${CLUSTER_NAME:-swing-postgres}"
APP_NAME="${APP_NAME:-swing-agent}"

echo "🚀 Deploying SwingAgent with CNPG in namespace: $NAMESPACE"

# Check if CNPG operator is installed
if ! kubectl get crd clusters.postgresql.cnpg.io >/dev/null 2>&1; then
    echo "❌ CNPG operator not found. Installing..."
    kubectl apply -f https://raw.githubusercontent.com/cloudnative-pg/cloudnative-pg/release-1.21/releases/cnpg-1.21.0.yaml
    echo "⏳ Waiting for CNPG operator to be ready..."
    kubectl wait --for=condition=Available deployment/cnpg-controller-manager -n cnpg-system --timeout=300s
    echo "✅ CNPG operator installed successfully"
fi

# Create namespace if it doesn't exist
kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -

# Deploy secrets first
echo "📦 Deploying database secrets..."
kubectl apply -f secrets.yaml -n $NAMESPACE

# Deploy monitoring configuration
echo "📊 Deploying monitoring configuration..."
kubectl apply -f monitoring.yaml -n $NAMESPACE

# Deploy PostgreSQL cluster
echo "🐘 Deploying PostgreSQL cluster..."
kubectl apply -f cluster.yaml -n $NAMESPACE

# Wait for cluster to be ready
echo "⏳ Waiting for PostgreSQL cluster to be ready..."
kubectl wait --for=condition=Ready cluster/$CLUSTER_NAME -n $NAMESPACE --timeout=600s

# Check cluster status
echo "📊 Cluster status:"
kubectl get cluster $CLUSTER_NAME -n $NAMESPACE

# Deploy application configuration
echo "⚙️ Deploying application configuration..."
kubectl apply -f config.yaml -n $NAMESPACE

# Deploy SwingAgent application
echo "🔄 Deploying SwingAgent application..."
kubectl apply -f deployment.yaml -n $NAMESPACE

# Wait for deployment to be ready
echo "⏳ Waiting for SwingAgent deployment to be ready..."
kubectl wait --for=condition=Available deployment/$APP_NAME -n $NAMESPACE --timeout=300s

# Display deployment status
echo "📊 Deployment status:"
kubectl get pods,services,hpa -l app=$APP_NAME -n $NAMESPACE

# Show connection information
echo "🔗 Connection information:"
echo "Database cluster: $CLUSTER_NAME"
echo "Read-write service: $CLUSTER_NAME-rw.$NAMESPACE.svc.cluster.local:5432"
echo "Read-only service: $CLUSTER_NAME-ro.$NAMESPACE.svc.cluster.local:5432"

# Test database connectivity
echo "🧪 Testing database connectivity..."
POD_NAME=$(kubectl get pods -l app=$APP_NAME -n $NAMESPACE -o jsonpath='{.items[0].metadata.name}')
if kubectl exec $POD_NAME -n $NAMESPACE -- python -c "
from swing_agent.database import get_database_info, get_database_config
info = get_database_info()
print(f'Connected to: {info[\"type\"]} - {info.get(\"cnpg_cluster\", \"Unknown\")}')
print(f'Database: {info[\"database\"]}')
print(f'Is CNPG: {info[\"is_cnpg\"]}')

# Test basic query
db = get_database_config()
with db.get_session() as session:
    result = session.execute('SELECT version()').fetchone()
    print(f'PostgreSQL version: {result[0][:50]}...')
"; then
    echo "✅ Database connectivity test passed!"
else
    echo "❌ Database connectivity test failed!"
    exit 1
fi

echo "🎉 Deployment completed successfully!"
echo ""
echo "Next steps:"
echo "1. Check logs: kubectl logs -l app=$APP_NAME -n $NAMESPACE"
echo "2. Port forward: kubectl port-forward svc/$APP_NAME-service -n $NAMESPACE 8080:80"
echo "3. Monitor cluster: kubectl get cluster $CLUSTER_NAME -n $NAMESPACE -w"
echo "4. Access PostgreSQL: kubectl exec -it $CLUSTER_NAME-1 -n $NAMESPACE -- psql -U swing_user swing_agent"
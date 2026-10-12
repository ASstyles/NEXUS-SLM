#!/usr/bin/env bash
# NEXUS-SLM Local Demo Launch Script for Unix/Linux/macOS
set -eo pipefail

echo "=========================================================="
echo "   NEXUS-SLM — Neural EXecution Support Intelligence     "
echo "   Local Demonstration Launcher (Unix/Linux/macOS)       "
echo "=========================================================="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${WORKSPACE_ROOT}"

# Activate virtual environment
if [ -f "${WORKSPACE_ROOT}/.venv/bin/activate" ]; then
    source "${WORKSPACE_ROOT}/.venv/bin/activate"
elif [ -f "${WORKSPACE_ROOT}/venv/bin/activate" ]; then
    source "${WORKSPACE_ROOT}/venv/bin/activate"
else
    echo "Warning: Python virtual environment not found. Using system python3."
fi

PYTHON_CMD="python3"
if command -v python &> /dev/null; then
    PYTHON_CMD="python"
fi

echo ""
echo "[1/3] Starting FastAPI Inference Service on port 8000..."
${PYTHON_CMD} -m uvicorn api.main:app --host 127.0.0.1 --port 8000 &
BACKEND_PID=$!

cleanup() {
    echo ""
    echo "Shutting down background demo processes..."
    kill ${BACKEND_PID} 2>/dev/null || true
    if [ ! -z "${FRONTEND_PID}" ]; then
        kill ${FRONTEND_PID} 2>/dev/null || true
    fi
    echo "Demo shut down cleanly."
    exit 0
}
trap cleanup SIGINT SIGTERM EXIT

# Wait for backend health
echo "Waiting for FastAPI service to initialize..."
MAX_RETRIES=15
RETRY=0
BACKEND_READY=false

while [ ${RETRY} -lt ${MAX_RETRIES} ]; do
    sleep 1
    if curl -s http://127.0.0.1:8000/health | grep -q '"status":"healthy"'; then
        BACKEND_READY=true
        echo "FastAPI Service is ONLINE"
        break
    fi
    RETRY=$((RETRY + 1))
done

if [ "${BACKEND_READY}" = false ]; then
    echo "Error: FastAPI backend failed to start within ${MAX_RETRIES} seconds."
    exit 1
fi

echo ""
echo "[2/3] Starting Next.js Dashboard on port 3000..."
cd "${WORKSPACE_ROOT}/dashboard"
npm run dev &
FRONTEND_PID=$!
cd "${WORKSPACE_ROOT}"

sleep 4

echo ""
echo "[3/3] DEMONSTRATION READY!"
echo "=========================================================="
echo "  Interactive UI: http://localhost:3000"
echo "  Swagger API Docs: http://localhost:8000/docs"
echo "  Health Endpoint: http://localhost:8000/health"
echo "=========================================================="
echo "Press Ctrl+C to terminate demo services..."

wait ${FRONTEND_PID}

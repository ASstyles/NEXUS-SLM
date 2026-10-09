import requests

base = "http://localhost:8000"

print("--- Testing GET Endpoints ---")
for ep in ["/health", "/model", "/metrics", "/version", "/labels", "/registry", "/benchmark", "/error-analysis"]:
    r = requests.get(base + ep)
    print(f"{ep}: status={r.status_code}")
    assert r.status_code == 200, f"Expected 200 for {ep}, got {r.status_code}"

print("\n--- Testing POST Edge Cases ---")
r_empty = requests.post(base + "/predict", json={"text": ""})
print(f"Empty text: status={r_empty.status_code} (Expected 422)")
assert r_empty.status_code == 422

r_missing = requests.post(base + "/predict", json={})
print(f"Missing text field: status={r_missing.status_code} (Expected 422)")
assert r_missing.status_code == 422

r_long = requests.post(base + "/predict", json={"text": "A" * 5000})
print(f"Overly long text (>4000): status={r_long.status_code} (Expected 422)")
assert r_long.status_code == 422

r_valid = requests.post(base + "/predict", json={"text": "Mera payment kat gaya but order confirm nahi hua."})
print(f"Valid ticket: status={r_valid.status_code}")
assert r_valid.status_code == 200
data = r_valid.json()
print("Prediction response:", data)
assert data["category"] == "payment_failure"
assert data["language"] == "hinglish"
assert data["requires_human"] is True
assert data["confidence"] > 0.0
assert "latency_ms" in data
print("\nALL LIVE API AUDIT TESTS PASSED!")

import urllib.request
import json
import subprocess
import sys

class CustomRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return urllib.request.Request(newurl, headers={"User-Agent": "Python/NEXUS-Audit"})

def get_token():
    proc = subprocess.Popen(
        ["git", "credential", "fill"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    stdout, _ = proc.communicate(input="protocol=https\nhost=github.com\n")
    for line in stdout.splitlines():
        if line.startswith("password="):
            return line.split("=", 1)[1].strip()
    return None

def main():
    token = get_token()
    headers = {"User-Agent": "Python/NEXUS-SLM-Audit"}
    if token:
        headers["Authorization"] = f"token {token}"
    
    # Get latest run
    url = "https://api.github.com/repos/ASstyles/NEXUS-SLM/actions/runs"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as resp:
        runs = json.loads(resp.read().decode()).get("workflow_runs", [])
        latest_run = runs[0]
        run_id = latest_run["id"]
        print(f"Latest Run ID: {run_id} ({latest_run['head_sha'][:7]}) Status: {latest_run['status']} Conclusion: {latest_run['conclusion']}")
    
    # Get container job
    jobs_url = latest_run["jobs_url"]
    with urllib.request.urlopen(urllib.request.Request(jobs_url, headers=headers)) as jresp:
        jobs = json.loads(jresp.read().decode())["jobs"]
        container_job = [j for j in jobs if "Container" in j["name"]][0]
        job_id = container_job["id"]
        print(f"Container Job ID: {job_id} Status: {container_job['status']} Conclusion: {container_job['conclusion']}")

    opener = urllib.request.build_opener(CustomRedirectHandler)
    req2 = urllib.request.Request(f"https://api.github.com/repos/ASstyles/NEXUS-SLM/actions/jobs/{job_id}/logs", headers=headers)
    try:
        logs = opener.open(req2).read().decode("utf-8", errors="ignore")
        print("\n--- RECENT LOG LINES ---")
        for line in logs.splitlines()[-40:]:
            print(line)
    except Exception as e:
        print(f"Could not retrieve logs: {e}")

if __name__ == "__main__":
    main()

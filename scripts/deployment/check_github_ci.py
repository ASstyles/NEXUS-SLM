import subprocess
import urllib.request
import json

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
    
    url = "https://api.github.com/repos/ASstyles/NEXUS-SLM/actions/runs"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode())
        runs = data.get("workflow_runs", [])
        if not runs:
            print("No workflow runs found.")
            return
        latest = runs[0]
        print(f"Latest Run ID: {latest['id']} | Status: {latest['status']} | Conclusion: {latest['conclusion']} | Commit: {latest['head_sha']}")
        
        # Query jobs for latest run
        jobs_url = latest["jobs_url"]
        jobs_req = urllib.request.Request(jobs_url, headers=headers)
        with urllib.request.urlopen(jobs_req) as jresp:
            jdata = json.loads(jresp.read().decode())
            for job in jdata.get("jobs", []):
                print(f"\n  Job: {job['name']} | Status: {job['status']} | Conclusion: {job['conclusion']}")
                for step in job.get("steps", []):
                    status_str = step["status"]
                    if step.get("conclusion"):
                        status_str += f" ({step['conclusion']})"
                    print(f"    - {step['name']}: {status_str}")

if __name__ == "__main__":
    main()

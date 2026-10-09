import subprocess
import urllib.request
import json
import sys

def main():
    try:
        proc = subprocess.Popen(
            ["git", "credential", "fill"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        stdout, _ = proc.communicate(input="protocol=https\nhost=github.com\n")
        token = None
        for line in stdout.splitlines():
            if line.startswith("password="):
                token = line.split("=", 1)[1].strip()
                break
        
        headers = {"User-Agent": "Python/NEXUS-SLM-Audit"}
        if token:
            headers["Authorization"] = f"token {token}"
            print("Found credential manager token. Querying GitHub API...")
        else:
            print("No credential token returned from git credential manager.")

        url = "https://api.github.com/repos/ASstyles/NEXUS-SLM/actions/runs"
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode())
            runs = data.get("workflow_runs", [])
            print(f"Total workflow runs found: {len(runs)}")
            for r in runs:
                print(f"ID: {r.get('id')} | Name: {r.get('name')} | Status: {r.get('status')} | Conclusion: {r.get('conclusion')} | Commit: {r.get('head_sha')}")
    except Exception as e:
        print(f"Error querying GitHub API: {e}")

if __name__ == "__main__":
    main()

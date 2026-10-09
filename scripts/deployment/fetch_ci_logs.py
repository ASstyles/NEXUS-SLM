import urllib.request
import json
import subprocess

class CustomRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Drop the Authorization header on redirect to Azure Blob storage
        return urllib.request.Request(newurl, headers={"User-Agent": "Python/NEXUS-Audit"})

def main():
    proc = subprocess.Popen(['git', 'credential', 'fill'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    token = [l.split('=', 1)[1].strip() for l in proc.communicate(input='protocol=https\nhost=github.com\n')[0].splitlines() if l.startswith('password=')][0]
    
    req = urllib.request.Request('https://api.github.com/repos/ASstyles/NEXUS-SLM/actions/runs/37968630491/jobs', headers={'User-Agent': 'Python', 'Authorization': f'token {token}'})
    jobs = json.loads(urllib.request.urlopen(req).read())['jobs']
    container_job = [j for j in jobs if 'Container' in j['name']][0]
    job_id = container_job['id']
    print(f"Container Job ID: {job_id}, conclusion: {container_job['conclusion']}")
    
    opener = urllib.request.build_opener(CustomRedirectHandler)
    req2 = urllib.request.Request(f'https://api.github.com/repos/ASstyles/NEXUS-SLM/actions/jobs/{job_id}/logs', headers={'User-Agent': 'Python', 'Authorization': f'token {token}'})
    logs = opener.open(req2).read().decode('utf-8', errors='ignore')
    
    print("\n--- LAST 80 LINES OF LOG ---")
    for line in logs.splitlines()[-80:]:
        print(line)

if __name__ == '__main__':
    main()

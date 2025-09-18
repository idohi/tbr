# GitHub Actions Runner – Service Commands (macOS)

## Why run the runner on your MacBook?
- GitHub gives only **2,000 free Actions minutes per month** for private repos.
- macOS runners on GitHub are very expensive (10× minutes).
- By using your **MacBook as a self-hosted runner**, you:
  - Avoid using up the free minutes.
  - Run jobs locally for free.
  - Keep full control over the environment.

When your project becomes **public**, GitHub gives **unlimited free minutes** (on Linux).
At that point, you can switch back to GitHub-hosted runners if you prefer.

---

## How to switch from GitHub-hosted to your MacBook
In your workflow file (e.g., `.github/workflows/ci.yml`), change this line:

```yaml
runs-on: ubuntu-latest
```

to:

```yaml
runs-on: self-hosted
```

This tells GitHub to run jobs on your **MacBook Pro self-hosted runner** instead of GitHub’s servers.

---

## How to switch back to GitHub-hosted
If you want jobs to run on GitHub’s servers again (like at the beginning):

1. Open your workflow file.
2. Change:
   ```yaml
   runs-on: self-hosted
   ```
   back to:
   ```yaml
   runs-on: ubuntu-latest
   ```
   (or `macos-latest` / `windows-latest` if needed).

That’s it — jobs will stop running on your MacBook and go back to GitHub-hosted runners.

---

## Side-by-side YAML comparison

**GitHub-hosted runner (default)**
```yaml
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: pytest
```

**MacBook self-hosted runner**
```yaml
jobs:
  build:
    runs-on: self-hosted
    steps:
      - uses: actions/checkout@v4
      - run: pytest
```

---

## Useful Commands for Self-Hosted Service

Run these inside your `actions-runner` folder:

### Install the runner as a service
```bash
./svc.sh install
```
Registers the runner as a macOS service (so it runs in the background).

### Start the service
```bash
./svc.sh start
```
Starts the runner in the background.

### Stop the service
```bash
./svc.sh stop
```
Stops the runner if it’s running.

### Check the status
```bash
./svc.sh status
```
Shows whether the runner service is running or stopped.

### Uninstall the service
```bash
./svc.sh uninstall
```
Removes the runner’s service registration (it will no longer auto-start).

---

## Tips
- Use **service mode** if you want the runner always available.
- Use **`./run.sh` manually** if you only want it active sometimes.
- Switch between **self-hosted** and **GitHub-hosted** runners anytime by editing `runs-on`.

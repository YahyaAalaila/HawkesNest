# Docker Deployment

To avoid "works on my machine" issues or platform-specific file system errors (like those encountered with Windows WSL 2), you can run HawkesNest entirely inside a Docker container. This guarantees a consistent Linux environment with all dependencies pre-installed.

## 1. Build the Docker Image

From the root of the project (where the `Dockerfile` is located), build the Docker image and tag it as `hawkesnest`:

```bash
docker build -t hawkesnest .
```

This command will:
- Download a lightweight Python 3.10 image.
- Install the fast package manager `uv`.
- Copy the HawkesNest source code into the container.
- Install the package and all its `[dev]` dependencies.

## 2. Run the Container

Once built, you can run the container. To ensure that the files (like datasets and plots) generated inside the container are saved back to your host machine, you should mount your local `outputs` directory to the container's `/app/outputs` directory.

### Interactive Bash Shell
To enter an interactive terminal inside the container where you can run Python scripts (like `main.py`) or the CLI commands:

**On Linux / macOS:**
```bash
docker run -it --rm -v $(pwd)/outputs:/app/outputs hawkesnest bash
```

**On Windows (PowerShell):**
```powershell
docker run -it --rm -v ${PWD}/outputs:/app/outputs hawkesnest bash
```

Once inside, you can run any of the quickstart commands:
```bash
# Generate a dataset using the CLI
python -m hawkesnest.cli generate entanglement --level L2 --n-events 50 --out outputs/entanglement_l2

# Or run your custom main.py
python main.py
```

### Running Commands Directly
You can also run specific commands directly without entering the bash shell:

```bash
docker run --rm -v $(pwd)/outputs:/app/outputs hawkesnest python main.py
```

## Summary
Using Docker completely bypasses the need for local virtual environments (`.venv`), Python version mismatches, and WSL interop permission errors.

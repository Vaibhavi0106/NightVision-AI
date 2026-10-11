import os
import sys
import subprocess
import uvicorn

def main():
    print("=========================================================")
    print("NightVision AI -- Deploying Production Studio Server")
    print("=========================================================")

    frontend_dir = os.path.join(os.path.dirname(__file__), "frontend")
    dist_dir = os.path.join(frontend_dir, "dist")

    if not os.path.exists(dist_dir):
        print("Building React Frontend bundle...")
        subprocess.run("npm run build", shell=True, cwd=frontend_dir, check=True)

    print("Frontend bundle verified at:", dist_dir)
    print("Starting Unified CUDA FastAPI Production Server on http://0.0.0.0:8000")
    print("=========================================================")

    backend_dir = os.path.join(os.path.dirname(__file__), "backend")
    sys.path.insert(0, backend_dir)

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False, workers=1)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Script to set up cross-environment module access via symlinks and PYTHONPATH
"""
import os
import sys

# container
# Add all app modules and site-packages to each environment via pth files
app_paths = ["/app/runner", "/app/server", "/app/evaluation"]
venv_paths = [
    "/opt/venv/runner/lib/python3.12/site-packages",
    "/opt/venv/server/lib/python3.12/site-packages",
    "/opt/venv/evaluation/lib/python3.12/site-packages",
]

# Add site-packages from each environment to the others
shared_paths = [
    "/opt/venv/runner/lib/python3.12/site-packages",
    "/opt/venv/server/lib/python3.12/site-packages",
    "/opt/venv/evaluation/lib/python3.12/site-packages",
]

# local
# app_paths = ["./runner", "./server", "./evaluation"]
# venv_paths = [
#     "./runner/venv/lib/python3.12/site-packages",
#     "./server/venv/lib/python3.12/site-packages",
#     "./evaluation/venv/lib/python3.9/site-packages",
# ]

for venv_site_packages in venv_paths:
    if os.path.exists(venv_site_packages):
        pth_file = os.path.join(venv_site_packages, "cross_env_access.pth")
        with open(pth_file, "w", encoding="utf-8") as f:
            # Add app paths first
            for app_path in app_paths:
                f.write(f"{app_path}\n")
            # Then add shared site-packages paths
            for shared_path in shared_paths:
                if shared_path != venv_site_packages:  # Don't add own site-packages
                    f.write(f"{shared_path}\n")
        print(f"Created {pth_file}")

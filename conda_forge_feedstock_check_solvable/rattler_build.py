import atexit
import os
import subprocess
import tempfile
import uuid

import yaml

from conda_forge_feedstock_check_solvable.utils import clean_rattler_cache, print_debug
from conda_forge_feedstock_check_solvable.virtual_packages import (
    virtual_package_repodata,
)

atexit.register(clean_rattler_cache)


def run_rattler_build(command):
    try:
        # Run the command and capture output
        print_debug("Running: %s", " ".join(command))
        result = subprocess.run(command, check=False, capture_output=True, text=True)

        # Get the status code
        status_code = result.returncode

        # Get stdout and stderr
        stdout = result.stdout.strip()
        stderr = result.stderr.strip()

        return status_code, stdout, stderr
    except Exception as e:
        return -1, "", str(e)


def invoke_rattler_build(
    recipe_dir: str, channels, build_platform, host_platform, variants
) -> (bool, str):
    # this is OK since there is an lru cache
    virtual_package_repo_url = virtual_package_repodata()
    # create a temporary file and dump the variants as YAML
    with tempfile.TemporaryDirectory() as tmpdir:
        variants_file_name = os.path.join(tmpdir, str(uuid.uuid4()) + ".yaml")
        with open(variants_file_name, "w") as fp:
            channel_sources = variants.get("channel_sources", [])
            # Add virtual package repo URL to channel sources
            if channel_sources:
                variants["channel_sources"] = [
                    source + f",{virtual_package_repo_url}"
                    for source in channel_sources
                ]

            yaml.dump(
                {k: v for k, v in variants.items()},
                fp,
            )

        channels_args = []
        if not channel_sources:
            for c in channels:
                channels_args.extend(["-c", c])

            channels_args.extend(["-c", virtual_package_repo_url])

        args = (
            ["rattler-build", "build", "--recipe", recipe_dir]
            + channels_args
            + ["--target-platform", host_platform]
            + ["--build-platform", build_platform]
            + ["-m", variants_file_name]
            + ["--render-only", "--with-solve"]
        )

        status, out, err = run_rattler_build(args)
        out = f"Command: {' '.join(args)}\nLogs:\n{out}"

        if status == 0:
            return True, ""
        else:
            return False, out + err

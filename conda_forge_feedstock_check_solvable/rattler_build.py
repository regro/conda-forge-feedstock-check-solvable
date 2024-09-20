import os
import subprocess

from conda_forge_feedstock_check_solvable.virtual_packages import (
    virtual_package_repodata,
)


def run_rattler_build(command):
    try:
        # Run the command and capture output
        print(" ".join(command))
        result = subprocess.run(
            command, shell=True, check=False, capture_output=True, text=True
        )

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
    print("invoke_rattler_build")
    virtual_package_repo_url = virtual_package_repodata()

    channels_args = []
    for c in channels:
        channels_args.extend(["-c", c])

    channels_args.extend(["-c", virtual_package_repo_url])

    variants_args = []
    # for v in variants:
    # variants_args.extend(["-m", v])
    args = (
        ["rattler-build", "build", "--recipe", recipe_dir]
        + channels_args
        + ["--target-platform", host_platform, "--build-platform", build_platform]
        + variants_args
        + ["--render-only", "--with-solve"]
    )
    print(" ".join(args))

    recipe = os.path.join(recipe_dir, "recipe.yaml")
    status, out, err = run_rattler_build(
        ["rattler-build", "build", "--recipe", recipe]  # + channels_args + \
        # ["--target-platform", host_platform, "--build-platform", build_platform] + \
        # variants_args + \
        # ["--render-only", "--with-solve"]
    )

    if status == 0:
        print(out, err)
        return True, ""
    else:
        return False, out + err

import yaml
import sys


def convert_conda_to_requirements(conda_file, output_file):
    with open(conda_file, "r") as f:
        conda_env = yaml.safe_load(f)

    requirements = []

    # System packages to exclude
    system_packages = {
        "_libgcc_mutex",
        "_openmp_mutex",
        "ca-certificates",
        "certifi",
        "ld_impl_linux-64",
        "libffi",
        "libgcc-ng",
        "libgomp",
        "libstdcxx-ng",
        "ncurses",
        "openssl",
        "readline",
        "sqlite",
        "tk",
        "xz",
        "zlib",
        "python",
        "pip",
        "wheel",
        "kenlm",  # Exclude Python and pip packages
    }

    # Add pip packages
    if "dependencies" in conda_env:
        for dep in conda_env["dependencies"]:
            if isinstance(dep, dict) and "pip" in dep:
                requirements.extend(dep["pip"])
            elif isinstance(dep, str) and not dep.startswith("_"):
                # Convert conda package to pip format
                pkg = dep.split("=")[0]
                if pkg not in system_packages:
                    if "=" in dep:
                        version = dep.split("=")[1]
                        requirements.append(f"{pkg}=={version}")
                    else:
                        requirements.append(pkg)

    # Write to requirements file
    with open(output_file, "w") as f:
        f.write("\n".join(requirements))


if __name__ == "__main__":
    convert_conda_to_requirements("conda_env_csnli.yml", "requirements_from_conda.txt")

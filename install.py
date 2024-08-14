import subprocess
import sys

def install_requirements(requirements_file):
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', requirements_file])
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")
        sys.exit(1)


def main():
    install_requirements("requirements.txt")


if __name__ == "__main__":
    main()
import re
import sys
import subprocess
from importlib import metadata

REQ_FILE = 'requirements.txt'

def parse_reqs(path):
    reqs = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            m = re.match(r'([^=<>!~]+)(?:[=<>!~]+(.+))?', line)
            if m:
                pkg = m.group(1).strip()
                ver = m.group(2).strip() if m.group(2) else None
                reqs.append((pkg, ver))
    return reqs


def is_installed(pkg):
    try:
        ver = metadata.version(pkg)
        return ver
    except metadata.PackageNotFoundError:
        return None


def install(pkg, ver=None):
    if ver:
        pkg_spec = f"{pkg}=={ver}"
    else:
        pkg_spec = pkg
    print(f"Installing {pkg_spec}...")
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', pkg_spec])


def main():
    reqs = parse_reqs(REQ_FILE)
    missing = []
    mismatch = []
    for pkg, ver in reqs:
        installed = is_installed(pkg)
        if not installed:
            missing.append((pkg, ver))
        else:
            if ver and installed != ver:
                mismatch.append((pkg, installed, ver))

    if not missing and not mismatch:
        print('All requirements present with matching versions.')
        return

    if missing:
        print('Missing packages:')
        for pkg, ver in missing:
            print(f'  - {pkg} (required {ver})')
    if mismatch:
        print('\nInstalled but version differs:')
        for pkg, inst, reqv in mismatch:
            print(f'  - {pkg}: installed {inst}, required {reqv}')

    # Install missing packages only (do not downgrade mismatched ones by default)
    for pkg, ver in missing:
        install(pkg, ver)

    print('\nDone. Re-run this script if you want to force install exact versions for mismatches.')

if __name__ == '__main__':
    main()

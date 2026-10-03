#!/bin/bash
# Double-click in Finder (or run in Terminal) to build LUMEN RAW for Apple silicon:
# tests, Metal / Core ML check, LUMEN RAW.app and the DMG. Needs Python 3.12 (python3.12 on PATH,
# or pass --python /path/to/python3.12). Logs: .publish/v<version>/macos/logs (about 15-25 minutes).
cd "$(dirname "$0")" || exit 1
bash tools/build_macos.sh "$@"
status=$?
echo
if [ $status -eq 0 ]; then echo "Build finished successfully."; else echo "Build FAILED - see .publish/v<version>/macos/logs/build.log"; fi
if [ -t 0 ]; then read -n 1 -s -r -p "Press any key to close"; echo; fi
exit $status

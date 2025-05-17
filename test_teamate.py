import os
import platform
import subprocess

print("OS:", os.name)
print("Platform:", platform.system())
print("Release:", platform.release())
print("WSL_ENV_VAR:", os.environ.get("WSL_DISTRO_NAME", "Not in WSL"))
print("DISPLAY:", os.environ.get("DISPLAY"))
print("PATH:", os.environ.get("PATH"))
print("WAYLAND_DISPLAY:", os.environ.get("WAYLAND_DISPLAY"))
print("XDG_RUNTIME_DIR:", os.environ.get("XDG_RUNTIME_DIR"))
print("DBUS_SESSION_BUS_ADDRESS:", os.environ.get("DBUS_SESSION_BUS_ADDRESS"))
subprocess.run(["glxinfo"])

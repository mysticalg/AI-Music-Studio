from __future__ import annotations

import json
import socket
import subprocess
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
HOST_EXE = REPO_ROOT / "build" / "native-vst3host" / "AIMusicStudioVSTHost_artefacts" / "Release" / "AI Music Studio VST Host.exe"


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class NativeVstHostBridge:
    def __init__(
        self,
        plugin_path: str | None = None,
        *,
        port: int | None = None,
        open_editor: bool = False,
        bridge_mode: bool = True,
    ) -> None:
        self.plugin_path = plugin_path
        self.port = port or _find_free_port()
        self.open_editor = open_editor
        self.bridge_mode = bool(bridge_mode)
        self.process: subprocess.Popen[str] | None = None

    def start(self, startup_timeout: float = 10.0) -> None:
        if not HOST_EXE.exists():
            raise FileNotFoundError(f"Native host executable not found: {HOST_EXE}")

        args = [str(HOST_EXE), "--port", str(self.port)]
        if self.bridge_mode:
            args.append("--bridge-mode")
        if self.plugin_path:
            args.extend(["--plugin", self.plugin_path])
        if self.open_editor:
            args.append("--open-editor")

        self.process = subprocess.Popen(args)
        self.wait_until_ready(timeout=startup_timeout)

    def wait_until_ready(self, timeout: float = 10.0) -> dict[str, Any]:
        deadline = time.time() + timeout
        last_error: Exception | None = None

        while time.time() < deadline:
            if self.process is not None and self.process.poll() is not None:
                raise RuntimeError(f"Native host exited early with code {self.process.returncode}")

            try:
                return self.command("status")
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                time.sleep(0.2)

        raise TimeoutError(f"Timed out waiting for native host on port {self.port}: {last_error}")

    def command(self, command: str, **payload: Any) -> dict[str, Any]:
        request = {"command": command, **payload}
        encoded = (json.dumps(request) + "\n").encode("utf-8")

        with socket.create_connection(("127.0.0.1", self.port), timeout=5.0) as sock:
            sock.sendall(encoded)
            sock.shutdown(socket.SHUT_WR)

            chunks: list[bytes] = []
            while True:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                chunks.append(chunk)

        raw = b"".join(chunks).decode("utf-8", errors="replace").strip()
        if not raw:
            raise RuntimeError("No response from native host")

        data = json.loads(raw)
        if not data.get("ok", False):
            raise RuntimeError(data.get("message", "Native host command failed"))

        return data

    def stop(self, timeout: float = 3.0) -> None:
        try:
            self.command("quit")
        except Exception:  # noqa: BLE001
            pass

        if self.process is None:
            return

        try:
            self.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=timeout)
        finally:
            self.process = None


__all__ = ["NativeVstHostBridge", "HOST_EXE"]

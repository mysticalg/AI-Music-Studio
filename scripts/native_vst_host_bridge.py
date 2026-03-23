from __future__ import annotations

import base64
import ctypes
import json
import os
import socket
import subprocess
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
HOST_EXE = REPO_ROOT / "build" / "native-vst3host" / "AIMusicStudioVSTHost_artefacts" / "Release" / "AI Music Studio VST Host.exe"
HOST_DLL_CANDIDATES = [
    REPO_ROOT / "build" / "native-vst3host" / "Release" / "AIMusicStudioVSTHostLib.dll",
    REPO_ROOT / "build" / "native-vst3host" / "Debug" / "AIMusicStudioVSTHostLib.dll",
    REPO_ROOT / "build" / "native-vst3host" / "RelWithDebInfo" / "AIMusicStudioVSTHostLib.dll",
    REPO_ROOT / "build" / "native-vst3host" / "MinSizeRel" / "AIMusicStudioVSTHostLib.dll",
]
_existing_host_dlls = [candidate for candidate in HOST_DLL_CANDIDATES if candidate.exists()]
HOST_DLL = (
    max(_existing_host_dlls, key=lambda candidate: candidate.stat().st_mtime_ns)
    if _existing_host_dlls
    else HOST_DLL_CANDIDATES[0]
)


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class _InProcessHostBackend:
    _dll: ctypes.WinDLL | None = None

    def __init__(
        self,
        plugin_path: str | None = None,
        *,
        open_editor: bool = False,
        sample_rate: int | None = None,
        buffer_size: int | None = None,
    ) -> None:
        self.plugin_path = plugin_path
        self.open_editor = bool(open_editor)
        self.sample_rate = int(sample_rate) if sample_rate else 0
        self.buffer_size = int(buffer_size) if buffer_size else 0
        self.handle: int | None = None
        self.process = self

    @classmethod
    def available(cls) -> bool:
        return HOST_DLL.exists()

    @classmethod
    def _loaded_dll(cls) -> ctypes.WinDLL:
        if cls._dll is None:
            if not HOST_DLL.exists():
                raise FileNotFoundError(f"Native host library not found: {HOST_DLL}")
            dll = ctypes.WinDLL(str(HOST_DLL))
            dll.aims_vst_host_create.argtypes = [
                ctypes.c_char_p,
                ctypes.c_int,
                ctypes.c_double,
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_int,
            ]
            dll.aims_vst_host_create.restype = ctypes.c_void_p
            dll.aims_vst_host_command.argtypes = [
                ctypes.c_void_p,
                ctypes.c_char_p,
                ctypes.c_char_p,
                ctypes.c_int,
            ]
            dll.aims_vst_host_command.restype = ctypes.c_int
            dll.aims_vst_host_destroy.argtypes = [ctypes.c_void_p]
            dll.aims_vst_host_destroy.restype = None
            cls._dll = dll
        return cls._dll

    def start(self, startup_timeout: float = 10.0) -> None:
        dll = self._loaded_dll()
        error_buffer = ctypes.create_string_buffer(4096)
        handle = dll.aims_vst_host_create(
            (self.plugin_path or "").encode("utf-8"),
            int(self.open_editor),
            float(self.sample_rate),
            int(self.buffer_size),
            error_buffer,
            len(error_buffer),
        )
        if not handle:
            detail = error_buffer.value.decode("utf-8", errors="replace").strip() or "Unknown host creation error"
            raise RuntimeError(detail)
        self.handle = int(handle)
        self.wait_until_ready(timeout=startup_timeout)

    def poll(self) -> int | None:
        return None if self.handle else 0

    def wait_until_ready(self, timeout: float = 10.0) -> dict[str, Any]:
        deadline = time.time() + timeout
        last_error: Exception | None = None
        while time.time() < deadline:
            try:
                return self.command("status")
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                time.sleep(0.05)
        raise TimeoutError(f"Timed out waiting for in-process host: {last_error}")

    def command(self, command: str, **payload: Any) -> dict[str, Any]:
        if not self.handle:
            raise RuntimeError("In-process host is not running")
        request = {"command": command, **payload}
        request_json = json.dumps(request, separators=(",", ":")).encode("utf-8")
        response_buffer = ctypes.create_string_buffer(4 * 1024 * 1024)
        ok = self._loaded_dll().aims_vst_host_command(
            ctypes.c_void_p(self.handle),
            request_json,
            response_buffer,
            len(response_buffer),
        )
        if ok == 0:
            raise RuntimeError(response_buffer.value.decode("utf-8", errors="replace").strip() or "Host command failed")
        raw = response_buffer.value.decode("utf-8", errors="replace").strip()
        if not raw:
            raise RuntimeError("No response from in-process host")
        data = json.loads(raw)
        if not data.get("ok", False):
            raise RuntimeError(data.get("message", "Native host command failed"))
        return data

    def stop(self, timeout: float = 3.0) -> None:
        _ = timeout
        if self.handle:
            try:
                self._loaded_dll().aims_vst_host_destroy(ctypes.c_void_p(self.handle))
            finally:
                self.handle = None


class _SubprocessHostBackend:
    def __init__(
        self,
        plugin_path: str | None = None,
        *,
        port: int | None = None,
        open_editor: bool = False,
        bridge_mode: bool = True,
        hidden: bool = True,
        sample_rate: int | None = None,
        buffer_size: int | None = None,
    ) -> None:
        self.plugin_path = plugin_path
        self.port = port or _find_free_port()
        self.open_editor = open_editor
        self.bridge_mode = bool(bridge_mode)
        self.hidden = bool(hidden)
        self.sample_rate = int(sample_rate) if sample_rate else 0
        self.buffer_size = int(buffer_size) if buffer_size else 0
        self.process: subprocess.Popen[str] | None = None
        self._socket: socket.socket | None = None
        self._recv_buffer = bytearray()
        self._command_timeout = 2.0
        self._last_command_at = 0.0

    def start(self, startup_timeout: float = 10.0) -> None:
        if not HOST_EXE.exists():
            raise FileNotFoundError(f"Native host executable not found: {HOST_EXE}")

        args = [str(HOST_EXE), "--port", str(self.port)]
        if self.bridge_mode:
            args.append("--bridge-mode")
        if self.hidden:
            args.append("--hidden")
        if self.plugin_path:
            args.extend(["--plugin", self.plugin_path])
        if self.sample_rate > 0:
            args.extend(["--sample-rate", str(self.sample_rate)])
        if self.buffer_size > 0:
            args.extend(["--buffer-size", str(self.buffer_size)])
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

    def _close_socket(self) -> None:
        sock = self._socket
        self._socket = None
        self._recv_buffer.clear()
        if sock is None:
            return
        try:
            sock.close()
        except Exception:
            pass

    def _ensure_socket(self) -> socket.socket:
        if self._socket is not None:
            return self._socket
        sock = socket.create_connection(("127.0.0.1", self.port), timeout=self._command_timeout)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sock.settimeout(self._command_timeout)
        self._socket = sock
        self._recv_buffer.clear()
        return sock

    def _read_response_line(self, sock: socket.socket) -> str:
        while True:
            newline_index = self._recv_buffer.find(b"\n")
            if newline_index >= 0:
                line = bytes(self._recv_buffer[:newline_index])
                del self._recv_buffer[:newline_index + 1]
                return line.decode("utf-8", errors="replace").strip()
            chunk = sock.recv(65536)
            if not chunk:
                raise ConnectionError("Native host closed the command socket")
            self._recv_buffer.extend(chunk)

    def _should_retry_after_transport_error(self) -> bool:
        last_command_at = float(self._last_command_at or 0.0)
        if last_command_at <= 0.0:
            return True
        return (time.monotonic() - last_command_at) >= 1.5

    def command(self, command: str, **payload: Any) -> dict[str, Any]:
        request = {"command": command, **payload}
        encoded = (json.dumps(request, separators=(",", ":")) + "\n").encode("utf-8")

        attempts = 2 if self._should_retry_after_transport_error() else 1
        last_transport_error: Exception | None = None
        for attempt in range(attempts):
            try:
                sock = self._ensure_socket()
                sock.sendall(encoded)
                raw = self._read_response_line(sock)
                if not raw:
                    raise RuntimeError("No response from native host")
                data = json.loads(raw)
                if not data.get("ok", False):
                    raise RuntimeError(data.get("message", "Native host command failed"))
                self._last_command_at = time.monotonic()
                return data
            except (ConnectionError, OSError, TimeoutError, json.JSONDecodeError) as exc:
                last_transport_error = exc
                self._close_socket()
                if attempt + 1 >= attempts:
                    break

        if last_transport_error is not None:
            raise RuntimeError(f"Native host transport failed: {last_transport_error}") from last_transport_error
        raise RuntimeError("Native host command failed")

    def stop(self, timeout: float = 3.0) -> None:
        try:
            self.command("quit")
        except Exception:  # noqa: BLE001
            pass
        finally:
            self._close_socket()

        if self.process is None:
            return

        try:
            self.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=timeout)
        finally:
            self.process = None


class NativeVstHostBridge:
    def __init__(
        self,
        plugin_path: str | None = None,
        *,
        port: int | None = None,
        open_editor: bool = False,
        bridge_mode: bool = True,
        hidden: bool = True,
        sample_rate: int | None = None,
        buffer_size: int | None = None,
    ) -> None:
        self.plugin_path = plugin_path
        self.port = port or _find_free_port()
        self.open_editor = open_editor
        self.bridge_mode = bool(bridge_mode)
        self.hidden = bool(hidden)
        self.sample_rate = int(sample_rate) if sample_rate else 0
        self.buffer_size = int(buffer_size) if buffer_size else 0
        self._backend: _InProcessHostBackend | _SubprocessHostBackend | None = None
        self.process: Any = None
        self.in_process = False

    def start(self, startup_timeout: float = 10.0) -> None:
        # Subprocess is the default backend.  The in-process backend (ctypes
        # DLL calls) can cause heap corruption with some VSTs, so it is
        # opt-in only.  Set AIMS_NATIVE_VST_IN_PROCESS=1 to enable it.
        force_in_process = str(os.getenv("AIMS_NATIVE_VST_IN_PROCESS", "")).strip().lower() in {"1", "true", "yes", "on"}
        use_in_process = force_in_process and _InProcessHostBackend.available()
        if use_in_process:
            backend: _InProcessHostBackend | _SubprocessHostBackend = _InProcessHostBackend(
                self.plugin_path,
                open_editor=self.open_editor,
                sample_rate=self.sample_rate,
                buffer_size=self.buffer_size,
            )
        else:
            backend = _SubprocessHostBackend(
                self.plugin_path,
                port=self.port,
                open_editor=self.open_editor,
                bridge_mode=self.bridge_mode,
                hidden=self.hidden,
                sample_rate=self.sample_rate,
                buffer_size=self.buffer_size,
            )
        backend.start(startup_timeout=startup_timeout)
        self._backend = backend
        self.process = getattr(backend, "process", None)
        self.in_process = isinstance(backend, _InProcessHostBackend)

    def wait_until_ready(self, timeout: float = 10.0) -> dict[str, Any]:
        if self._backend is None:
            raise RuntimeError("Native VST host is not started")
        return self._backend.wait_until_ready(timeout=timeout)

    def command(self, command: str, **payload: Any) -> dict[str, Any]:
        if self._backend is None:
            raise RuntimeError("Native VST host is not started")
        return self._backend.command(command, **payload)

    def queue_audio(self, audio_f32le: bytes, *, channels: int = 2, stream_id: str = "main") -> dict[str, Any]:
        if not audio_f32le:
            return self.command("status")
        return self.command(
            "queue_audio",
            audio_b64=base64.b64encode(audio_f32le).decode("ascii"),
            channels=max(1, int(channels)),
            stream_id=str(stream_id or "main"),
            format="f32le-interleaved",
        )

    def clear_audio_queue(self, *, reset_counters: bool = False, stream_id: str = "main") -> dict[str, Any]:
        return self.command("clear_audio_queue", reset_counters=bool(reset_counters), stream_id=str(stream_id or "main"))

    def start_audio_stream(self, *, stream_id: str = "main") -> dict[str, Any]:
        return self.command("start_audio_stream", stream_id=str(stream_id or "main"))

    def stop(self, timeout: float = 3.0) -> None:
        if self._backend is None:
            return
        self._backend.stop(timeout=timeout)
        self._backend = None
        self.process = None
        self.in_process = False


__all__ = ["NativeVstHostBridge", "HOST_DLL", "HOST_EXE"]

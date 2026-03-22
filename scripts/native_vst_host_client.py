import argparse
import json
import socket
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Send commands to the native AI Music Studio VST host.")
    parser.add_argument("--port", type=int, default=47653, help="localhost port exposed by the native host")
    parser.add_argument("--command", required=True,
                        choices=[
                            "ping",
                            "status",
                            "load",
                            "unload",
                            "open_editor",
                            "close_editor",
                            "note_on",
                            "note_off",
                            "all_notes_off",
                            "quit",
                        ])
    parser.add_argument("--path", help="plugin path for the load command")
    parser.add_argument("--note", type=int, help="MIDI note number")
    parser.add_argument("--velocity", type=float, help="MIDI velocity 0..1")
    parser.add_argument("--channel", type=int, help="MIDI channel 1..16")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    payload = {"command": args.command}
    if args.path:
        payload["path"] = args.path
    if args.note is not None:
        payload["note"] = args.note
    if args.velocity is not None:
        payload["velocity"] = args.velocity
    if args.channel is not None:
        payload["channel"] = args.channel

    request = (json.dumps(payload) + "\n").encode("utf-8")

    with socket.create_connection(("127.0.0.1", args.port), timeout=5.0) as sock:
        sock.sendall(request)
        sock.shutdown(socket.SHUT_WR)

        chunks: list[bytes] = []
        while True:
            chunk = sock.recv(4096)
            if not chunk:
                break
            chunks.append(chunk)

    raw = b"".join(chunks).decode("utf-8", errors="replace").strip()
    if not raw:
        print(json.dumps({"ok": False, "message": "No response from host"}))
        return 1

    try:
        response = json.loads(raw)
    except json.JSONDecodeError:
        print(raw)
        return 1

    print(json.dumps(response, indent=2, sort_keys=True))
    return 0 if response.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())

"""ハンドとの USB 通信を、つなぎっぱなしにしておく係。

  ハンドの基板は USB の口を開くたびにリセットされて原点合わせをする。物を持ったまま開き直すと、
  0 点がずれて全開で突き当たり「ガリガリ」鳴った（10/15）。そこで口はこの係が 1 回だけ開き、
  ほかのプログラムは 127.0.0.1:5005 に 1 行の命令を送って、ハンドの返事（OK／ERR の行まで）を受け取る。
使い方:  .venv_ur\\Scripts\\python.exe real_ur\\hand_server.py        （裏で動かしておく）
"""
import socketserver
import threading
import time

import serial

from hand import find_port

HOST, PORT = "127.0.0.1", 5005
lock = threading.Lock()
ser = None


def ask(line, wait=1.5):
    with lock:
        ser.reset_input_buffer()
        ser.write((line.strip() + "\n").encode())
        out, end = [], time.time() + wait
        while time.time() < end:
            ln = ser.readline().decode("utf-8", "replace").strip()
            if ln:
                out.append(ln)
                if ln.startswith(("OK", "ERR")):
                    break
        return out


class H(socketserver.StreamRequestHandler):
    def handle(self):
        for raw in self.rfile:
            line = raw.decode("utf-8", "replace").strip()
            if not line:
                continue
            wait = 12.0 if line == "home" else 1.5
            for ln in ask(line, wait) or ["ERR no_reply"]:
                self.wfile.write((ln + "\n").encode())
            self.wfile.write(b".\n")                          # 返事の終わりの印


def main():
    global ser
    port = find_port()
    ser = serial.Serial(port, 115200, timeout=0.1)
    end = time.time() + 12
    while time.time() < end:                                   # リセット後の自動原点合わせを待つ
        ln = ser.readline().decode("utf-8", "replace").strip()
        if ln:
            print("ハンド:", ln, flush=True)
            if ln.startswith("OK home"):
                break
    print(f"係を開始: {port} を {HOST}:{PORT} で受け付け", flush=True)
    socketserver.ThreadingTCPServer.allow_reuse_address = True
    with socketserver.ThreadingTCPServer((HOST, PORT), H) as srv:
        srv.serve_forever()


if __name__ == "__main__":
    main()

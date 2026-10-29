"""電動ハンド（TAIYO ESG1-SS-4225 を自作基板 Arduino Nano ＋ TMC2209 で駆動）を USB で動かす。

  通信 115200bps・改行区切り。応答は「OK …」「ERR …」、通知は「EVT …」。
  座標＝指の開き量（エンコーダのカウント）。0＝全閉、64カウント＝1mm、全開の手前＝1590（約 24.8mm）。
  USB の口を開くと Nano がリセットされ、自動で原点合わせ（全閉 → 開く → 全閉、3〜4 秒）をする。

使い方:
  .venv_ur\\Scripts\\python.exe real_ur\\hand.py ports                 USB の口を一覧
  .venv_ur\\Scripts\\python.exe real_ur\\hand.py status                状態を読む（原点合わせは起きる）
  .venv_ur\\Scripts\\python.exe real_ur\\hand.py open | close
  .venv_ur\\Scripts\\python.exe real_ur\\hand.py move 10.0             開き量を mm で指定
  .venv_ur\\Scripts\\python.exe real_ur\\hand.py grip 50               力 50% で閉じてつかむ
"""
import sys
import time

import serial
import serial.tools.list_ports

COUNTS_PER_MM = 64.0
OPEN_MAX = 1590
CH340 = (0x1A86, 0x7523)


def find_port():
    ports = list(serial.tools.list_ports.comports())
    for p in ports:
        if (p.vid, p.pid) == CH340 or "CH340" in (p.description or ""):
            return p.device
    usb = [p.device for p in ports if p.vid]                 # Bluetooth の仮の口（vid なし）は除く
    return usb[0] if len(usb) == 1 else None


class Hand:
    def __init__(self, port=None, log=print):
        self.port = port or find_port()
        if not self.port:
            raise SystemExit("ハンドの USB が見つからない（ケーブルを確かめるか、--port で指定）")
        self.log = log
        self.s = serial.Serial(self.port, 115200, timeout=0.1)
        self.wait_ready()

    def lines(self, seconds):
        end = time.time() + seconds
        while time.time() < end:
            ln = self.s.readline().decode("utf-8", "replace").strip()
            if ln:
                yield ln

    def wait_ready(self, seconds=12.0):
        """口を開いた直後のリセットと自動原点合わせが終わるまで待つ（status に OK が返るまで）。"""
        time.sleep(2.0)                                       # ブートローダが終わるまで
        end = time.time() + seconds
        while time.time() < end:
            r = self.cmd("status", wait=1.0, quiet=True)
            if r and r.startswith("OK") and "homed=1" in r:
                self.log("準備できた: " + r)
                return r
            time.sleep(0.5)
        raise SystemExit("ハンドから応答がない（24V の電源か、USB の口を確かめる）")

    def cmd(self, text, wait=1.0, quiet=False):
        self.s.reset_input_buffer()
        self.s.write((text + "\n").encode())
        for ln in self.lines(wait):
            if not quiet:
                self.log("  " + ln)
            if ln.startswith(("OK", "ERR")):
                return ln
        return None

    def pos_mm(self):
        r = self.cmd("p", quiet=True) or ""
        for tok in r.split():
            if tok.startswith("pos="):
                return int(tok[4:]) / COUNTS_PER_MM
        return None

    def check(self, why):
        """動かす前に、今の位置と状態を確かめる（決まり：ハンドを動かす時は、初めに位置を確認する）。
        原点合わせ済みでない・位置が 0〜全開の外・異常の印がある時は、動かさずに止める。"""
        for _ in range(3):                                     # PC が重い時、前の命令の返事が遅れて届くことがある（10/15）。状態の返事が来るまで聞き直す
            st = self.cmd("status", quiet=True) or ""
            if st.startswith("OK status"):
                break
            self.log(f"  [確認] 状態の返事でない（{st[:40]!r}）ので聞き直す")
        kv = dict(t.split("=", 1) for t in st.split() if "=" in t)
        pos = int(kv.get("pos", "-9999"))
        mm = pos / COUNTS_PER_MM
        self.log(f"  [確認] {why}の前: 開き量 {mm:.1f}mm、原点 {'済' if kv.get('homed') == '1' else 'まだ'}、"
                 f"ドライバ {kv.get('driver', '?')}、過熱 {kv.get('ot', '?')}/{kv.get('otpw', '?')}、ショート {kv.get('s2g', '?')}")
        bad = []
        if kv.get("homed") != "1":
            bad.append("原点合わせがまだ")
        if not -32 <= pos <= OPEN_MAX + 32:                   # 0.5mm の余裕
            bad.append(f"位置が範囲の外（{pos} カウント）")
        if kv.get("driver") != "ok" or kv.get("ot") not in ("0", None) or kv.get("s2g") not in ("0", None):
            bad.append("ドライバの異常の印")
        if bad:
            raise SystemExit("ハンドを動かさずに止めます: " + "、".join(bad))
        return mm

    def move_mm(self, mm, wait=3.0):
        self.check(f"{mm:.1f}mm へ動かす")
        c = int(max(0, min(OPEN_MAX, round(mm * COUNTS_PER_MM))))
        self.cmd(f"move {c}")
        return self.settle(wait)

    def open(self, wait=3.0):
        self.check("開く")
        self.cmd("open")
        return self.settle(wait)

    def grip(self, force=50, wait=3.0):
        """力 force% で閉じる。物に当たればその位置で止まって押さえ続ける。戻り値：止まった開き量（mm）。"""
        self.check(f"力 {int(force)}% で閉じる")
        self.cmd(f"force {int(force)}")
        self.cmd("close")
        return self.settle(wait)

    def grip_start(self, force=50):
        """力 force% で閉じ始めるだけ（止まるのを待たない）。腕を止めずに追いながら閉じる時に使う（10/22）。"""
        self.check(f"力 {int(force)}% で閉じ始める")
        self.cmd(f"force {int(force)}", quiet=True)
        self.cmd("close", quiet=True)

    def settle(self, wait):
        """指が止まるまで待って、開き量（mm）を返す。"""
        last, still, end = None, 0, time.time() + wait
        while time.time() < end:
            p = self.pos_mm()
            if p is not None and last is not None and abs(p - last) < 0.05:
                still += 1
                if still >= 3:
                    return p
            else:
                still = 0
            last = p
            time.sleep(0.1)
        return last

    def close_port(self):
        self.s.close()


class HandClient(Hand):
    """hand_server.py（USB をつなぎっぱなしにする係）を通してハンドを動かす。口を開け直さないのでリセットが起きない。"""

    def __init__(self, log=print, host="127.0.0.1", port=5005):
        import socket
        self.log = log
        self.sock = socket.create_connection((host, port), timeout=15)
        self.f = self.sock.makefile("rw", encoding="utf-8", newline="\n")
        r = self.cmd("status", quiet=True)
        if not (r and "homed=1" in r):
            raise SystemExit("ハンドの係から正しい返事がない: " + str(r))

    def cmd(self, text, wait=None, quiet=False):
        self.f.write(text + "\n")
        self.f.flush()
        last = None
        for ln in self.f:
            ln = ln.strip()
            if ln == ".":
                break
            if not quiet:
                self.log("  " + ln)
            if ln.startswith(("OK", "ERR")):
                last = ln
        return last

    def close_port(self):
        self.sock.close()


def main():
    a = sys.argv[1:] or ["ports"]
    if a[0] == "ports":
        for p in serial.tools.list_ports.comports():
            print(p.device, p.description, hex(p.vid or 0), hex(p.pid or 0))
        print("ハンドらしい口:", find_port())
        return
    h = Hand()
    try:
        if a[0] == "status":
            h.cmd("status")
            h.cmd("params")
        elif a[0] == "open":
            print("開き量:", h.open(), "mm")
        elif a[0] == "close":
            print("開き量:", h.grip(int(a[1]) if len(a) > 1 else 50), "mm")
        elif a[0] == "grip":
            print("開き量:", h.grip(int(a[1])), "mm")
        elif a[0] == "move":
            print("開き量:", h.move_mm(float(a[1])), "mm")
        else:
            print(h.cmd(" ".join(a)))
    finally:
        h.close_port()


if __name__ == "__main__":
    main()

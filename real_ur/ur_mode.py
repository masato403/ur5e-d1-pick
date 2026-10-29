"""実機モードかシミュレーションモードかを読む（読むだけ）。

ポート 30001 の状態データの「Robot Mode Data」に、実機がつながっているか・実機が有効か
（＝ペンダントのシミュレーションがオフか）が入っている。
使い方:  .venv_ur\\Scripts\\python.exe real_ur\\ur_mode.py --ip 192.0.2.10
"""
import argparse
import socket
import struct


def read_robot_mode(ip):
    with socket.create_connection((ip, 30001), timeout=3) as s:
        for _ in range(20):
            head = b""
            while len(head) < 5:
                head += s.recv(5 - len(head))
            size, ptype = struct.unpack(">iB", head)
            body = b""
            while len(body) < size - 5:
                body += s.recv(size - 5 - len(body))
            if ptype != 16:                       # 16 = ロボットの状態のまとまり
                continue
            i = 0
            while i < len(body):
                sub_size, sub_type = struct.unpack(">iB", body[i:i + 5])
                if sub_type == 0:                 # 0 = Robot Mode Data
                    d = body[i + 5:i + sub_size]
                    names = ("実機がつながっている", "実機が有効（シミュレーションがオフ）", "腕の電源が入っている",
                             "非常停止中", "保護停止中", "プログラム実行中", "プログラム一時停止中")
                    return dict(zip(names, (bool(b) for b in d[8:15])))
                i += sub_size
    return None


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", required=True)
    for k, v in (read_robot_mode(ap.parse_args().ip) or {"読めなかった": True}).items():
        print(f"  {k}: {'はい' if v else 'いいえ'}")

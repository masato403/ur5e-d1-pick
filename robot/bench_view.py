r"""ページに出すカメラの映像（test_track_real.py の frame の中、10 コマごと）を描く時間を測り、送る間を縮めた時にシミュレーターがどれだけ遅れるかを出す（10/23）。
使い方:  .venv\Scripts\python.exe robot\bench_view.py   （試験を流していない時に）"""
import os
import sys
import time

os.environ.setdefault("SIM_TRIPOD", "1")
os.environ.setdefault("SIM_MAT", "real")
import cv2  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sim_real as S  # noqa: E402
import sim_rtde  # noqa: E402

TW, TH = 960, 540
N = 60


def avg(fn):
    fn()                                                         # 1 回目は準備の時間が入るので数えない
    t0 = time.perf_counter()
    for _ in range(N):
        fn()
    return (time.perf_counter() - t0) / N * 1000


def main():
    env = S.RealLikeEnv()
    q = [cv2.IMWRITE_JPEG_QUALITY, 85]
    t_frame = avg(lambda: sim_rtde.frame(env))                   # 物理の 1 コマ（20ms 分）。VIEWER が無いので待たない
    t_tri = avg(lambda: cv2.resize(env.render_tripod()[0], (TW, TH), interpolation=cv2.INTER_AREA))
    t_wr = avg(lambda: cv2.resize(env.render_real(), (TW, TH), interpolation=cv2.INTER_AREA))
    img = cv2.resize(env.render_tripod()[0], (TW, TH))
    t_enc = avg(lambda: cv2.imencode(".jpg", img, q))
    small = cv2.resize(img, (TW // 2, TH // 2))
    t_enc_s = avg(lambda: cv2.imencode(".jpg", small, q))
    import mujoco
    rv = mujoco.Renderer(env.m, 270, 480)

    def small_shot():
        for cam in ("tripod", "wrist_real"):
            rv.update_scene(env.d, camera=cam)
            cv2.imencode(".jpg", cv2.cvtColor(rv.render(), cv2.COLOR_RGB2BGR), q)
    t_small = avg(small_shot)
    print(f"見せる映像だけ 480×270 で描いた時：1 回送る（2 枚）のに {t_small:.1f}ms")
    shot = t_tri + t_wr + 2 * t_enc                              # 1 回送る（三脚と手首の 2 枚）のにかかる時間
    print(f"物理の 1 コマ {t_frame:.1f}ms（本当の時間では 20ms）、三脚を描く {t_tri:.1f}ms、手首を描く {t_wr:.1f}ms、"
          f"jpg にする {t_enc:.1f}ms（半分の大きさなら {t_enc_s:.1f}ms）→ 1 回送るのに {shot:.1f}ms")
    print("送る間（コマ） | 枚/秒（本当の時間） | シミュレーターの速さ（本当の時間に対して）")
    for every in (10, 5, 3, 2, 1):
        sim_s = every * 0.02
        wall = max(sim_s, every * t_frame / 1000 + shot / 1000)   # 本当の時間より速い分は frame が待って合わせる
        print(f"{every:>3} | {1 / wall:5.1f} | {sim_s / wall * 100:5.0f}%")


if __name__ == "__main__":
    main()

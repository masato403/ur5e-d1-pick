r"""シミュレーション（robot/track_sim.py）と同じ流れの実機版（10/20 夕方に作った。実機ではまだ動かしていない）。
手首では 3 枚撮らない。三脚のカメラで物を見つけ、d1-3B が種類（tripod_view.kinds）と運ぶ物（plan_ai.instruction）・つかむ高さ（plan_ai.grip）を決める。
腕を物の上へ動かし、下りながら段ごとに三脚で物の位置を測り直して寄せ、手首も物の向きに回す（物が動いても追う）、pick_by_camera.grasp でつかんで箱へ落とす。
  ・下りる時は servoL で止まらずに下りながら、三脚の最新の写真で物の位置と向きを測り直して寄せる（TR_SERVO=1、標準。10/22）。
    TR_SERVO=0 で前のやり方（100/60/35mm で moveL で止まって測る）。
  ・下りた高さからそのまま下ろしてつかむ（PB.grasp の from_here）。2 個目からは箱に落とした所で三脚を見て、見つからない時だけ LOOK へ戻る（10/22）。
  ・三脚の写真で腕の先の所は除く（tripod_view.robot_mask）。指が物に近づいて三脚から見えなくなったら、最後に測った位置を使う。
  ・記録：logs_track_real/時刻/ に三脚の写真と、段ごとの位置（steps.json）を残す。
使い方:  .venv_ur\Scripts\python.exe real_ur\track_real.py --ip 192.0.2.10 [--say 全部の物を箱に入れて] [--go]
  --go が無ければ、三脚で見つけて種類と運ぶ物を決めるだけ（腕もハンドも動かさない）。
  先に d1_server.py と、CAM_TRI_EVERY=0.1 で cam_view.py --cam 2 --cam2 1 を動かしておく。
"""
import argparse
import json
import math
import os
import time

import cv2
import numpy as np

import kind_ai as KA
import locate as L
import measure as M
import pick_any as PA
import pick_by_camera as PB
import plan_ai as PL
import teach_box as TB
import tripod_view as TV
from plan_hover import TABLE_Z

HERE = os.path.dirname(os.path.abspath(__file__))
LOOK = (20.0, -350.0, 150.0)                  # 三脚で見る時の指先の位置（机から mm）。腕がマットの左上の外に出る（10/20 の記録 arm_ の 1 枚目）
STEPS = (100.0, 60.0, 35.0)                   # 物の上で下りながら三脚で測り直す高さ（机から mm）
FOLLOW_MM = 40.0                              # 前の位置からこれより遠い物は、同じ物とみなさない
H_NOM = {"消しゴム": 12.0, "ペン": 10.0, "銀の棒": 8.0}   # 高さ（三脚では測れないので、実機で測った値）
SPEED, ACCEL = 0.10, 0.4
SERVO = os.environ.get("TR_SERVO", "1") == "1"   # 1：servoL で止まらずに下りながら追う（10/22）。0：STEPS の高さで止まって測る
SERVO_DT = 0.02                               # servoL を送る間（秒）
MEAS_EVERY = 0.1                              # 三脚で測り直す間（秒）。cam_view の CAM_TRI_EVERY=0.1 に合わせる
DESC_V = float(os.environ.get("TR_DESC_V", 0.05))   # 下りる速さ（m/秒）
XY_V = 0.20                                   # 物を追って横へ動く速さの上限（m/秒）
APPROACH_MM = 15.0                            # 物の真上からこれ以内に来たら下り始める
K_XY = 4.0                                    # 行き先との差 → 速さ（1/秒）
A_MAX = 0.8                                   # 道具の点の加速度の上限（m/秒²）
GATE_GROW = 60.0                              # 見えない間、同じ物とみなす範囲を 1 秒にこれ（mm）ずつ広げる（150 では腕の下に隠れた間に 130mm 先の別の物と取り違えた）
GATE_MAX = 100.0                              # 同じ物とみなす範囲の上限（mm）
PRED_LOW_S = float(os.environ.get("TR_PRED_LOW_S", "0"))   # >0：指先が物の上 15mm より下の時は、隠れてもこの秒まで最後の速さで先を読み、腕も動かし続ける
                                               # （0 は PRED_S と同じ。10/22、横に動く物で、0.3 秒で止まるとみなして腕が止まり、物だけ動いて爪に当たった）
PRED_S = 0.3                                  # 見えない間、最後の速さで先を読むのはこの秒まで（その先は止まっているとみなす）
VEL_FIT = float(os.environ.get("TR_VEL_FIT", 0))   # >0：物の速さを、同じカメラの直近この秒数の測り値に直線を当てはめて出す（0 は前の値と半々。三脚の 3〜6mm の揺れで 30〜65mm/秒に跳ね、行き過ぎて時間切れになった。10/23 X10_m）
OBJ_V_MAX = 150.0                             # 物の速さの見当の上限（mm/秒）
LEAD = 0.1                                    # 物の動きを、この秒だけ先回りする（写真の遅れの分）
WRIST_EVERY = float(os.environ.get("TR_WRIST_EVERY", 0.2))   # 追う間、手首のカメラでも測る間（秒）。0 で使わない（10/22、三脚と両方を常に使う。止まらずに追えるので標準に）
WRIST_LIVE_S = 0.5                            # 追う間の手首の写真で、この秒の間に物を測れていれば、三脚から隠れても止まって撮らない（測れていなければ止まって 1 枚。
                                               # 写真が届くだけで止まらないことにしたら、写真の端で切れた物で上がり直しを時間切れまで繰り返した。10/22）
WRIST_BORDER = os.environ.get("TR_WRIST_BORDER", "1") == "1"   # 1：手首の写真の端に触れている物は使わない（切れた物は真ん中がずれる。10/22、ペンで 10mm）
VEL_SAME = os.environ.get("TR_VEL_SAME", "1") == "1"           # 1：物の速さは同じカメラ同士の値の差だけから出す（手首と三脚の食い違いを動きと読み違えた）
WRIST_LAG = 0.05                              # 手首の写真の時刻から、実際に撮った時までの遅れの見当（秒）
WRIST_STILL = 2.0                             # 手首のカメラで 2 枚続けてこれ（mm）以内なら、物は止まったとみなす
WRIST_MAX = int(os.environ.get("TR_WRIST_MAX", 0))                                # 手首のカメラで撮り直す回数の上限
WRIST_GATE = 60.0                             # 手首のカメラで、見当からこれ（mm）以内の物を同じ物とみなす
WRIST = os.environ.get("TR_WRIST", "1") == "1"   # 1：真上で三脚から見えない時は、手首のカメラで 1 枚撮って直してから下りる（10/22）
FOLLOW_MAX_S = 10.0                           # 追いかけの時間の上限（秒）
ROT_V = 1.5                                   # 手首を回す速さの上限（rad/秒）
END_ON = os.environ.get("TR_END", "0") == "1"   # 1：隠れた物を見えている端から測る。標準は 0（10/22、隠れて短くなる銀の棒で真ん中が一方へずれ続け、見当がマットの外へ逃げた。
                                               # 止まった物 10 場面でも、銀の棒が長い向きに 50mm ずれてつかみ損ねた 1 回が 0 にした後は無くなった）
END_RATIO = 2.5                               # 長さが幅のこの倍以上の物だけ、隠れた時に見えている端から真ん中を出す
RETRY = int(os.environ.get("TR_RETRY", 1))     # つかみ損ねた時に、その場で三脚を見直してやり直す回数（10/22）
NEAR_LOOK = os.environ.get("TR_NEAR_LOOK", "0") == "1"   # 1：運ぶ種類が三脚で見つからない時、「それ以外」の物の上へ行き手首のカメラで d1-3B に聞き直す（10/22）
MIN_R = float(os.environ.get("TR_MIN_R", 0))  # >0：腕の根元（真上から見た原点）からこれ（mm）より近い物は取りに行かない。
                                               # 大きいマットの奥（根元から 196mm）の消しゴムで、ハンドが上腕に当たって保護停止した（10/22、シミュレーター）
MAX_R = float(os.environ.get("TR_MAX_R", 750))  # 腕の根元からこれ（mm）より遠い物は、届かないので取りに行かない（シミュレーターで 777mm は届かず、743mm は届いた。10/22）
SAME_COLOR = float(os.environ.get("TR_SAME_COLOR", 0))   # >0：測り直しで、最初に見た時と色合い（Lab の a・b の平均）がこれより違う塊は別の物とみなす。
                                               # 止まった銀の棒を追う間に、腕に半分隠れて長さ 89mm に見えた青いペンへ乗り移り、50mm ずれてつかみ損ねた（10/22）
COMMIT_MM = float(os.environ.get("TR_COMMIT_MM", 12))   # 下りる先（STEPS[-1]）からこれ（mm）以内まで来たら、見えなくなっても上がり直さずに下り切る。
                                               # 手首の写真に物が収まらなくなる 36mm あたりで見失い、36〜46mm を時間切れまで上下した（10/22）
D1_STEER = int(os.environ.get("TR_D1_STEER", "0"))   # 2：動く向きも d1-3B が選ぶ（8 方向・待つ・下りる）   # 1：追いかけの方針（先回り・真上へ・下りる）を d1-3B が 0.2 秒ごとに選ぶ（10/22、d1-3B が主に決めるように）。
                                               # 数字（距離・速さ・先回りの位置）はプログラムが計算して説明に書き、行き先へ滑らかに動かすのもプログラム
TO_GRIP = os.environ.get("TR_TO_GRIP", "0") == "1"   # 1：35mm で止めずに、つかむ高さまで追いながら下り、その場で閉じる（10/22。35mm からつかむまでの約 1.4 秒は
                                               # 追っておらず、10mm/秒で動き続ける物が 13mm ずれて外れた）
DIR_TEXT = int(os.environ.get("TR_DIR_TEXT", "2"))     # 向き選び（TR_D1_STEER=2）の書き方。0：前の文。1：選んだ結果を書く＋中立の問い
PARTIAL = os.environ.get("TR_PARTIAL", "0") == "1"   # 1：手首の写真で一部だけ見える物（端で切れた・近くて全体が入らない）も、幅と向きが合えば横の位置だけ使う
                                               # （10/22、近づくと物の全体が見えず、違う物と判断して上がっていた）
WRIST_GRASP = os.environ.get("TR_WRIST_GRASP", "0") == "1"   # 1：手首は「つかめるか」だけを見る。何の物か・長さは三脚に任せ、幅と向きが合えば
                                                           # 指を閉じる向き（横）の位置だけを使う（10/22、手首のカメラはつかめるかどうかの確認だけに使う）
KEEP_AXIS = os.environ.get("TR_KEEP_AXIS", "0") == "1"   # 1：追う間、長い向きが AXIS_JUMP° より大きく変わった測り値は、向きを使わない
AXIS_JUMP = 25.0
FAST_LAST = float(os.environ.get("TR_FAST_LAST", 0))   # >0：物の上の面＋CLEAR_MM より下は、この速さ（m/秒）で一気に下りる（10/23 消しゴムの横移動）
WRIST_LOW_EVERY = float(os.environ.get("TR_WRIST_LOW_EVERY", 0))   # >0：指先が物の上 15mm より下の時、手首で測る間（秒）
FINE_REAL = os.environ.get("TR_FINE_REAL", "0") == "1"   # 1：指と物のすき間を本当の値（片側、余裕を引かない）にする。前は最低 2mm（消しゴムは本当は 0.4mm）
FINE_MIN = 0.3
LEAD_CLOSE = float(os.environ.get("TR_LEAD_CLOSE", 0))   # >0：指を閉じる向きの物の動きだけ、この秒数さらに先回りする
HIT_UP = os.environ.get("TR_HIT_UP", "0") == "1"   # 1：下りる途中で指が物に当たったら（力・下りられない）、閉じずに物の上へ上がって合わせ直す
                                               # （10/22、当たっているのに閉じていたのを直す）
HIT_N = 8.0                                    # 当たったとみなす力の増え（N）。シミュレーターの保護停止は 60N
SAFE_N = float(os.environ.get("TR_SAFE_N", 20.0))   # >0：安全の決まり。下りている間に力がこれだけ増えたら、決めごとに関係なく上がる（10/23 実機：時間切れの後にペンを押し付けてひびを入れた。HIT_UP を切ると力を全く見ていなかった）
HIT_LAG_MM = 3.0                               # 送った高さより本当の高さがこれ以上高ければ、物に乗って下りられない
HIT_LEAD_S = 0.5                               # 当たった後、物の動く向きにこの秒数ぶん先へ寄せながら上がる
HIT_LOOK = float(os.environ.get("TR_HIT_LOOK", 0))   # >0（TR_HIT_UP=1 と使う）：当たったら机から この mm まで上がり、止まって手首で測り直す（10/23）
HIT_IN_MM = float(os.environ.get("TR_HIT_IN_MM", 0))   # >0：当たった時に指先が物の上の面よりこの mm 以上下なら、物は指の間（横に触れただけ）→ 上がらずに閉じる（10/23 C7：-5.2mm で上がって外した）
HIT_HOLD_S = 0.6                               # 当たった後、上がって合わせ直す時間（秒）


def force_now(r):
    """手先にかかる力の大きさ（N）。実機は RTDE の getActualTCPForce、シミュレーターはハンドに当たる力。"""
    try:
        return float(np.linalg.norm(r.getActualTCPForce()[:3]))
    except Exception:
        return 0.0


CLOSE_MOVING = os.environ.get("TR_CLOSE_MOVING", "0") == "1"   # 1：つかむ高さでそろったら、腕を止めずに物の速さで追いながら閉じる（TO_GRIP の時）
CLOSE_MOVE_S = 1.5                                         # 追いながら閉じる時の最長（秒）
CLOSE_WAIT_S = float(os.environ.get("TR_CLOSE_WAIT_S", 4.0))   # 追いながら閉じた後、指が閉じ切るまで待つ最長（秒）。10/23 実機
TOUCH_W0 = os.environ.get("TR_TOUCH_W0", "0") == "1"   # 1：「触れた」は開き量が閉じ始めより 2mm 以上減ってから（10/26 実機：三脚の幅 23mm＋2 が開いた時の 24.8mm より大きく、閉じ始めてすぐ触れたと出て、なでて終わった）
HOLD_LOW = os.environ.get("TR_HOLD_LOW", "0") == "1"   # 1：物の近くで下りられない時は、STEPS[0] の高さまで上がらず、その高さで待つ（見えないだけで上がらない）
UNC_K = float(os.environ.get("TR_UNC_K", "0.3"))   # 見えない間のずれの見積もり：速さの測り違いの割合（先を読んでいるので、速さ全部ではない）
UNC_BASE = float(os.environ.get("TR_UNC_BASE", "2.0"))   # 見えない間のずれの見積もり：測る誤差（mm）
CLOSE_TEXT = int(os.environ.get("TR_CLOSE_TEXT", "5"))   # 閉じるかの問いの書き方。0：前の文。1：選んだ結果を書く＋中立の問い。2：1 で問いは前のまま。
                                                         # 3：失敗をはっきり＋追う結果も書く＋中立の問い。4：3 で問いは「今閉じる？」
LOOK_TEXT = int(os.environ.get("TR_LOOK_TEXT", "2"))   # 1：「止まって撮る」がある場面（見えていない・物の近く）だけ、シューティング式の文で聞く
                                                         # （10/22 文章だけの比べ：撮るべき場面で 文章 5〜7 は 0〜3/96、シューティング式 85/96・全体 97/108）
STEER_TEXT = int(os.environ.get("TR_STEER_TEXT", "5"))   # 追い方の問いの書き方。0：前の文。2：選択肢ごとに選んだ結果を数字で書く。3：2 ＋ 問いを中立に。4：2 で遠い時の「下りる」の結果を強く（失敗する）。5：4 ＋ 中立の問い
TIMEOUT_UP = os.environ.get("TR_TIMEOUT_UP", "1") == "1"   # 1：TO_GRIP で追いかけが時間切れ（そろわない）なら、下ろさずに上がって見直す（やり直し 1 回に数える）
                                               # （0 は前と同じく、その場から下ろす。10/22 G_d1kcn で、下ろした指が物に当たって失敗した）
STEER_FACTS = os.environ.get("TR_STEER_FACTS", "0") == "1"   # 1：追い方の問いに、指と物のすき間と、今下りれば指の間に入るかを書く
                                               # （10/22：d1-3B が毎回「先回り」を選び、例外を外すと一度も下りなかった）
D1_LOOK = os.environ.get("TR_D1_LOOK", "0") == "1"   # 1：見えない時に止まって手首で撮るかも d1-3B が選ぶ（TR_D1_STEER=1 の選択肢「look」。0 はプログラムが決める）
D1_CLOSE = os.environ.get("TR_D1_CLOSE", "0") == "1"   # 1：下り切ってそろった後、閉じるかどうかを d1-3B が決める（0 は物の速さが 5mm/秒 より遅くなるまで待つ。
                                               # ずっと動く物で 7〜10 秒閉じなかった、10/22）
CLOSE_T = float(os.environ.get("TR_CLOSE_T", "0.4"))   # 指が閉じるのにかかる時間の目安（秒、説明に書くだけ）
CLEAR_MM = float(os.environ.get("TR_CLEAR_MM", "0"))   # >0：TO_GRIP の時、ずれが指のすき間より大きい間は物の上の面からこの mm 上で追う（0 はしない。
                                               # 10/22 G_std で、つかむ高さで追っていた指が銀の棒の横に当たって保護停止した）
OPEN_MM = 24.8                                 # 指を開いた時の内側の幅（mm）
D1_NOEXC = os.environ.get("TR_D1_NOEXC", "0") == "1"   # 1：迷い 3 秒・下り切る 12mm の決まりも外し、迷った時間・上がり直しの回数を状態に書いて d1-3B に任せる（10/22）
HES_TEXT = int(os.environ.get("TR_HES_TEXT", 1))   # D1_HES の文章。1：状態に迷った時間と待つと何が起きるかを書く。2：＋迷いが 3 秒以上なら、待つ選択肢に「迷って見える（bad）」、外れそうな下りるに「外れたらやり直す」と書く
D1_CLOSE_STILL = os.environ.get("TR_D1_CLOSE_STILL", "0") == "1"   # 1：止まっている物でも、閉じるか（閉じる／追ってそろえ直す）を d1-3B に聞く（0 は「閉じる」しか残らないので聞かない。10/23 全部 d1-3B の確かめ）
D1_HES = os.environ.get("TR_D1_HES", "0") == "1"   # 1：迷い 3 秒の決まりだけ外し、迷った時間と「待つと何が起きるか」を数字で書いて d1-3B に任せる（10/23。下り切る 12mm の決まりは残す）
STEER_EVERY = 0.2                              # d1-3B に方針を聞く間（秒）
STEER_LEAD = 0.6                               # 「先回り」の時、物の何秒先の位置で待つか
KIND_EN = {"消しゴム": "an eraser", "ペン": "a pen", "銀の棒": "a silver rod"}
_STEER_POOL = None


def hes_fact(hes, t, v_mm):
    """D1_HES：迷った時間と、待つと何が起きるか（d1-3B に渡す事実）。"""
    return (f" The hand has waited near the object for {hes:.1f} s without going down. Visitors are watching: waiting more than {HESITATE_S:.0f} s "
            f"looks hesitant (bad). The attempt is given up at {FOLLOW_MAX_S:.0f} s ({max(0.0, FOLLOW_MAX_S - t):.1f} s left), "
            f"and each more second of waiting lets the object move {v_mm:.0f} mm further.")


def steer_ask(kind, h_mm, dist_mm, v_mm, age, near, desc_mm, extra="", can_look=False, gap_mm=None, hes=0.0):
    """追いかけの方針を d1-3B に聞く（裏の糸で。腕の servoL を止めないため）。選択肢は、できない物を聞く前に外す。"""
    global _STEER_POOL
    import concurrent.futures as cf
    if _STEER_POOL is None:
        _STEER_POOL = cf.ThreadPoolExecutor(1)
    seen = "It is visible in the cameras now." if age < 0.15 else f"It has not been seen for {age:.1f} s."
    state = (f"A robot hand is following {KIND_EN.get(kind, 'an object')} lying on a cutting mat to pick it up. "
             f"The hand is {h_mm:.0f} mm above the table and {dist_mm:.0f} mm away from the spot right above the object. "
             f"The object is moving at {v_mm:.0f} mm/s. " + seen + extra)
    ch = {"over": f"Move to right above where the object is now, staying {STEPS[0]:.0f} mm above the table."}
    if v_mm >= 3:
        ch["lead"] = (f"Move ahead to where the object will be in {STEER_LEAD:.1f} s ({v_mm * STEER_LEAD:.0f} mm ahead in its moving direction), "
                      f"staying {STEPS[0]:.0f} mm above the table, and wait for it there.")
    if STEER_FACTS and gap_mm is not None:                     # 下りれば指の間に入るか（プログラムが計算した事実）を説明に書く
        state += (f" The open fingers leave about {gap_mm:.0f} mm on each side of the object, so the hand is "
                  + ("already close enough to grasp it." if dist_mm <= gap_mm else f"{dist_mm - gap_mm:.0f} mm too far to grasp it now."))
        if "lead" in ch:
            ch["lead"] += f" This moves the hand about {v_mm * STEER_LEAD:.0f} mm away from right above the object now."
    if near:
        ch["descend"] = (f"Go down now and grasp it ({desc_mm:.0f} mm down, slowly)."
                         + (" Even if it is not visible now, go down to the current estimate; if the grasp misses, it is retried once." if D1_NOEXC else ""))
    if can_look:                                                 # D1_LOOK：止まって手首で撮るかも d1-3B が選ぶ（見えていない時だけ残す）
        ch["look"] = "Stop here for a moment and take one photo with the wrist camera to find the object again (the hand stops for about 0.5 s)."
    q = "What should the hand do next to grasp the moving object quickly and surely?"
    if STEER_TEXT >= 2 and gap_mm is not None:                 # 選択肢ごとに「選ぶと何が起きるか」を数字で書く（10/22。前の文は d1-3B が毎回「先回り」を選んだ）
        off_mm = dist_mm + (v_mm * age if STEER_TEXT >= 6 and age >= 0.15 else 0.0)   # 6：見えていない間のずれも足して、指の間に入るかを決める
        inside = off_mm <= gap_mm
        state = (f"A robot hand is chasing {KIND_EN.get(kind, 'an object')} on a cutting mat to grasp it. "
                 f"The fingertips are {h_mm:.0f} mm above the table. The hand is {dist_mm:.0f} mm off from right above the object, "
                 f"and the open fingers leave {gap_mm:.0f} mm on each side of it. The object moves at {v_mm:.0f} mm/s. " + seen
                 + (f" So its position may be off by about {v_mm * age:.0f} mm." if STEER_TEXT >= 6 and age >= 0.15 and v_mm * age >= 1 else "") + extra)
        ch = {"over": f"Stay {STEPS[0]:.0f} mm above the table and move to right above the object. Nothing is grasped yet."}
        if v_mm >= 3:
            ch["lead"] = (f"Stay {STEPS[0]:.0f} mm above the table and move {v_mm * STEER_LEAD:.0f} mm ahead of the object to wait for it. "
                          "Nothing is grasped yet.")
        if near:
            ch["descend"] = (f"Go down {desc_mm:.0f} mm while following the object, then close the fingers. "
                             + ("The object is between the fingers now, so this grasps it." if inside else
                                (f"The object may be {off_mm - gap_mm:.0f} mm outside the fingers, so a finger lands on the object and pushes it away; the grasp fails."
                                 if STEER_TEXT >= 6 else
                                 f"The hand is {dist_mm - gap_mm:.0f} mm too far off, so a finger lands on the object and pushes it away; the grasp fails."
                                 if STEER_TEXT >= 4 else f"The hand is {dist_mm - gap_mm:.0f} mm too far off, so a finger may hit the object.")))
        if can_look:
            ch["look"] = "Stop for about 0.5 s and take one photo with the wrist camera to find the object again. Nothing is grasped yet."
        unc_mm = v_mm * age if age >= 0.15 else 0.0
        if STEER_TEXT >= 7 and unc_mm >= 1:                       # 7：見えていない時、寄せる先が古い見当だと書き、撮れば位置が分かり直すと書く（6 は撮るを一度も選ばなかった）
            for k_ in ("over", "lead"):
                if k_ in ch:
                    ch[k_] = ch[k_].replace("Nothing is grasped yet.", f"It moves toward an old estimate that may be {unc_mm:.0f} mm off. Nothing is grasped yet.")
            if can_look:
                ch["look"] = ("Stop for about 0.5 s and take one photo with the wrist camera. Then the object's position is known again "
                              "to about 2 mm, and the hand can line up with it. Nothing is grasped yet.")
        if STEER_TEXT == 9:                                      # 9：7 の文のまま、問いに「位置が不確かなら先に確かめる」だけ足す（撮るかを選べるか）
            q = "If the object's position is unsure, make it sure first. What should the hand do next?"
        if STEER_TEXT in (3, 5, 6, 7):
            q = "What should the hand do next?"
    if HES_TEXT >= 2 and hes >= HESITATE_S and STEER_TEXT >= 2 and gap_mm is not None:   # 3 秒以上迷った：待つと何が起きるかを選択肢に書く（10/23。状態に書くだけでは 3 秒以上で下りたのは 17/36）
        for k_ in ("over", "lead"):
            if k_ in ch:
                ch[k_] = ch[k_].replace("Nothing is grasped yet.", f"Nothing is grasped yet, and the hand keeps waiting after {hes:.1f} s; visitors see it hesitate (bad).")
        if "descend" in ch and not inside:
            ch["descend"] = (f"Go down {desc_mm:.0f} mm while following the object, then close the fingers. The hand may be {off_mm - gap_mm:.0f} mm off, "
                             "so a finger may touch the object; if the grasp misses, the hand tries again.")
    if (STEER_TEXT == 8 or (LOOK_TEXT and can_look)) and gap_mm is not None:   # LOOK_TEXT：撮れる場面だけシューティング式（ほかは満点の文章 5 のまま）                 # 8：シューティングの改良と同じ形。選択肢は短い結果＋評価の言葉、指示文に決め方の順を書く
        unc = v_mm * age if age >= 0.15 else 0.0
        sure = unc <= gap_mm
        off = dist_mm + unc
        state = (f"The robot hand is chasing {KIND_EN.get(kind, 'an object')} to grasp it. Hand {h_mm:.0f} mm above the table, "
                 f"{dist_mm:.0f} mm off from above the object. Finger margin {gap_mm:.0f} mm on each side. Object speed {v_mm:.0f} mm/s. "
                 + ("Position: sure (seen now)." if unc < 1 else
                    f"Position: {'sure' if sure else 'unsure'} (not seen for {age:.1f} s, may be off {unc:.0f} mm)."))
        bad = " (unsure position, bad)" if LOOK_TEXT < 2 else f" toward an old position that may be {unc:.0f} mm off (bad)"   # 2：古い見当だとはっきり
        ch = {"over": "move above the object" + (" (good)" if sure else bad)}
        if v_mm >= 3:
            ch["lead"] = "move ahead of the object" + (" (good)" if sure else bad)
        if near:
            ch["descend"] = "go down and grasp (best)" if off <= gap_mm else f"go down and grasp: {off - gap_mm:.0f} mm outside the fingers (fails)"
        if can_look:
            ch["look"] = ("stop and photograph with the wrist camera" + (" (position becomes sure, good)" if not sure else " (wastes 0.5 s, bad)")
                          if LOOK_TEXT < 2 else
                          ("photograph with the wrist camera to make the position sure (best)" if not sure else "stop and photograph (wastes 0.5 s, bad)"))
        q = SHOOT_Q
    if len(ch) == 1:
        return None, next(iter(ch))
    body = {"state": state, "question": q, "choices": ch}
    return _STEER_POOL.submit(KA.post, body, 2.0), None


def close_ask(kind, v_mm, w_mm, age, extra="", dist_mm=0.0):
    """下り切って物の真上にそろった時、今閉じるか、もう少し追うかを d1-3B に聞く（裏の糸で。10/22 から）。
    止まっている物は「閉じる」しか残らないので聞かない。戻り値 (先の答え, 1 つだけ残った選択肢)。"""
    global _STEER_POOL
    import concurrent.futures as cf
    if _STEER_POOL is None:
        _STEER_POOL = cf.ThreadPoolExecutor(1)
    if v_mm < 3 and not D1_CLOSE_STILL:
        return None, "close"
    gap = max(0.0, (OPEN_MM - w_mm) / 2)                         # 指の内側と物のすき間（片側）
    seen = "It is visible in the cameras now." if age < 0.15 else f"It has not been seen for {age:.1f} s."
    state = (f"A robot hand is right above {KIND_EN.get(kind, 'an object')} on a cutting mat, at the grasp height, with the open fingers around it. "
             f"The object is {w_mm:.0f} mm wide and the open fingers leave about {gap:.0f} mm on each side. "
             f"The object is still moving at {v_mm:.0f} mm/s. " + seen + extra)
    ch = {"close": f"Close the fingers now. While they close (about {CLOSE_T:.1f} s) the object moves about {v_mm * CLOSE_T:.0f} mm.",
          "follow": f"Keep following it at this height for {STEER_EVERY:.1f} s more, then decide again."}
    q = "Should the hand close the fingers now to grasp the moving object?"
    if CLOSE_TEXT >= 1:                                          # 選んだ結果を書く（10/22。追い方の問いで、結果を書く＋中立の問いが 90/90 だった）
        gap = max(2.0, (OPEN_MM - w_mm) / 2 - 1.0)
        unc = v_mm * age if age >= 0.15 else 0.0                 # 見えていない間に、見当がずれているかもしれない量
        off = dist_mm + unc
        state = (f"A robot hand is chasing {KIND_EN.get(kind, 'an object')} on a cutting mat, at the grasp height with the fingers open. "
                 f"The hand is {dist_mm:.0f} mm off from right above the object, and the open fingers leave {gap:.0f} mm on each side of it. "
                 f"The object moves at {v_mm:.0f} mm/s. " + seen
                 + (f" So its position may be off by about {unc:.0f} mm." if unc >= 1 else "") + extra)
        ch = {"close": "Close the fingers now. " + ("The object is between the fingers, so this grasps it." if off <= gap else
                                                    f"The object may be {off - gap:.0f} mm outside the fingers, so a finger lands on it and the grasp fails."),
              "follow": f"Keep following it for {STEER_EVERY:.1f} s more and decide again. Nothing is grasped yet."}
        if CLOSE_TEXT >= 3:                                      # 3・4：失敗をはっきり、追う方の結果（そろい直す）も書く（10/22。1 は追うべき時に 29/70 しか追わなかった）
            ch = {"close": "Close the fingers now. " + ("The object is between the fingers, so this grasps it." if off <= gap else
                                                        f"The object is {off - gap:.0f} mm outside the fingers, so a finger lands on the object "
                                                        "and pushes it away; the grasp fails."),
                  "follow": (f"Keep following it for {STEER_EVERY:.1f} s more so the hand lines up with it again, then decide again. "
                             "Nothing is grasped yet.")}
        if CLOSE_TEXT >= 6:                                      # 6：先読みしているので、見えない間のずれは「速さの測り違い×秒＋測る誤差」（速さ全部×秒は大きすぎ、ずっと追った）
            unc = UNC_K * v_mm * age + UNC_BASE if age >= 0.15 else 0.0
            off = dist_mm + unc
        if CLOSE_TEXT >= 5:                                      # 5：シューティングの改良と同じ形
            state = (f"The robot hand is at the grasp height with the fingers open around {KIND_EN.get(kind, 'an object')}. "
                     f"{dist_mm:.0f} mm off from above the object. Finger margin {gap:.0f} mm on each side. Object speed {v_mm:.0f} mm/s. "
                     + ("Position: sure (seen now)." if unc < 1 else
                        f"Position: {'sure' if unc <= gap else 'unsure'} (not seen for {age:.1f} s, may be off {unc:.0f} mm)."))
            ch = {"close": "close the fingers now (best)" if off <= gap else f"close the fingers now: {off - gap:.0f} mm outside the fingers (fails)",
                  "follow": "keep following and line up again (good)"}
            if CLOSE_TEXT == 6 and age >= 0.15:                  # 見えていない時は、追っても位置は分からないまま（正直に書く）。7 は見積もりだけ 6 と同じで、これを書かない
                ch["follow"] = "keep following the old estimate without seeing it; the position stays unsure (bad)"
            q = SHOOT_Q
        if CLOSE_TEXT in (1, 3):
            q = "What should the hand do next?"
        elif CLOSE_TEXT == 4:
            q = "Should the hand close the fingers now?"
    body = {"state": state, "question": q, "choices": ch}
    return _STEER_POOL.submit(KA.post, body, 2.0), None


SHOOT_Q = ("Never pick an action marked 'fails'. Prefer 'best', then 'good'. "
           "If the object's position is unsure, pick the action that makes it sure.")   # シューティングの指示文と同じ形：決め方の順をはっきり書く


DIRS = {"right": (1, 0), "front-right": (0.7071, 0.7071), "front": (0, 1), "front-left": (-0.7071, 0.7071), "left": (-1, 0),
        "back-left": (-0.7071, -0.7071), "back": (0, -1), "back-right": (0.7071, -0.7071)}   # 根元の座標の +x・+y の向き（説明の言葉は目安で、数字で選ばせる）
STEER_PRED = 0.5                               # 向きを選ぶ時、物の何秒先の位置までの距離も説明に書くか


def steer_ask_dir(kind, h_mm, cur_xy, tgt, tgt_pred, v_mm, age, near, desc_mm, extra="", gap_mm=None):
    """動く向きを d1-3B に聞く。cur_xy：今の道具の点、tgt：物の真上（道具の点の行き先）、tgt_pred：STEER_PRED 秒後の物の真上（mm）。
    選択肢ごとに、その向きへ 1 歩動いた後の、物の真上と STEER_PRED 秒後の位置までの距離を書く。戻り値 (先の答え, 1 歩の長さ mm)。"""
    global _STEER_POOL
    import concurrent.futures as cf
    if _STEER_POOL is None:
        _STEER_POOL = cf.ThreadPoolExecutor(1)
    step = float(np.clip(np.linalg.norm(tgt - cur_xy), 5.0, max(20.0, v_mm * STEER_EVERY * 2)))
    seen = "It is visible in the cameras now." if age < 0.15 else f"It has not been seen for {age:.1f} s."
    state = (f"A robot hand is following {KIND_EN.get(kind, 'an object')} lying on a cutting mat to pick it up. "
             f"The hand is {h_mm:.0f} mm above the table and {np.linalg.norm(tgt - cur_xy):.0f} mm away from the spot right above the object. "
             f"The object is moving at {v_mm:.0f} mm/s. " + seen + extra)
    ch = {"wait": f"Stay here. Then {np.linalg.norm(tgt - cur_xy):.0f} mm from above the object now, "
                  f"{np.linalg.norm(tgt_pred - cur_xy):.0f} mm from where it will be in {STEER_PRED:.1f} s."}
    for k_, d_ in DIRS.items():
        p_ = cur_xy + step * np.array(d_)
        if not on_mat(p_, margin=-40.0):                        # 指がマットの外 40mm より先へ出る向きは外す
            continue
        ch[k_] = (f"Move {step:.0f} mm {k_}. Then {np.linalg.norm(tgt - p_):.0f} mm from above the object now, "
                  f"{np.linalg.norm(tgt_pred - p_):.0f} mm from where it will be in {STEER_PRED:.1f} s.")
    if near:
        ch["descend"] = (f"Go down now and grasp it ({desc_mm:.0f} mm down, slowly)."
                         + (" Even if it is not visible now, go down to the current estimate; if the grasp misses, it is retried once." if D1_NOEXC else ""))
    q = "Which way should the hand move next to grasp the moving object quickly and surely?"
    if DIR_TEXT >= 1:                                            # 選んだ結果を数字で書き、問いは中立に（追い方の問いで 90/90 だった書き方）
        gap = gap_mm if gap_mm is not None else 3.0
        d_now = float(np.linalg.norm(tgt - cur_xy))
        state = (f"A robot hand is chasing {KIND_EN.get(kind, 'an object')} on a cutting mat to grasp it. "
                 f"The fingertips are {h_mm:.0f} mm above the table. The hand is {d_now:.0f} mm off from right above the object, "
                 f"and the open fingers leave {gap:.0f} mm on each side of it. The object moves at {v_mm:.0f} mm/s. " + seen + extra)
        for k_ in list(ch):
            if k_ == "descend":
                continue
            p_ = cur_xy if k_ == "wait" else cur_xy + step * np.array(DIRS[k_])
            a_, b_ = np.linalg.norm(tgt - p_), np.linalg.norm(tgt_pred - p_)
            ch[k_] = (("Stay here." if k_ == "wait" else f"Move {step:.0f} mm {k_}.")
                      + f" Then the hand is {a_:.0f} mm off from the object now and {b_:.0f} mm off from where it will be in {STEER_PRED:.1f} s. Nothing is grasped yet.")
        if "descend" in ch:
            ch["descend"] = (f"Go down {desc_mm:.0f} mm while following the object, then close the fingers. "
                             + ("The object is between the fingers now, so this grasps it." if d_now <= gap else
                                f"The object is {d_now - gap:.0f} mm outside the fingers, so a finger lands on the object and pushes it away; the grasp fails."))
        q = "What should the hand do next?"
        if DIR_TEXT >= 2:                                        # 2：シューティングの改良と同じ形。いちばん近づく向きは good、離れる向きは bad
            dists = {k_: float(np.linalg.norm(tgt - (cur_xy if k_ == "wait" else cur_xy + step * np.array(DIRS[k_])))) for k_ in ch if k_ != "descend"}
            best_d = min(dists.values())
            for k_, d_ in dists.items():
                ch[k_] = (("stay" if k_ == "wait" else f"move {k_}") + f": then {d_:.0f} mm off"
                          + (" (good)" if d_ <= best_d + 2.0 else " (farther, bad)" if d_ > d_now else ""))
            if "descend" in ch:
                ch["descend"] = "go down and grasp (best)" if d_now <= gap else f"go down and grasp: {d_now - gap:.0f} mm outside the fingers (fails)"
            q = SHOOT_Q
    body = {"state": state, "question": q, "choices": ch}
    return _STEER_POOL.submit(KA.post, body, 2.0), step


HESITATE_S = float(os.environ.get("TR_HESITATE_S", 3.0))   # 物の近くで下りられずに迷う時間の上限（秒）。超えたら見えていなくても下りてつかみに行く
                                               # （10/22。展示会などで迷って見えないように。迷っている時も、たいてい下りればつかめる）
HIDE_WRIST = os.environ.get("TR_HIDE_WRIST", "1") == "1"   # 1：下りている途中で三脚から隠れたら、その場で止まって手首で 1 枚撮る。
                                               # 前は腕が止まるのを待つ間に STEPS[0] へ上がり、また見えて下り…を時間切れ（10 秒）まで繰り返した（10/22、取ろうとしてやめるのを繰り返していた）
NEAR_Z = 150.0                                 # 聞き直す時の指先の高さ（机から mm）
LOOK_FROM_BOX = os.environ.get("TR_LOOK_FROM_BOX", "1") == "1"   # 1：2 個目からは箱の上で三脚を見る（見つからなければ LOOK へ）


def long_dir(cnt, h=TV.OBJ_H):
    """輪郭（三脚の画素）を机の面へ写した時の、長い向きの単位ベクトル（根元の座標）。"""
    pts = np.array([TV.to_table(p, h) for p in cnt.reshape(-1, 2).astype(float)], np.float32)
    b = cv2.boxPoints(cv2.minAreaRect(pts))
    e1, e2 = b[1] - b[0], b[2] - b[1]
    v = e1 if np.linalg.norm(e1) >= np.linalg.norm(e2) else e2
    return v / max(np.linalg.norm(v), 1e-9)


def in_box(xy, box, half_w=70.0):
    """箱の中（もう運んだ物）か。箱の真ん中から、長い向きに長さの半分・横に half_w mm 以内。"""
    if not box:
        return False
    d = np.asarray(xy, float) - np.asarray(box["center"], float)
    ax = np.asarray(box["axis"], float)
    return abs(d @ ax) < box["length"] / 2 + 10 and abs(d @ np.array([-ax[1], ax[0]])) < half_w


def to_f(o):
    """三脚の物を、つかむ関数（pick_by_camera.grasp・plan_ai.grip）が使う形にする。"""
    k = o["kind"]
    return {"center": np.array([o["xy"][0], o["xy"][1], H_NOM[k]]), "longv": long_dir(o["contour"]), "size": o["size"], "h": H_NOM[k],
            "kind": k, "round": k in ("ペン", "銀の棒"), "on_mat": 1.0, "color": o.get("color")}


def look(r, rec, tag, log=print):
    """今の三脚の写真で物を見つけ、種類を決める。"""
    tcp = r.getActualTCPPose()
    objs, img = TV.objects(tcp, after=time.time() + 0.15)       # 腕が止まってから撮った写真を使う
    if objs is None:
        raise SystemExit("三脚の写真が来ない（cam_view.py --cam2 が動いているか）")
    cv2.imwrite(os.path.join(rec, f"{tag}.jpg"), img)
    TV.kinds(img, objs, log=log)
    for o in objs:
        o["color"] = mean_lab(img, o["contour"])
    return objs


def mean_lab(img, cnt):
    """輪郭の中の色の平均（Lab）。"""
    m = np.zeros(img.shape[:2], np.uint8)
    cv2.drawContours(m, [cnt], -1, 255, -1)
    m = cv2.erode(m, np.ones((3, 3), np.uint8))
    return np.array(cv2.mean(cv2.cvtColor(img, cv2.COLOR_BGR2LAB), mask=m if m.any() else None)[:3])


def follow(c, r, f, rot, rec, steps, log=print):
    """物の上で STEPS の高さへ順に下りながら、三脚で物の位置を測り直して f["center"]・f["longv"] を寄せる。"""
    for z in STEPS:
        # 下りながら手首も物の向きに回す（つかむ時と同じ計算 PB.aim。10/20 夜、前は真上まで向きを変えず、つかむ直前に回した）
        _, rv, tx, ty, _, _, _ = PB.aim(c, r, r.getActualTCPPose(), f["center"], f["longv"], z)
        c.moveL([tx / 1000, ty / 1000, (TABLE_Z + z) / 1000] + rv, SPEED, ACCEL)
        tcp = r.getActualTCPPose()
        objs, img = TV.objects(tcp, after=time.time() + 0.15)
        near = [o for o in (objs or []) if np.linalg.norm(o["xy"] - f["center"][:2]) < FOLLOW_MM]
        row = {"z": z, "tcp": list(tcp), "before": f["center"][:2].tolist()}
        if near:
            o = min(near, key=lambda o: np.linalg.norm(o["xy"] - f["center"][:2]))
            d = o["xy"] - f["center"][:2]
            f["center"][:2] = o["xy"]
            f["longv"] = long_dir(o["contour"])
            row["after"] = o["xy"].tolist()
            log(f"   机から {z:.0f}mm：三脚で測り直した → 動いた分 dx {d[0]:+.1f} dy {d[1]:+.1f}mm")
        else:
            log(f"   机から {z:.0f}mm：三脚から見えない（腕の陰）ので、前の位置のまま")
        if img is not None:
            cv2.imwrite(os.path.join(rec, f"{f['kind_tag']}_z{int(z)}.jpg"), img)
        steps.append(row)


def same_size(o, f):
    """三脚の物 o の大きさが、最初に測った f["size"]（幅, 長さ mm）と合うか。腕に一部が隠れた物は小さく出て、真ん中もずれるので使わない
    （10/22、シミュレーターで止まったペンを「19mm 動いた」と測った）。"""
    w, l = TV.size_mm(o["contour"])
    w0, l0 = f["size"]
    return abs(l - l0) < max(8.0, 0.15 * l0) and abs(w - w0) < max(5.0, 0.3 * w0)


def end_center(o, f, tcp):
    """腕に片方の端が隠れて短く見える物の真ん中（10/22）。幅が合い、長さが最初の 30〜100% なら、
    腕から遠い方の見えている端から、最初の長さの半分だけ腕の方へ戻った所を真ん中とする。使えなければ None。"""
    w0, l0 = f["size"]
    if not END_ON or l0 < END_RATIO * w0:                                      # 細長い物（ペン・銀の棒）だけ。消しゴムは隠れると長い向きが入れ替わり、23mm ずれた（10/22）
        return None
    pts = np.array([TV.to_table(p) for p in o["contour"].reshape(-1, 2).astype(float)])
    v = np.asarray(f["longv"], float)[:2]                        # 向きは最初に測った長い向き（見えている部分の向きは使わない）
    v = v / max(np.linalg.norm(v), 1e-9)
    if abs(long_dir(o["contour"]) @ v) < math.cos(math.radians(20)):
        return None
    s_ = (pts - pts.mean(0)) @ v
    q_ = (pts - pts.mean(0)) @ np.array([-v[1], v[0]])
    l, w = s_.max() - s_.min(), q_.max() - q_.min()
    if abs(w - w0) >= max(5.0, 0.3 * w0) or not (0.3 * l0 <= l < l0 + max(8.0, 0.15 * l0)):
        return None
    ends = [pts.mean(0) + v * s_.min(), pts.mean(0) + v * s_.max()]
    arm = np.array(tcp[:2]) * 1000.0
    far = max(ends, key=lambda e: np.linalg.norm(e - arm))
    toward = (arm - far) @ v
    return far + v * np.sign(toward if toward else 1.0) * l0 / 2, v


def axis_keep(f, v):
    """KEEP_AXIS の時、測った長い向き v が今の向きから AXIS_JUMP° より大きく変わったら、前の向きのまま使う（位置だけ使う）。
    隠れて一部しか見えない・指と重なった塊は、向きが 90° 回って出やすい。物は追う間にほとんど回らない（10/23、無理な縦の向きでつかもうとしていた）。"""
    v = np.asarray(v, float)
    if not KEEP_AXIS or f.get("longv") is None:
        return v
    old = np.asarray(f["longv"][:2], float)
    c = abs(float(np.dot(v[:2], old))) / max(np.linalg.norm(v[:2]) * np.linalg.norm(old), 1e-9)
    if c < math.cos(math.radians(AXIS_JUMP)):
        f["axis_kept"] = f.get("axis_kept", 0) + 1
        return np.array(f["longv"], float)
    return v


def remeasure(f, tcp, row, log, gate=FOLLOW_MM, guess=None):
    """三脚の最新の写真で f の位置と向きを測り直す。測れたら True。
    全部見えている物（大きさが合う）はその真ん中、片方の端が腕に隠れた物は見えている端から出した真ん中を使う（10/22）。"""
    objs, img = TV.objects(tcp, after=0, cut=True)               # 待たずに、いちばん新しい写真を使う。腕の先の画素は消して探す
    best = None
    row["seen"] = []
    for o in objs or []:
        w_, l_ = TV.size_mm(o["contour"])
        row["seen"].append([round(float(o["xy"][0]), 1), round(float(o["xy"][1]), 1), round(w_, 1), round(l_, 1)])   # 見えた塊（x, y, 幅, 長さ）。外した理由を後で調べる
        if SAME_COLOR > 0 and f.get("color") is not None and np.linalg.norm(mean_lab(img, o["contour"])[1:] - f["color"][1:]) > SAME_COLOR:   # 明るさ（L）は腕の陰で変わるので、色合い（a・b）だけ比べる
            continue
        if same_size(o, f):
            c, v, how = o["xy"], long_dir(o["contour"]), "全部"
        else:
            e = end_center(o, f, tcp)
            if e is None:
                continue
            (c, v), how = e, "端から"
        if not on_mat(c):
            continue
        d = np.linalg.norm(np.asarray(c) - (f["center"][:2] if guess is None else guess))
        if d < gate and (best is None or d < best[0]):
            best = (d, c, v, how)
    if best is None:
        return False, img
    _, c, v, how = best
    f["center"][:2] = c
    f["longv"] = axis_keep(f, v)
    row["after"], row["how"] = list(map(float, c)), how
    return True, img


def wrist_xy(px, tcp, h):
    """手首のカメラの画素 → 机から h mm の面の上の位置（根元の座標 mm）。locate.pixel_to_table の面を物の上の面にしたもの。"""
    n = cv2.undistortPoints(np.array([[px]], np.float64), L.K, L.DIST).reshape(2)
    RB, tB = PB.rotvec_to_R(tcp[3:]), np.array(tcp[:3]) * 1000.0
    o = RB @ L.TC + tB
    d = RB @ L.RC @ np.array([n[0], n[1], 1.0])
    s_ = np.dot(L.G0 + L.GN * h - o, L.GN) / np.dot(d, L.GN)
    return (o + s_ * d)[:2]


WRIST_LIVE = os.path.join(HERE, "cam_shots", "wrist_live.jpg")   # cam_view.py（CAM_WRIST_EVERY）が置く手首の最新の写真


def wrist_live(r, hist, seen_t):
    """手首の最新の写真と、撮った時の腕の姿勢。seen_t（前に使った写真の時刻）より新しい写真が無ければ (None, None, seen_t)。
    腕は動いているので、写真の時刻に近い姿勢を hist（[(時刻, 姿勢)]）から選ぶ。シミュレーターでは差し替える。"""
    try:
        mt = os.path.getmtime(WRIST_LIVE)
    except OSError:
        return None, None, seen_t
    if mt <= seen_t:
        return None, None, seen_t
    img = cv2.imread(WRIST_LIVE)
    if img is None or not hist:
        return None, None, seen_t
    tcp = min(hist, key=lambda h_: abs(h_[0] - (mt - WRIST_LAG)))[1]
    return img, tcp, mt


def wrist_measure(img, tcp, f, guess, gate=None):
    """手首の写真で、見当 guess のまわり gate mm 以内の大きさの合う物。(真ん中, 長い向き, 幅, 長さ, 見当からの距離, 一部だけか) か None。
    PARTIAL の時は、写真の端で切れた・指に隠れて短く見える物も、幅と向きが合えば「一部だけ」として返す（横の位置だけ使う）。"""
    best = None
    H_, W_ = img.shape[:2]
    for cnt in M.find_objects(img)[0]:
        x_, y_, w_, h_ = cv2.boundingRect(cnt)
        cut = x_ <= 3 or y_ <= 3 or x_ + w_ >= W_ - 3 or y_ + h_ >= H_ - 3
        if WRIST_BORDER and cut and not (PARTIAL or WRIST_GRASP):
            continue
        pts = np.array([wrist_xy(p, tcp, f["h"]) for p in cnt.reshape(-1, 2)[::4].astype(float)], np.float32)
        if len(pts) < 5:
            continue
        (cx, cy), (a, b), _ = cv2.minAreaRect(pts)
        w, l = sorted((a, b))
        w0, l0 = f["size"]
        bx = cv2.boxPoints(cv2.minAreaRect(pts))
        e1, e2 = bx[1] - bx[0], bx[2] - bx[1]
        v = e1 if np.linalg.norm(e1) >= np.linalg.norm(e2) else e2
        v = v / max(np.linalg.norm(v), 1e-9)
        whole = not WRIST_GRASP and not (WRIST_BORDER and cut) and abs(l - l0) <= max(10.0, 0.2 * l0) and abs(w - w0) <= max(6.0, 0.35 * w0)
        part = False
        if not whole:
            lv = np.asarray(f["longv"][:2], float)
            lv = lv / max(np.linalg.norm(lv), 1e-9)
            # 一部だけ：幅が合い、見えている長さが本当より短く、細長くて向きが合う（20° 以内）。切れた向きの位置は分からないので横だけ使う
            part = ((PARTIAL or WRIST_GRASP) and abs(w - w0) <= max(4.0, 0.25 * w0) and (WRIST_GRASP or l < l0 - 5.0) and l >= 1.3 * w
                    and abs(float(np.dot(v, lv))) >= math.cos(math.radians(20)))
            if not part:
                continue
        lv_ = np.asarray(f["longv"][:2], float)
        lv_ = lv_ / max(np.linalg.norm(lv_), 1e-9)
        dvec = np.array([cx - guess[0], cy - guess[1]])
        d = abs(float(dvec[0] * -lv_[1] + dvec[1] * lv_[0])) if part else float(np.hypot(*dvec))   # 一部だけの時は横のずれだけで比べる
        if part and abs(float(np.dot(dvec, lv_))) > max(0.0, (f["size"][1] - l) / 2) + (WRIST_GATE if gate is None else gate):
            continue                                             # 長い向きにも、見えた部分が物のあるはずの所に重ならなければ別の物（10/22 消しゴムと逆へ行った）
        if d < (WRIST_GATE if gate is None else gate) and (best is None or (not part and best[5]) or (part == best[5] and d < best[4])):
            best = (np.array([cx, cy]), v, w, l, d, part)
    return best


def wrist_fix(r, f, guess, rec, log=print):
    """止まって手首のカメラで 1 枚撮り、見当 guess のまわりで大きさの合う物の真ん中と長い向きを返す（10/22。三脚は腕の真下の物が見えない）。
    見つからなければ None。"""
    tcp = r.getActualTCPPose()
    img = L.grab_now(settle=0.1)
    if img is None:
        return None
    best = wrist_measure(img, tcp, f, guess)
    cv2.imwrite(os.path.join(rec, f"{f['kind_tag']}_wrist.jpg"), cv2.resize(img, (960, 540)))
    if best is None:
        log("   手首のカメラ：見当のまわりに大きさの合う物が無い")
        return None
    c_, v, w, l, d, part = best
    if part:                                                     # 一部だけ：横の位置だけ直す（長い向きは見当のまま）
        lv = np.asarray(f["longv"][:2], float) / max(np.linalg.norm(f["longv"][:2]), 1e-9)
        nrm = np.array([-lv[1], lv[0]])
        c_ = np.asarray(guess, float) + nrm * float(np.dot(c_ - np.asarray(guess, float), nrm))
        v = lv
        log(f"   手首のカメラで横だけ直した（一部だけ見えた、幅 {w:.0f}・見えた長さ {l:.0f}mm）：横に {d:.1f}mm")
        return c_, v
    log(f"   手首のカメラで直した：見当から {d:.1f}mm（幅 {w:.0f}・長さ {l:.0f}mm）")
    return c_, v


def near_look(c, r, objs, k, rot, box, rec, log=print):
    """三脚で「それ以外」になった物の上へ順に行き、手首のカメラで種類を d1-3B に聞き直す。k と答えた物（位置は手首で測り直す）か None。
    三脚から遠い物は小さく写り（大きいマットの奥の消しゴムは約 25 画素）、d1-3B が「それ以外」と答えた（10/22、シミュレーター）。"""
    key = {v: k_ for k_, v in KA.NAME.items()}[k]
    cand = [o for o in objs if o["kind"] is None and not in_box(o["xy"], box) and on_mat(o["xy"])]
    cand.sort(key=lambda o: -(o["kind_p"] or {}).get(key, 0.0))
    for o in cand:
        c.moveL([o["xy"][0] / 1000, o["xy"][1] / 1000, (TABLE_Z + NEAR_Z) / 1000] + rot, SPEED, ACCEL)
        tcp = r.getActualTCPPose()
        img = L.grab_now(settle=0.1)
        if img is None:
            continue
        cv2.imwrite(os.path.join(rec, f"near_{k}_{o['xy'][0]:.0f}_{o['xy'][1]:.0f}.jpg"), cv2.resize(img, (960, 540)))
        best = None
        for cnt in M.find_objects(img)[0]:
            pts = np.array([wrist_xy(p, tcp, 10.0) for p in cnt.reshape(-1, 2)[::4].astype(float)], np.float32)
            if len(pts) < 5:
                continue
            (cx, cy), (a, b), _ = cv2.minAreaRect(pts)
            d = np.hypot(cx - o["xy"][0], cy - o["xy"][1])
            if d < WRIST_GATE and (best is None or d < best[0]):
                best = (d, np.array([cx, cy]), tuple(sorted((a, b))), cnt)
        if best is None:
            log(f"   x {o['xy'][0]:.0f} y {o['xy'][1]:.0f} の物：手首のカメラで見つからない")
            continue
        d, xy, size, cnt = best
        f = {"size": size, "contour": cnt}
        choices = KA.options(f)
        if len(choices) == 1:
            log(f"   x {o['xy'][0]:.0f} y {o['xy'][1]:.0f} の物：手首で長さ {size[1]:.0f}・幅 {size[0]:.0f}mm、どの種類にも合わない")
            continue
        crop = KA.crop_upright(img, cnt)
        ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
        res = KA.post({"state": "Close-up image from the robot's wrist camera of one object lying on a dark cutting mat. "
                                f"Measured by the robot's camera: seen from above it is {size[1]:.0f} mm long and {size[0]:.0f} mm wide.",
                       "question": "What is this object?", "choices": choices,
                       "image": __import__("base64").b64encode(buf.tobytes()).decode("ascii")})
        name = KA.NAME[res["choice"]] if res["probs"][res["choice"]] >= KA.P_MIN else None
        top = sorted(res["probs"].items(), key=lambda kv: -kv[1])[:2]
        log(f"   手首で聞き直した（x {o['xy'][0]:.0f} y {o['xy'][1]:.0f}）：d1-3B {name or 'それ以外'}（"
            + "、".join(f"{KA.NAME.get(a) or 'それ以外'} {p:.2f}" for a, p in top) + f"、長さ {size[1]:.0f}・幅 {size[0]:.0f}mm）")
        if name == k:
            o["kind"], o["kind_p"], o["xy"] = k, res["probs"], xy
            return o
    return None


def on_mat(xy, margin=10.0):
    """根元の座標 xy（mm）がマットの上（端から margin mm 内）か。見当や測った位置がマットの外なら使わない（10/22、シミュレーターで
    見当が外へ逃げ、腕が三脚の脚に当たって保護停止した）。"""
    poly = np.array([TV.to_table(p, 0.0) for p in TV.MAT_PX], np.float32)
    return cv2.pointPolygonTest(poly, (float(xy[0]), float(xy[1])), True) >= margin


def close_moving(c, r, cur, vel, ov, goal_rv, force, log, w_obj=None):
    """腕を止めずに、物の速さ ov（mm/秒）で追い続けながらハンドを閉じる。指が止まったら開き量（mm）を返す（10/22）。
    高さは今のまま。加速度は A_MAX まで。最長 CLOSE_MOVE_S 秒。
    指が物に触れたら（開き量が物の幅 w_obj ＋2mm 以下）、追うのをやめて止まる（触れた後も動くと物を引きずって落とす。10/22 W_p）。"""
    h = PB.HandClient()
    h.grip_start(force)
    w0 = h.pos_mm()
    t, last, still, w, touched = 0.0, None, 0, None, False
    while t < CLOSE_WAIT_S:                                      # 追うのは CLOSE_MOVE_S まで。その後は止まって、指が閉じ切るのを待つ
        t0 = c.initPeriod()
        want = np.zeros(3)
        if not touched and t < CLOSE_MOVE_S:
            want[:2] = np.asarray(ov, float) / 1000
        dv = want - vel
        if np.linalg.norm(dv) > A_MAX * SERVO_DT:
            dv *= A_MAX * SERVO_DT / np.linalg.norm(dv)
        vel = vel + dv
        nxt = cur.copy()
        nxt[:3] = cur[:3] + vel * SERVO_DT
        nxt[3:] = goal_rv
        c.servoL(list(nxt), 0.0, 0.0, SERVO_DT, 0.1, 300)
        c.waitPeriod(t0)
        cur, t = nxt, t + SERVO_DT
        if int(round(t / SERVO_DT)) % 5 == 0:                    # 開き量は 0.1 秒ごとに読む（実機は読むのに時間がかかる）
            w = h.pos_mm()
            if w is not None and w_obj is not None and w <= w_obj + 2.0 and (not TOUCH_W0 or w0 is None or w <= w0 - 2.0):
                touched = True                                   # 指が物に触れた：ここから先は止まる
            moved = w is not None and (w0 is None or w <= w0 - 2.0 or touched)   # 実機のハンドは閉じ始めが遅く、0.3 秒で 0.6mm しか動かない。
            if moved and last is not None and abs(w - last) < 0.3 and t > 0.15 and np.linalg.norm(vel) < 1e-3:   # まだ動き出していないのを「止まった」と取らない（10/23 実機：閉じ切る前に持ち上げて外した）
                still += 1
                if still >= 2:
                    break
            else:
                still = 0
            last = w
    if hasattr(h, "report_offset"):
        h.report_offset()
    if hasattr(h, "close_port"):
        h.close_port()
    if np.linalg.norm(vel) > 1e-4:                               # まだ動いていたら、止まってから返す
        for _ in range(int(np.linalg.norm(vel) / (A_MAX * SERVO_DT)) + 1):
            t0 = c.initPeriod()
            dv = -vel
            if np.linalg.norm(dv) > A_MAX * SERVO_DT:
                dv *= A_MAX * SERVO_DT / np.linalg.norm(dv)
            vel = vel + dv
            cur = cur.copy()
            cur[:3] = cur[:3] + vel * SERVO_DT
            c.servoL(list(cur), 0.0, 0.0, SERVO_DT, 0.1, 300)
            c.waitPeriod(t0)
        w = h.pos_mm()
    log(f"   追いながら閉じた：{t:.2f} 秒（触れた {'あり' if touched else 'なし'}）、開き量 {w if w is None else round(w, 1)}mm")
    return w


def follow_servo(c, r, f, rec, steps, log=print, z_end=None, force=None):
    """今の位置から servoL で止まらずに物の上へ行き、そのまま zE mm まで下りる。その間ずっと MEAS_EVERY 秒ごとに三脚で
    物の位置と向きを測り直し、物の速さも出して先回りする（10/22。前は真上まで moveL で行き、その間に物が 30〜40mm 動いて見失った）。
    物の真上から APPROACH_MM 以内に来るまでは STEPS[0] mm の高さで寄せ、来たら DESC_V で下りる。"""
    zE = STEPS[-1] if z_end is None else z_end                   # 下り切る高さ（机から mm）。TO_GRIP の時は、つかむ高さまで追いながら下りる
    z0, z1 = (TABLE_Z + STEPS[0]) / 1000, (TABLE_Z + zE) / 1000
    z_clear = (TABLE_Z + f["h"] + CLEAR_MM) / 1000 if (z_end is not None and CLEAR_MM > 0) else None   # 指先が物の上の面より上にいる高さ
    z_low = (TABLE_Z + f["h"] + 15.0) / 1000                     # 指先がこれより下なら、物は指の間で、三脚からは腕の陰になりやすい
    fine = (max(FINE_MIN, (OPEN_MM - float(f["size"][0])) / 2) if FINE_REAL else max(2.0, (OPEN_MM - float(f["size"][0])) / 2 - 1.0)) / 1000   # 指の内側と物のすき間（片側、1mm の余裕を引く）。ずれがこれ以内なら下りても当たらない
    cur = np.array(r.getActualTCPPose(), float)
    vel = np.zeros(3)                                            # 今、送っている道具の点の速さ（m/秒）
    _, rv, tx, ty, _, _, _ = PB.aim(c, r, list(cur), f["center"], f["longv"], zE)
    off = np.array([tx, ty]) - f["center"][:2]                   # 物の真ん中から道具の点の行き先までの差（指のずれの補正）
    goal_rv = np.array(rv)
    ov = np.zeros(2)                                             # 物の速さ（mm/秒）
    t_meas, t, t_seen, n_meas, n_seen, first = 0.0, 0.0, 0.0, 0, 0, f["center"][:2].copy()
    c_seen = f["center"][:2].copy()
    wrist_done = not WRIST
    hist, w_seen_t, t_wrist, n_wrist, t_wimg = [], 0.0, 0.0, 0, -9.0
    hes = 0.0
    hit_in = False                                               # 指の間で触れた → 閉じる（HIT_IN_MM）
    hit_look = False                                             # 当たった後、手首で測り直す途中か（HIT_LOOK）
    t_hit, n_hit = -1.0, 0                                       # 指が物に当たった時刻と回数（HIT_UP）
    f_base = force_now(r) if (HIT_UP or SAFE_N > 0) else 0.0     # 当たっていない時の力（実機は 0 にならないので差で見る）
    steer, steer_fut, t_steer, steer_n, steer_ms = ("wait" if D1_STEER == 2 else "over"), None, 0.0, {}, []
    steer_from, steer_step = cur[:2].copy(), 0.0
    n_up, was_down = 0, False
    close_fut, t_close, close_n, close_ms, ready = None, 0.0, {}, [], False

    def hes_text():
        """迷っている時間・上がり直しの回数（d1-3B に渡す事実。D1_NOEXC の時だけ）。"""
        if D1_HES:                                               # 待つと何が起きるかを数字で書く（D1_NOEXC は「何秒迷ったか」だけで、d1-3B は 4〜7 秒待ち続けた。10/22 U_n）
            if hes < 0.05:
                return ""
            return hes_fact(hes, t, np.linalg.norm(ov))
        if not D1_NOEXC or (hes < 0.05 and n_up == 0):
            return ""
        return (f" The hand has been near the object without going down for {hes:.1f} s"
                + (f", and has gone back up {n_up} times" if n_up else "") + ".")
    vel_hist = {}                                                # VEL_FIT：カメラごとの直近の測り値 [(時刻, 位置)]
    last_src = {}                                                # カメラごとの最後の値 {"手首"/"三脚": (位置, 時刻)}

    def speed_from(src, new_c):
        """物の速さを出し直す。VEL_SAME なら同じカメラの前の値との差から、でなければ前に見た値（どちらのカメラでも）との差から。"""
        nonlocal ov
        if VEL_FIT > 0:
            hs = vel_hist.setdefault(src if VEL_SAME else "", [])
            hs.append((t, new_c.copy()))
            hs[:] = [h_ for h_ in hs if t - h_[0] <= VEL_FIT]
            if len(hs) >= 3 and hs[-1][0] - hs[0][0] > 0.15:
                T_ = np.array([h_[0] for h_ in hs])
                ov = np.polyfit(T_ - T_.mean(), np.array([h_[1] for h_ in hs]), 1)[0]
                if np.linalg.norm(ov) > OBJ_V_MAX:
                    ov *= OBJ_V_MAX / np.linalg.norm(ov)
            last_src[src] = (new_c.copy(), t)
            return
        prev = last_src.get(src) if VEL_SAME else ((c_seen, t_seen) if n_seen else None)
        if prev is not None and t - prev[1] > 1e-3 and t - prev[1] < 0.5:
            ov = 0.5 * ov + 0.5 * (new_c - prev[0]) / (t - prev[1])
            if np.linalg.norm(ov) > OBJ_V_MAX:
                ov *= OBJ_V_MAX / np.linalg.norm(ov)
        last_src[src] = (new_c.copy(), t)
    while True:
        t0 = c.initPeriod()
        pred = PRED_LOW_S if (PRED_LOW_S > 0 and cur[2] < z_low) else PRED_S   # 指が物のまわりにいる低い所では、隠れても長く先を読む
        guess = c_seen + ov * min(t - t_seen, pred)            # 今の物の位置の見当（最後に見た所 ＋ 速さ × たった時間。長く見えない時は先を読まない）
        if not on_mat(guess):                                    # マットの外へは追わない（速さを捨て、最後に見た所に戻す）
            ov, guess = np.zeros(2), c_seen.copy()
        hist.append((time.time(), list(r.getActualTCPPose())))
        del hist[:-50]
        if WRIST_EVERY > 0 and t >= t_wrist:                     # 手首：見えていれば三脚より先に使う（真下が見えるのは手首だけ）
            t_wrist = t + (WRIST_LOW_EVERY if (WRIST_LOW_EVERY > 0 and cur[2] < z_low) else WRIST_EVERY)   # 低い所では手首で測る間を短くする
            wimg, wtcp, w_seen_t = wrist_live(r, hist, w_seen_t)
            if wimg is not None:
                got = wrist_measure(wimg, wtcp, f, guess, gate=min(GATE_MAX, FOLLOW_MM + GATE_GROW * (t - t_seen)))
                row = {"t": round(t, 2), "z": round(wtcp[2] * 1000 - TABLE_Z, 1), "tcp": list(wtcp), "before": guess.tolist(), "cam": "手首"}
                if got is not None and on_mat(got[0]) and not (np.linalg.norm(got[0] - c_seen) > OBJ_V_MAX * max(t - t_seen, MEAS_EVERY) + 10 and n_seen):
                    t_wimg = t                                   # 手首で物を測れた時刻
                    new_c = got[0]
                    if got[5]:                                   # 一部だけ見えた：横の位置だけ使い、長い向きは見当のまま
                        lv = np.asarray(f["longv"][:2], float) / max(np.linalg.norm(f["longv"][:2]), 1e-9)
                        nrm = np.array([-lv[1], lv[0]])
                        new_c = guess + nrm * float(np.dot(got[0] - guess, nrm))
                        got = (new_c, lv) + tuple(got[2:])
                        row["how"] = "手首で一部だけ（横だけ）"
                    speed_from("手首", new_c)
                    f["center"][:2], f["longv"] = new_c, axis_keep(f, got[1])
                    c_seen, t_seen, guess = new_c.copy(), t, new_c.copy()
                    n_seen += 1
                    n_wrist += 1
                    row["after"] = new_c.tolist()
                    _, rv, tx, ty, _, _, _ = PB.aim(c, r, wtcp, f["center"], f["longv"], zE)
                    off, goal_rv = np.array([tx, ty]) - new_c, np.array(rv)
                steps.append(row)
        if t >= t_meas and t_seen < t:                           # 三脚：手首でこのコマに測れていなければ
            t_meas = t + MEAS_EVERY
            tcp = r.getActualTCPPose()
            row = {"t": round(t, 2), "z": round(tcp[2] * 1000 - TABLE_Z, 1), "tcp": list(tcp), "before": guess.tolist()}
            back = (f["center"].copy(), np.array(f["longv"]).copy())
            ok, img = remeasure(f, tcp, row, log, gate=min(GATE_MAX, FOLLOW_MM + GATE_GROW * (t - t_seen)), guess=guess)
            if ok and n_seen and np.linalg.norm(f["center"][:2] - c_seen) > OBJ_V_MAX * max(t - t_seen, MEAS_EVERY) + 10:
                f["center"], f["longv"] = back                   # 速すぎる飛び方は別の物（腕の陰で取り違えた）とみなして使わない
                row["after"], row["how"], ok = None, "速すぎて外した", False
            n_meas += 1
            n_seen += ok
            if ok:
                new = f["center"][:2].copy()
                if VEL_SAME:
                    speed_from("三脚", new)
                elif n_seen > 1 and t - t_seen > 1e-3:
                    ov = 0.5 * ov + 0.5 * (new - c_seen) / (t - t_seen)      # 物の速さ（なめらかにする）
                    if np.linalg.norm(ov) > OBJ_V_MAX:
                        ov *= OBJ_V_MAX / np.linalg.norm(ov)
                c_seen, t_seen, guess = new, t, new
                _, rv, tx, ty, _, _, _ = PB.aim(c, r, tcp, f["center"], f["longv"], zE)
                off, goal_rv = np.array([tx, ty]) - new, np.array(rv)
            if img is not None and n_meas % 3 == 1:
                cv2.imwrite(os.path.join(rec, f"{f['kind_tag']}_t{int(t * 1000):04d}.jpg"), img)
            steps.append(row)
        # 行き先：物の見当の少し先（LEAD 秒）＋ 指のずれの補正。真上に来るまでは STEPS[0] の高さ、来たら下りる
        gxy = (guess + (ov * LEAD if t - t_seen < pred else 0) + off) / 1000
        if LEAD_CLOSE > 0 and t - t_seen < pred:                 # 指を閉じる向きの動きだけ、写真の遅れと下りる時間の分さらに先回りする（すき間が小さい向き）
            lv_ = np.asarray(f["longv"][:2], float) / max(np.linalg.norm(f["longv"][:2]), 1e-9)
            nrm_ = np.array([-lv_[1], lv_[0]])
            gxy = gxy + nrm_ * float(np.dot(ov, nrm_)) * LEAD_CLOSE / 1000
        exy = np.linalg.norm(gxy - cur[:2])
        fresh = t - t_seen < PRED_S                              # 三脚で今見えているか
        can_look = exy < APPROACH_MM / 1000 and not fresh and not wrist_done and t - t_wimg > WRIST_LIVE_S and cur[2] <= z0 + 2e-3
        look_hit = hit_look and cur[2] >= (TABLE_Z + HIT_LOOK - 2) / 1000   # 当たって上がり終えた → 止まって手首で 1 枚
        if look_hit:
            hit_look = False
            log(f"   当たった後、机から {HIT_LOOK:.0f}mm まで上がった：止まって手首で測り直す")
        if look_hit or ((steer == "look" and can_look) if D1_LOOK else (can_look and (np.linalg.norm(vel) < 0.02 or (HIDE_WRIST and cur[2] < z0 - 3e-3)))):
            steer = "over" if D1_LOOK else steer
            c.servoStop()                                        # 真上で三脚から見えない → 止まって手首のカメラで 1 枚
            wrist_done = True
            got = wrist_fix(r, f, guess, rec, log)
            for _ in range(WRIST_MAX):                           # 動いていれば腕を寄せて撮り直し、2 枚続けて WRIST_STILL mm 以内なら止まったとみなす
                if got is None:
                    break
                _, rv, tx, ty, _, _, _ = PB.aim(c, r, r.getActualTCPPose(), np.r_[got[0], f["h"]], got[1], zE)
                c.moveL([tx / 1000, ty / 1000, r.getActualTCPPose()[2]] + rv, SPEED * 2, ACCEL)
                nxt_ = wrist_fix(r, f, got[0], rec, log)
                if nxt_ is None:
                    break
                moved = np.linalg.norm(nxt_[0] - got[0])
                got = nxt_
                if moved < WRIST_STILL:
                    break
                log(f"   手首のカメラ：まだ動いている（{moved:.1f}mm）ので、寄せて撮り直す")
            cur = np.array(r.getActualTCPPose(), float)
            if got is not None:
                f["center"][:2], f["longv"] = got[0], axis_keep(f, got[1])
                c_seen, t_seen, ov, guess = got[0].copy(), t, np.zeros(2), got[0].copy()
                _, rv, tx, ty, _, _, _ = PB.aim(c, r, r.getActualTCPPose(), f["center"], f["longv"], zE)
                off, goal_rv = np.array([tx, ty]) - got[0], np.array(rv)
                gxy = (guess + off) / 1000
                exy = np.linalg.norm(gxy - cur[:2])
            vel = np.zeros(3)
        if D1_STEER:                                             # d1-3B が方針を選ぶ。答えが来るまでは前の方針のまま動く
            if steer_fut is not None and steer_fut.done():
                try:
                    a_ = steer_fut.result()
                    steer = a_["choice"]
                    steer_ms.append(a_.get("ms", 0))
                except Exception:
                    pass
                steer_fut = None
                steer_n[steer] = steer_n.get(steer, 0) + 1
            if D1_STEER == 2 and steer_fut is None and t >= t_steer:
                t_steer = t + STEER_EVERY
                tgt_ = guess + off
                steer_fut, steer_step = steer_ask_dir(f.get("kind"), cur[2] * 1000 - TABLE_Z, cur[:2] * 1000, tgt_, tgt_ + ov * STEER_PRED,
                                                      np.linalg.norm(ov), t - t_seen, exy < 2 * APPROACH_MM / 1000, (cur[2] - z1) * 1000, hes_text(), fine * 1000)
                steer_from = cur[:2].copy()
            elif D1_STEER == 1 and steer_fut is None and t >= t_steer:
                t_steer = t + STEER_EVERY
                steer_fut, only = steer_ask(f.get("kind"), cur[2] * 1000 - TABLE_Z, exy * 1000, np.linalg.norm(ov), t - t_seen,
                                            exy < 2 * APPROACH_MM / 1000, (cur[2] - z1) * 1000, hes_text(), D1_LOOK and can_look, fine * 1000, hes if D1_HES else 0.0)
                if only:
                    steer = only
            if steer == "lead":
                gxy = (guess + ov * STEER_LEAD + off) / 1000
                exy = np.linalg.norm(gxy - cur[:2])
            elif D1_STEER == 2 and steer in DIRS:                # 選ばれた向きへ 1 歩（聞いた時の位置から）。真上との距離 exy は物の今の真上で測る
                gxy = steer_from + steer_step / 1000 * np.array(DIRS[steer])
            elif D1_STEER == 2 and steer == "wait":
                gxy = steer_from.copy()
            near_ok = (steer == "descend" and exy < 2 * APPROACH_MM / 1000) or (not D1_NOEXC and (cur[2] < z1 + COMMIT_MM / 1000 or (not D1_HES and exy < APPROACH_MM / 1000 and hes > HESITATE_S)))
        else:
            near_ok = exy < APPROACH_MM / 1000 and (fresh or wrist_done or cur[2] < z1 + COMMIT_MM / 1000 or hes > HESITATE_S)
        if exy < 2 * APPROACH_MM / 1000 and not near_ok:
            hes += SERVO_DT                                      # 物の近くで下りられずにいる時間（迷っている時間）
        if was_down and not near_ok:
            n_up += 1                                            # 下りていたのに上がり直した
        was_down = near_ok
        gz = z1 if near_ok else z0
        if HOLD_LOW and not near_ok and exy < 2 * APPROACH_MM / 1000:
            gz = min(z0, max(cur[2], z_clear if z_clear is not None else z1))   # 近くで下りられない時は今の高さで待つ（上がり直さない）
        # 真上で、三脚で見えているか手首で直した後だけ下りる（離れたら STEPS[0] の高さへ上がる）
        if z_clear is not None and gz < z_clear and not (exy < fine and (fresh or wrist_done)):
            gz = z_clear                                         # 指が物の横に当たらないよう、そろうまでは物の上 CLEAR_MM で追う（安全のため、聞く前に外す）
        if SAFE_N > 0 and not HIT_UP and cur[2] < z_low and t - t_hit > HIT_HOLD_S:   # 安全：強く押し付けたら上がる（HIT_UP の時は下の当たり判定が受け持つ）
            fs = force_now(r) - f_base
            if fs > SAFE_N:
                t_hit, n_hit = t, n_hit + 1
                log(f"   安全：指先に {fs:.1f}N かかった（指先は物の上の面から {(cur[2] * 1000 - TABLE_Z - f['h']):+.1f}mm）ので、閉じずに上がる")
        if HIT_UP and cur[2] < z_low:                            # 指が物に当たった（力が増えた・下りる先まで下りられない）→ 閉じずに少し上がって合わせ直す
            fz = force_now(r) - f_base
            blocked = r.getActualTCPPose()[2] - cur[2] > HIT_LAG_MM / 1000
            depth = TABLE_Z + f["h"] - cur[2] * 1000             # 指先が物の上の面より何 mm 下か
            if (fz > HIT_N or blocked) and HIT_IN_MM > 0 and depth > HIT_IN_MM and not hit_in:
                hit_in = True
                log(f"   指が物に触れた（力 {fz:.1f}N、指先は物の上の面から {-depth:+.1f}mm で物は指の間）：上がらずに閉じる")
            elif (fz > HIT_N or blocked) and t - t_hit > HIT_HOLD_S and not hit_in:
                t_hit, n_hit = t, n_hit + 1
                hit_look = HIT_LOOK > 0
                what = ""
                if hasattr(r, "hand_force"):                     # シミュレーターだけ：何に当たったか
                    what = "、" + "、".join(f"{b_} {fn:.0f}N" for fn, a_, b_ in r.hand_force(detail=True)[1][:2])
                log(f"   指が物に当たった（力 {fz:.1f}N" + ("・下りられない" if blocked else "") + what + f"、指先は物の上の面から {(cur[2] * 1000 - TABLE_Z - f['h']):+.1f}mm）：閉じずに少し上がって合わせ直す")
        if hit_look:                                             # 測り直す：手首から見える高さまで上がる（物の上 5mm では見えない）
            gz = max(gz, (TABLE_Z + HIT_LOOK) / 1000)
        if (HIT_UP or SAFE_N > 0) and t - t_hit < HIT_HOLD_S:    # 当たった：物の動く向きへ寄せながら上がる（真上に上がるだけでは、また同じ所に当たる。10/22）
            gz = max(gz, z_clear if z_clear is not None else z1 + 0.01)
            if np.linalg.norm(ov) > 2.0:
                gxy = (guess + ov * HIT_LEAD_S + off) / 1000     # 物の先回り（動く向き）
            else:
                gxy = (guess + off) / 1000                       # 止まっている物：物の真上へ
        want = np.zeros(3)
        want[:2] = (gxy - cur[:2]) * K_XY + (ov / 1000 if t - t_seen < pred else 0)   # 寄せる速さ ＋ 物の速さ（見えている間だけ。低い所では pred 秒まで）
        sp = np.linalg.norm(want[:2])
        if sp > XY_V:
            want[:2] *= XY_V / sp
        dmax = DESC_V if cur[2] <= z0 + 1e-3 else XY_V
        if FAST_LAST > 0 and z_clear is not None and cur[2] <= z_clear + 1e-3 and gz < cur[2]:
            dmax = FAST_LAST                                     # 物の上の面より下は一気に下りる（ずれる時間を短く）
        want[2] = float(np.clip((gz - cur[2]) * K_XY, -dmax, XY_V))   # STEPS[0] より下は DESC_V でゆっくり下りる
        dv = want - vel                                          # 加速度の上限
        if np.linalg.norm(dv) > A_MAX * SERVO_DT:
            dv *= A_MAX * SERVO_DT / np.linalg.norm(dv)
        vel = vel + dv
        nxt = cur.copy()
        nxt[:3] = cur[:3] + vel * SERVO_DT
        Rc, Rg = PB.rotvec_to_R(cur[3:]), PB.rotvec_to_R(goal_rv)
        w = np.array(PB.R_to_rotvec(Rg @ Rc.T))
        ang = np.linalg.norm(w)
        if ang > ROT_V * SERVO_DT:
            w *= ROT_V * SERVO_DT / ang
        nxt[3:] = PB.R_to_rotvec(PB.rotvec_to_R(w) @ Rc)
        c.servoL(list(nxt), 0.0, 0.0, SERVO_DT, 0.1, 300)
        c.waitPeriod(t0)
        cur, t = nxt, t + SERVO_DT
        ready = abs(cur[2] - z1) < 0.5e-3 and exy < 1.5e-3 and ang < math.radians(2)
        if D1_CLOSE:                                             # 閉じるかどうかは d1-3B（答えが来るまでは追い続ける）
            done = False
            if close_fut is not None and close_fut.done():
                try:
                    a_ = close_fut.result()
                    close_n[a_["choice"]] = close_n.get(a_["choice"], 0) + 1
                    close_ms.append(a_.get("ms", 0))
                    done = ready and a_["choice"] == "close"
                    if a_["choice"] != "close":
                        t_close = t + STEER_EVERY
                except Exception:
                    pass
                close_fut = None
            if ready and not done and close_fut is None and t >= t_close:
                close_fut, only = close_ask(f.get("kind"), np.linalg.norm(ov) if (t - t_seen < pred or CLOSE_TEXT >= 6) else 0.0, float(f["size"][0]), t - t_seen, hes_text(), exy * 1000)
                done = only == "close"
        else:
            done = ready and (np.linalg.norm(ov) < 5 or t - t_seen >= PRED_S)
        if hit_in:
            done = True
        elif done and HIT_UP and t - t_hit < HIT_HOLD_S:
            done = False                                         # 当たっている間は閉じない
        if done or t > FOLLOW_MAX_S:
            break
    f.pop("closed_w", None)
    if done and CLOSE_MOVING and force is not None:              # 止まらずに、物の速さで追い続けながら閉じる
        f["closed_w"] = close_moving(c, r, cur, vel, ov if t - t_seen < pred else np.zeros(2), list(goal_rv), force, log, float(f["size"][0]))
    c.servoStop()
    if D1_CLOSE and (close_n or ready):
        log("   d1-3B（閉じるか）：" + ("・".join(f"{'閉じる' if k_ == 'close' else 'もう少し追う'} {v_} 回" for k_, v_ in close_n.items()) or "止まっていたので閉じる")
            + (f"、1 問 平均 {np.mean(close_ms):.0f}ms" if close_ms else "") + ("" if done else "（時間切れ）"))
    f["center"][:2] = guess                                      # つかむ所は、最後の見当
    d = guess - first
    if KEEP_AXIS and f.get("axis_kept"):
        log(f"   長い向きが大きく変わった測り値を使わなかった（向きは前のまま）：{f.pop('axis_kept')} 回")
    if HIT_UP and n_hit:
        log(f"   指が物に当たって上がり直した：{n_hit} 回")
    log(f"   止まらずに寄せて下りながら三脚で {n_meas} 回測り直し（三脚と手首で見えた {n_seen} 回、うち手首 {n_wrist} 回）、{t:.1f} 秒（迷った {hes:.1f} 秒）→ 動いた分 dx {d[0]:+.1f} dy {d[1]:+.1f}mm、"
        f"物の速さ {np.linalg.norm(ov):.0f}mm/秒" + ("（時間切れ）" if not done else ""))
    if D1_STEER:
        nm = {"over": "真上へ", "lead": "先回り", "descend": "下りる", "look": "止まって手首で撮る", "wait": "待つ", "right": "右", "front-right": "右前", "front": "前",
              "front-left": "左前", "left": "左", "back-left": "左後ろ", "back": "後ろ", "back-right": "右後ろ"}
        log("   d1-3B（追いかけの方針）：" + "・".join(f"{nm.get(k_, k_)} {v_} 回" for k_, v_ in steer_n.items())
            + (f"、1 問 平均 {np.mean(steer_ms):.0f}ms" if steer_ms else ""))
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", required=True)
    ap.add_argument("--say", default="全部の物を箱に入れて")
    ap.add_argument("--go", action="store_true", help="腕とハンドを動かす（無ければ見つけて決めるだけ）")
    a = ap.parse_args()
    rec = os.path.join(HERE, "logs_track_real", time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(rec, exist_ok=True)
    from rtde_receive import RTDEReceiveInterface
    r = RTDEReceiveInterface(a.ip)
    rot = list(r.getActualTCPPose()[3:])
    to_box = PL.instruction(a.say, kinds=tuple(H_NOM))
    carry = [k for k in H_NOM if to_box.get(k)]
    print(f"運ぶ物：{'・'.join(carry) or 'なし'}", flush=True)
    box = TB.load()
    img = TV.grab(0)
    if img is None:
        raise SystemExit("三脚の写真が来ない（cam_view.py --cam2 が動いているか）")
    TV.check_still(img)                                          # 三脚が校正した時から動いていたら止める（10/22）
    if not a.go:
        objs = look(r, rec, "look")
        print("--go が無いので、ここまで（腕もハンドも動かしていません）。見つけた物：" + "、".join(f"{o['kind'] or 'それ以外'} x {o['xy'][0]:.0f} y {o['xy'][1]:.0f}" for o in objs))
        return
    from hand import HandClient
    from rtde_control import RTDEControlInterface
    h = HandClient(log=lambda s: print(s, flush=True))
    c = RTDEControlInterface(a.ip)
    try:
        results = run(r, c, h, carry, box, rot, rec)
    finally:
        c.stopScript()
        c.disconnect()
    print("==== 結果：" + "、".join(f"{k} {v}" for k, v in results.items()) + f"（記録 {rec}）", flush=True)


def run(r, c, h, carry, box, rot, rec, log=print):
    """carry の種類を順に、三脚で見つけ→下りながら測り直し→つかんで箱（box、無ければ元の所）へ。種類ごとの結果を返す。
    r・c・h は実機（ur_rtde・HandClient）でも、シミュレーター（robot/sim_rtde の SimRobot・SimHand）でもよい。"""
    steps, results = [], {}
    h.check("開く")
    h.open()
    look_pose = [LOOK[0] / 1000, LOOK[1] / 1000, (TABLE_Z + LOOK[2]) / 1000] + rot
    try:
        for i, k in enumerate(carry):
            res = None
            for n_try in range(1 + RETRY):
                objs = []
                if (i or n_try) and LOOK_FROM_BOX:                # 箱に落とした所（やり直しは物の真上）から、そのまま三脚で探す（LOOK へ戻らない）
                    objs = [o for o in look(r, rec, f"look{i + 1}_{n_try}_here", log) if o["kind"] == k and not in_box(o["xy"], box)]
                    if not objs:
                        log(f"   今の所からは {k} が見えないので、見る位置へ戻る")
                if not objs:
                    c.moveL(look_pose, SPEED, ACCEL)
                    objs = [o for o in look(r, rec, f"look{i + 1}_{n_try}", log) if o["kind"] == k and not in_box(o["xy"], box)]
                if not objs and NEAR_LOOK:
                    o_ = near_look(c, r, look(r, rec, f"look{i + 1}_{n_try}_near", log), k, rot, box, rec, log)
                    objs = [o_] if o_ else []
                if MAX_R > 0 and objs:
                    for o_ in objs:
                        if np.hypot(*o_["xy"][:2]) > MAX_R:
                            log(f"   {k}（x {o_['xy'][0]:.0f} y {o_['xy'][1]:.0f}）は腕の根元から {np.hypot(*o_['xy'][:2]):.0f}mm と遠すぎて、届かないので取りに行かない")
                    objs = [o_ for o_ in objs if np.hypot(*o_["xy"][:2]) <= MAX_R]
                    if not objs:
                        res = "届かない"
                        break
                if MIN_R > 0 and objs:
                    far = [o for o in objs if np.hypot(*o["xy"][:2]) >= MIN_R]
                    for o_ in objs:
                        if np.hypot(*o_["xy"][:2]) < MIN_R:
                            log(f"   {k}（x {o_['xy'][0]:.0f} y {o_['xy'][1]:.0f}）は腕の根元から {np.hypot(*o_['xy'][:2]):.0f}mm と近すぎて、腕どうしが当たるので取りに行かない")
                    objs = far
                    if not objs:
                        res = "近すぎる"
                        break
                if not objs:
                    log(f"{k} が三脚で見つからない")
                    res = "見つからない"
                    break
                o = max(objs, key=lambda o: max(o["kind_p"].values()))
                f = to_f(o)
                f["kind_tag"] = f"obj{i + 1}_{n_try}"
                log(f"→ {k}：三脚で x {o['xy'][0]:.1f} y {o['xy'][1]:.1f}、長さ {o['size'][1]:.0f}・幅 {o['size'][0]:.0f}mm" + (f"（やり直し {n_try} 回目）" if n_try else ""))
                if SERVO and TO_GRIP:                            # 先につかむ高さを d1-3B に決めてもらい、その高さまで追いながら下りる（閉じる直前まで追う）
                    grip = PL.grip(f, 0.0, False)
                    if not follow_servo(c, r, f, rec, steps, log, z_end=grip, force=PA.FORCE.get(k, 60)) and TIMEOUT_UP:
                        cur_ = r.getActualTCPPose()                  # 時間切れでそろっていない → 見当の所へ下ろすと物に当たるので、上がって三脚で見直す
                        c.moveL([cur_[0], cur_[1], (TABLE_Z + STEPS[0]) / 1000] + list(cur_[3:]), PB.SPEED, PB.ACCEL)
                        log(f"   {k}：時間切れでそろわなかったので、下ろさずに上がって見直す")
                        res = "失敗"
                        continue
                elif SERVO:
                    follow_servo(c, r, f, rec, steps, log)
                    grip = PL.grip(f, 0.0, False)
                else:
                    follow(c, r, f, rot, rec, steps, log)
                    grip = PL.grip(f, 0.0, False)
                place = TB.spot(box, i, len(carry)) if box else [float(f["center"][0]), float(f["center"][1])]
                kw = {"place_far": PB.BOX_FAR, "place_h": TB.release_h(box, grip)} if box else {}
                ok = PB.grasp(c, r, r.getActualTCPPose(), f["center"], f["longv"], grip, PA.FORCE.get(k, 60), place, False, miss="retry",
                              from_here=True, closed_w=f.pop("closed_w", None), **kw)
                res = "成功" if ok is not False else "失敗"
                if ok is not False or (hasattr(c, "isProgramRunning") and not c.isProgramRunning()):
                    break
                log(f"   {k} を持てなかったので、三脚で見直してやり直す")
            results[k] = res
            if hasattr(c, "isProgramRunning") and not c.isProgramRunning():
                log("腕のプログラムが止まった（保護停止など）ので終わる")
                break
        c.moveL(look_pose, SPEED, ACCEL)
    finally:
        json.dump({"carry": carry, "results": results, "steps": steps}, open(os.path.join(rec, "steps.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return results


if __name__ == "__main__":
    main()

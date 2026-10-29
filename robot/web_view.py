"""シミュレーターの様子を、窓（mujoco.viewer や cv2 の窓）ではなく、Chrome の 1 枚のページで見せる（10/19）。

  ・3D の動き：画面の外で描いた絵（mujoco.Renderer）を、動画（MJPEG）として送る。絵の上をドラッグで回す・ホイールで寄る
  ・写真：AI やカメラが見た画像を、名前ごとに並べる（image(名前, BGR の画像)）
  ・文字：print の出力を、そのままページの下に流す
  mujoco.viewer の代わりに使えるよう、sync()・is_running()・close()・cam を持つ。
  Chrome には、いつもの Chrome の普通のタブとして開く（ガントチャートと同じ）。
"""
import io
import json
import os
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import cv2
import mujoco
import numpy as np

SEEN_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "web_view_seen.txt")
SEEN_KEEP = 12 * 3600.0                                  # この秒数以内にページが読みに来ていれば、新しいタブを開かない
# （10/20：90 秒では足りずタブが増えた。Chrome は裏のタブを 5 分ほどで凍らせ、見張りの Worker も止まる。凍ったタブは、見た時に動き出して自分でつなぎ直すので、開き直さない）
BOOT = str(time.time())                                  # 立ち上げごとに変わる印。ページが次の回に気づいてつなぎ直す
PORT = int(os.environ.get("SIM_WEB_PORT", "8765"))
W3D, H3D = 1280, 720          # 3D の絵の大きさ
FPS = 20                      # 3D の絵を描く回数（1 秒あたり）。物理の速さには関係しない
TITLE = "シミュレーター"
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"

PAGE = """<!doctype html><html lang="ja"><head><meta charset="utf-8"><title>%TITLE%</title>
<style>
:root{--bg:#f4f5f7;--card:#fff;--fg:#1d2330;--sub:#5b6474;--line:#d9dde4}
@media (prefers-color-scheme:dark){:root{--bg:#15181e;--card:#1f232b;--fg:#e6e9ef;--sub:#9aa3b2;--line:#323844}}
html,body{margin:0;height:100%;background:var(--bg);color:var(--fg);font-family:"Yu Gothic UI","Meiryo",sans-serif}
#wrap{display:grid;grid-template-columns:minmax(0,3fr) minmax(0,1.3fr);grid-template-rows:minmax(0,1fr) 16vh;gap:10px;height:100vh;padding:10px;box-sizing:border-box}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;overflow:hidden;display:flex;flex-direction:column;min-height:0}
.card h2{font-size:14px;margin:0;padding:6px 10px;border-bottom:1px solid var(--line);color:var(--sub);font-weight:600}
#v3d{flex:1;min-height:0;display:flex;align-items:center;justify-content:center;background:#000}
#v3d img{max-width:100%;max-height:100%;cursor:grab;user-select:none}
#imgs{flex:1;overflow:auto;padding:8px;display:flex;flex-direction:column;gap:8px}
#imgs figure{margin:0}#imgs img{width:100%;max-height:calc((84vh - 110px) / 2);object-fit:contain;border-radius:4px;display:block}
#imgs figcaption{font-size:12px;color:var(--sub);margin:2px 0}
#log{grid-column:1 / span 2}
#logtxt{flex:1;overflow:auto;margin:0;padding:8px 10px;font:13px/1.45 Consolas,"Yu Gothic UI",monospace;white-space:pre-wrap}
#state{font-weight:400;margin-left:8px}
</style></head><body><div id="wrap">
<div class="card"><h2>3D の動き<span id="state"></span>（ドラッグで回す・ホイールで寄る）</h2><div id="v3d"><img id="s3d" draggable="false"></div></div>
<div class="card"><h2>カメラ・AI が見た画像</h2><div id="imgs"></div></div>
<div class="card" id="log"><h2>進み具合</h2><pre id="logtxt"></pre></div>
</div><script>
let n=0,ver={},boot=null,down=false;const L=document.getElementById('logtxt'),I=document.getElementById('imgs'),st=document.getElementById('state');
let fno=-1,floop=0;async function live(){const my=++floop;fno=-1;const s3=document.getElementById('s3d');while(my===floop){try{const r=await fetch('/frame?after='+fno,{cache:'no-store'});if(r.status===200){fno=+r.headers.get('X-No');const u=URL.createObjectURL(await r.blob());const o=s3.src;s3.src=u;if(o.startsWith('blob:'))URL.revokeObjectURL(o);}}catch(e){await new Promise(z=>setTimeout(z,500));}}}live();
let gen=0;async function poll(my){if(my===undefined)my=gen;if(my!==gen)return;try{const r=await fetch('/log?from='+n);const j=await r.json();
 if(j.boot!==boot||down){if(boot!==null&&j.boot!==boot){location.reload();return;n=0;ver={};I.innerHTML='';L.textContent+='\\n──── 次の回 ────\\n';document.title=j.title;
   document.querySelector('h1')&&(document.querySelector('h1').textContent=j.title);setTimeout(()=>poll(my),0);boot=j.boot;down=false;live();return;}
  if(down)live();boot=j.boot;down=false;}
 if(j.lines.length){const atEnd=L.scrollTop+L.clientHeight>=L.scrollHeight-20;L.textContent+=j.lines.join('');n=j.next;if(atEnd)L.scrollTop=L.scrollHeight;}
 for(const [k,v] of Object.entries(j.images)){if(ver[k]===v)continue;ver[k]=v;let f=document.getElementById('im_'+k);
  if(!f){f=document.createElement('figure');f.id='im_'+k;f.innerHTML='<figcaption></figcaption><img>';f.firstChild.textContent=k;I.appendChild(f);}
  f.lastChild.src='/img?name='+encodeURIComponent(k)+'&v='+v;}
 st.textContent=j.running?'':'（終わりました）';}catch(e){st.textContent='（シミュレーターが止まっています。次の回が始まると自動でつながります）';down=true;}setTimeout(()=>poll(my),100);}
poll();
try{const wk=new Worker(URL.createObjectURL(new Blob([`let up=null;async function t(){try{const r=await fetch('/ping',{cache:'no-store'});const j=await r.json();
 if(up!==j.boot){up=j.boot;postMessage(j.boot);}}catch(e){if(up!==null){up=null;postMessage(null);}}setTimeout(t,500);}t();`],{type:'text/javascript'})));
 wk.onmessage=e=>{if(e.data===null){down=true;}else if(e.data!==boot){gen++;poll(gen);}};}catch(e){}
const s=document.getElementById('s3d');let drag=null;
s.addEventListener('error',()=>{down=true;});                     // 映像が切れたら、記録が読めた時につなぎ直す（次の回の立ち上げを待つ）
s.addEventListener('mousedown',e=>{drag=[e.clientX,e.clientY];s.style.cursor='grabbing';});
addEventListener('mouseup',()=>{drag=null;s.style.cursor='grab';});
addEventListener('mousemove',e=>{if(!drag)return;const dx=e.clientX-drag[0],dy=e.clientY-drag[1];drag=[e.clientX,e.clientY];fetch('/cam?daz='+(-dx*0.3)+'&del='+(-dy*0.3));});
s.addEventListener('wheel',e=>{e.preventDefault();fetch('/cam?zoom='+(e.deltaY>0?1.1:1/1.1));},{passive:false});
</script></body></html>"""


class _Tee(io.TextIOBase):
    """print の出力を、元の画面とページの両方へ出す。"""

    def __init__(self, orig, sink):
        self.orig, self.sink = orig, sink

    def write(self, s):
        self.orig.write(s)
        if s:
            self.sink(s)
        return len(s)

    def flush(self):
        self.orig.flush()


_ONE = None


def WebView(m, d, title=TITLE):
    """ページは 1 つだけ。2 回目からは（--batch で場面を続ける時など）同じページに、新しい場面をつなぐ。"""
    global _ONE
    if _ONE is None:
        _ONE = _View(m, d, title)
    else:
        _ONE.attach(m, d)
    return _ONE


class _View:
    def __init__(self, m, d, title=TITLE, open_browser=True):
        self.m, self.d = m, d
        self.cam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(m, self.cam)
        m.vis.global_.offwidth = max(m.vis.global_.offwidth, W3D)
        m.vis.global_.offheight = max(m.vis.global_.offheight, H3D)
        self.rr = mujoco.Renderer(m, H3D, W3D)
        self.lock = threading.Lock()
        self.jpg, self.jpg_no = None, 0
        self.new_jpg = threading.Condition(self.lock)
        self.images, self.img_ver = {}, {}
        self.lines, self.partial = [], ""
        self.running, self.last = True, 0.0
        self.title = title
        sys.stdout = _Tee(sys.stdout, self._log)
        self.seen = False                                         # ページが一度でも読みに来たか
        self.seen_t = 0.0
        self.srv = ThreadingHTTPServer(("127.0.0.1", PORT), self._handler())
        self.srv.daemon_threads = True
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.sync(force=True)
        if open_browser:                                          # 前の回のページが開いていれば、それが自分でつなぎ直すので新しく開かない（10/20）
            threading.Thread(target=self._open_if_none, daemon=True).start()

    def attach(self, m, d):
        """新しい場面の模型につなぎ直す。"""
        self.m, self.d = m, d
        m.vis.global_.offwidth = max(m.vis.global_.offwidth, W3D)
        m.vis.global_.offheight = max(m.vis.global_.offheight, H3D)
        self.rr = mujoco.Renderer(m, H3D, W3D)
        with self.lock:
            self.running = True
        self.sync(force=True)

    # ---- mujoco.viewer と同じ使い方
    def is_running(self):
        return self.running

    def sync(self, force=False):
        """3D の絵を描き直す（FPS 回/秒まで。間に合わない分は飛ばす）。"""
        now = time.perf_counter()
        if not force and now - self.last < 1.0 / FPS:
            return
        self.last = now
        self.rr.update_scene(self.d, camera=self.cam)
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(self.rr.render(), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 80])
        with self.lock:
            self.jpg, self.jpg_no = buf.tobytes(), self.jpg_no + 1
            self.new_jpg.notify_all()

    def close(self):
        """ページは開いたまま残す（最後の様子と文字が見えるように）。プログラムが終わると、ページは「止まっています」になる。"""
        self.sync(force=True)
        with self.lock:
            self.running = False

    # ---- 画像と文字
    def image(self, name, bgr, width=960):
        """名前 name の画像を、ページの右に出す（同じ名前なら差し替える）。"""
        h, w = bgr.shape[:2]
        if w > width:
            bgr = cv2.resize(bgr, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        with self.lock:
            self.images[name] = buf.tobytes()
            self.img_ver[name] = self.img_ver.get(name, 0) + 1

    def _log(self, s):
        with self.lock:
            self.partial += s
            if "\n" in self.partial:
                *done, self.partial = self.partial.split("\n")
                self.lines.extend(x + "\n" for x in done)

    # ---- Chrome
    def _open_if_none(self, wait=2.5):
        # 開いているページは /log を読みに来る。ただし裏のタブは Chrome が時計を遅らせる（最大 1 分に 1 回）ので、
        # 最後に読みに来た時刻を SEEN_FILE に残し、SEEN_KEEP 秒以内なら、どこかにページが開いているとみなす（10/20：回ごとにタブが増えた）
        time.sleep(wait)
        try:
            recent = time.time() - os.path.getmtime(SEEN_FILE) < SEEN_KEEP
        except OSError:
            recent = False
        if not self.seen and not recent:
            self._open_chrome()

    def mark_seen(self):
        """ページが来た時刻を SEEN_FILE に残す（2 秒に 1 回まで）。"""
        if time.time() - self.seen_t > 2.0:
            self.seen_t = time.time()
            try:
                with open(SEEN_FILE, "w") as fh:
                    fh.write(str(self.seen_t))
            except OSError:
                pass

    def _open_chrome(self):
        """いつもの Chrome に、普通のタブとして開く（ガントチャートの HTML と同じ開き方。10/19）。"""
        url = f"http://127.0.0.1:{PORT}/"
        try:                                                     # 開いた時刻を残す（タブが増えていないかを後で確かめる）
            with open(os.path.join(os.path.dirname(SEEN_FILE), "web_view_opened.txt"), "a", encoding="utf-8") as fh:
                fh.write(time.strftime("%H:%M:%S") + "\n")
        except OSError:
            pass
        exe = CHROME if os.path.exists(CHROME) else "chrome"
        try:
            subprocess.Popen([exe, url])
        except OSError:
            import webbrowser
            webbrowser.open(url)

    def _handler(self):
        view = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def handle(self):
                try:                                             # ページが読み込み直すと、送っている途中のつながりが切れる（害は無いので黙る）
                    super().handle()
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    pass

            def _send(self, body, ctype):
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                u = urlparse(self.path)
                q = parse_qs(u.query)
                if u.path == "/":
                    self._send(PAGE.replace("%TITLE%", view.title).encode("utf-8"), "text/html; charset=utf-8")
                elif u.path == "/ping":                                 # 裏のタブの見張り（Worker）が 0.5 秒ごとに来る
                    view.seen = True
                    view.mark_seen()
                    self._send(json.dumps({"boot": BOOT}).encode("utf-8"), "application/json")
                elif u.path == "/log":
                    view.seen = True
                    view.mark_seen()
                    i = int(q.get("from", ["0"])[0])
                    with view.lock:
                        body = {"lines": view.lines[i:], "next": len(view.lines), "images": dict(view.img_ver),
                                "running": view.running, "boot": BOOT, "title": view.title}
                    self._send(json.dumps(body, ensure_ascii=False).encode("utf-8"), "application/json")
                elif u.path == "/img":
                    with view.lock:
                        b = view.images.get(q.get("name", [""])[0])
                    if b is None:
                        self.send_error(404)
                    else:
                        self._send(b, "image/jpeg")
                elif u.path == "/cam":
                    c = view.cam
                    c.azimuth += float(q.get("daz", ["0"])[0])
                    c.elevation = float(np.clip(c.elevation + float(q.get("del", ["0"])[0]), -89, 89))
                    c.distance *= float(q.get("zoom", ["1"])[0])
                    self._send(b"ok", "text/plain")
                elif u.path == "/frame":                                # 映像は 1 枚ずつ取りに来る（10/20：/stream の長いつながりが止まると、読み込み直すまで映らなかった）
                    after = int(q.get("after", ["-1"])[0])
                    with view.lock:
                        view.new_jpg.wait_for(lambda: view.jpg_no != after, timeout=2.0)
                        b, no = view.jpg, view.jpg_no
                    if b is None:
                        self.send_response(204)
                        self.end_headers()
                    else:
                        self.send_response(200)
                        self.send_header("Content-Type", "image/jpeg")
                        self.send_header("Cache-Control", "no-store")
                        self.send_header("X-No", str(no))
                        self.send_header("Content-Length", str(len(b)))
                        self.end_headers()
                        self.wfile.write(b)
                elif u.path == "/stream":
                    self.send_response(200)
                    self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=f")
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    seen = -1
                    try:
                        while True:
                            with view.lock:
                                view.new_jpg.wait_for(lambda: view.jpg_no != seen, timeout=1.0)
                                b, seen = view.jpg, view.jpg_no
                            if b is None:
                                continue
                            self.wfile.write(b"--f\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                             + str(len(b)).encode() + b"\r\n\r\n" + b + b"\r\n")
                    except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                        pass
                else:
                    self.send_error(404)
        return H

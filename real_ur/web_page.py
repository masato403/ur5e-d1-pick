"""実機の様子を、Chrome の 1 枚のページ（普通のタブ）で見せる（10/19。シミュレーターの robot/web_view.py と同じ見た目）。

  ・ページを出す係（Page）：cam_view.py が持つ。左にカメラの生の映像、右に撮った写真、下に進み具合の文字
  ・送る係（push_log / push_image / tee_stdout）：pick_pass.py などが、写真と print の文字をページへ送る
    送るのは裏の糸（スレッド）がするので、ロボットの動きは待たせない。ページが無ければ黙って捨てる
使い方:  ページ  http://127.0.0.1:8766/
"""
import io
import json
import os
import queue
import subprocess
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, quote, urlparse

import cv2

PORT = int(os.environ.get("REAL_WEB_PORT", "8766"))
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"

PAGE = """<!doctype html><html lang="ja"><head><meta charset="utf-8"><title>%TITLE%</title>
<style>
:root{--bg:#f4f5f7;--card:#fff;--fg:#1d2330;--sub:#5b6474;--line:#d9dde4;--btn:#e8ebf0}
@media (prefers-color-scheme:dark){:root{--bg:#15181e;--card:#1f232b;--fg:#e6e9ef;--sub:#9aa3b2;--line:#323844;--btn:#2b313b}}
html,body{margin:0;height:100%;background:var(--bg);color:var(--fg);font-family:"Yu Gothic UI","Meiryo",sans-serif}
#wrap{display:grid;grid-template-columns:minmax(0,3fr) minmax(0,1.3fr);grid-template-rows:minmax(0,1fr) 30vh;gap:10px;height:100vh;padding:10px;box-sizing:border-box}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;overflow:hidden;display:flex;flex-direction:column;min-height:0}
.card h2{font-size:14px;margin:0;padding:6px 10px;border-bottom:1px solid var(--line);color:var(--sub);font-weight:600;display:flex;gap:8px;align-items:center}
.card h2 button{font:inherit;font-size:12px;padding:2px 8px;border:1px solid var(--line);border-radius:4px;background:var(--btn);color:var(--fg);cursor:pointer}
#live{flex:1;min-height:0;display:flex;align-items:center;justify-content:center;background:#000}
#live img{max-width:100%;max-height:100%;min-width:0}
#live.two{gap:4px}#live.two figure{flex:1;min-width:0;height:100%;margin:0;display:flex;flex-direction:column}
#live figure figcaption{color:#ccc;font-size:12px;padding:2px 4px}#live figure div{flex:1;min-height:0;display:flex;align-items:center;justify-content:center}
#imgs{flex:1;overflow:auto;padding:8px;display:flex;flex-direction:column;gap:8px}
#imgs figure{margin:0}#imgs img{width:100%;border-radius:4px;display:block}
#imgs figcaption{font-size:12px;color:var(--sub);margin:2px 0}
#log{grid-column:1 / span 2}
#logtxt{flex:1;overflow:auto;margin:0;padding:8px 10px;font:13px/1.45 Consolas,"Yu Gothic UI",monospace;white-space:pre-wrap}
#state{font-weight:400}
@media (max-width:800px){#wrap{grid-template-columns:1fr;grid-template-rows:50vh auto 40vh;height:auto}#log{grid-column:1}}
</style></head><body><div id="wrap">
<div class="card"><h2>カメラ（今の映像）<span id="state"></span><span style="flex:1"></span><button onclick="cmd('u')">曲がりを直す 切り替え</button><button onclick="cmd('s')">今の画像を保存</button></h2><div id="live"><figure><figcaption>手首</figcaption><div><img id="cam"></div></figure><figure id="f2" style="display:none"><figcaption>三脚</figcaption><div><img id="cam2"></div></figure></div></div>
<div class="card"><h2>撮った写真・見つけた物</h2><div id="imgs"></div></div>
<div class="card" id="log"><h2>進み具合</h2><pre id="logtxt"></pre></div>
</div><script>
let n=0,ver={},down=false;const L=document.getElementById('logtxt'),I=document.getElementById('imgs'),st=document.getElementById('state');
function cmd(k){fetch('/cmd?k='+k);}
// 映像を 1 本つなぎ直す（カメラの係を立ち上げ直すと前のつながりが切れて、映らないままになった。10/20）
// 1 枚ずつ取りに行く（multipart の流れは、裏のタブや別の窓で映らないことがあった。シミュレーターのページと同じ。10/20）
let fno=0,fno2=0,busy=false;
async function live(){if(busy)return;busy=true;const im=document.getElementById('cam');
 while(true){try{const r=await fetch('/frame?after='+fno,{cache:'no-store'});
  if(r.status===200){fno=+r.headers.get('X-No')||fno;const u=URL.createObjectURL(await r.blob());const old=im.src;im.src=u;if(old.startsWith('blob:'))URL.revokeObjectURL(old);}
  else await new Promise(z=>setTimeout(z,200));}catch(e){await new Promise(z=>setTimeout(z,1000));}}}
live();
// 三脚のカメラ（cam_view --cam2 の時だけ）。手首と同じページに並べる（10/20、1 つの画面で見られるように）
async function live2(){const im=document.getElementById('cam2'),f=document.getElementById('f2'),L_=document.getElementById('live');
 while(true){try{const r=await fetch('/frame?w=2&after='+fno2,{cache:'no-store'});
  if(r.status===200){fno2=+r.headers.get('X-No')||fno2;f.style.display='';L_.classList.add('two');const u=URL.createObjectURL(await r.blob());const old=im.src;im.src=u;if(old.startsWith('blob:'))URL.revokeObjectURL(old);}
  else await new Promise(z=>setTimeout(z,500));}catch(e){await new Promise(z=>setTimeout(z,1000));}}}
live2();
async function poll(){try{const r=await fetch('/log?from='+n);const j=await r.json();
 // 係が止まって戻った・立ち上げ直した（数が前より少ない）時は、映像をつなぎ直し、文字と写真を初めから読み直す
 if(down||j.next<n){down=false;fno=0;fno2=0;if(j.next<n){n=0;ver={};L.textContent='';I.textContent='';setTimeout(poll,0);return;}}
 if(j.lines.length){const atEnd=L.scrollTop+L.clientHeight>=L.scrollHeight-20;L.textContent+=j.lines.join('');n=j.next;if(atEnd)L.scrollTop=L.scrollHeight;}
 for(const [k,v] of Object.entries(j.images)){if(ver[k]===v)continue;ver[k]=v;let f=document.getElementById('im_'+k);
  if(!f){f=document.createElement('figure');f.id='im_'+k;f.innerHTML='<figcaption></figcaption><img>';f.firstChild.textContent=k;I.prepend(f);}
  f.lastChild.src='/img?name='+encodeURIComponent(k)+'&v='+v;}
 st.textContent='';}catch(e){down=true;st.textContent='（カメラの係が止まっています）';}setTimeout(poll,400);}
poll();
</script></body></html>"""


def _jpg(bgr, width, q=85):
    h, w = bgr.shape[:2]
    if w > width:
        bgr = cv2.resize(bgr, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)
    return cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, q])[1].tobytes()


class Page:
    """ページを出す係。live(画像) で生の映像を差し替え、image・log で写真と文字を足す。押されたボタンは commands() で受け取る。"""

    def __init__(self, title="実機（UR5e）", open_browser=True):
        self.title = title
        self.lock = threading.Lock()
        self.cond = threading.Condition(self.lock)
        self.jpg, self.jpg_no = None, 0
        self.jpg2, self.jpg2_no = None, 0                # 三脚のカメラ（10/20）
        self.images, self.img_ver = {}, {}
        self.lines, self.partial = [], ""
        self.cmds = []
        self.srv = ThreadingHTTPServer(("127.0.0.1", PORT), self._handler())
        self.srv.daemon_threads = True
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        if open_browser:
            open_tab(f"http://127.0.0.1:{PORT}/")

    def live(self, bgr, width=1280):
        b = _jpg(bgr, width, 80)
        with self.lock:
            self.jpg, self.jpg_no = b, self.jpg_no + 1
            self.cond.notify_all()

    def live2(self, bgr, width=1280):
        b = _jpg(bgr, width, 80)
        with self.lock:
            self.jpg2, self.jpg2_no = b, self.jpg2_no + 1
            self.cond.notify_all()

    def image_jpg(self, name, b):
        with self.lock:
            self.images[name] = b
            self.img_ver[name] = self.img_ver.get(name, 0) + 1

    def image(self, name, bgr, width=960):
        self.image_jpg(name, _jpg(bgr, width))

    def log(self, s):
        with self.lock:
            self.partial += s
            if "\n" in self.partial:
                *done, self.partial = self.partial.split("\n")
                self.lines.extend(x + "\n" for x in done)

    def commands(self):
        with self.lock:
            out, self.cmds = self.cmds, []
        return out

    def _handler(self):
        page = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, body, ctype):
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                u = urlparse(self.path)
                q = parse_qs(u.query)
                body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
                if u.path == "/push_log":
                    page.log(body.decode("utf-8", "replace"))
                elif u.path == "/push_img":
                    page.image_jpg(q.get("name", ["写真"])[0], body)
                self._send(b"ok", "text/plain")

            def do_GET(self):
                u = urlparse(self.path)
                q = parse_qs(u.query)
                if u.path == "/":
                    self._send(PAGE.replace("%TITLE%", page.title).encode("utf-8"), "text/html; charset=utf-8")
                elif u.path == "/log":
                    i = int(q.get("from", ["0"])[0])
                    with page.lock:
                        body = {"lines": page.lines[i:], "next": len(page.lines), "images": dict(page.img_ver)}
                    self._send(json.dumps(body, ensure_ascii=False).encode("utf-8"), "application/json")
                elif u.path == "/img":
                    with page.lock:
                        b = page.images.get(q.get("name", [""])[0])
                    if b is None:
                        self.send_error(404)
                    else:
                        self._send(b, "image/jpeg")
                elif u.path == "/cmd":
                    with page.lock:
                        page.cmds.append(q.get("k", [""])[0])
                    self._send(b"ok", "text/plain")
                elif u.path == "/frame":                             # 次の 1 枚（after より新しい物）。1 秒待っても無ければ 204
                    after = int(q.get("after", ["0"])[0] or 0)
                    two = q.get("w", ["1"])[0] == "2"                # w=2 は三脚のカメラ
                    get = (lambda: (page.jpg2, page.jpg2_no)) if two else (lambda: (page.jpg, page.jpg_no))
                    with page.lock:
                        if get()[1] < after:
                            after = -1                               # 係を立ち上げ直して番号が戻った
                        page.cond.wait_for(lambda: get()[0] is not None and get()[1] != after, timeout=1.0)
                        b, no = get()
                    if b is None or no == after:
                        self.send_response(204)
                        self.end_headers()
                        return
                    self.send_response(200)
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("X-No", str(no))
                    self.send_header("Content-Length", str(len(b)))
                    self.end_headers()
                    try:
                        self.wfile.write(b)
                    except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                        pass
                elif u.path == "/stream":
                    self.send_response(200)
                    self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=f")
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    seen = -1
                    try:
                        while True:
                            with page.lock:
                                page.cond.wait_for(lambda: page.jpg_no != seen, timeout=1.0)
                                b, seen = page.jpg, page.jpg_no
                            if b is None:
                                continue
                            self.wfile.write(b"--f\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                             + str(len(b)).encode() + b"\r\n\r\n" + b + b"\r\n")
                    except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                        pass
                else:
                    self.send_error(404)
        return H


def open_tab(url):
    """いつもの Chrome に、普通のタブとして開く（ガントチャートと同じ開き方）。"""
    exe = CHROME if os.path.exists(CHROME) else "chrome"
    try:
        subprocess.Popen([exe, url])
    except OSError:
        import webbrowser
        webbrowser.open(url)


# ---- 送る係（別のプログラムから使う）
_Q = queue.Queue(maxsize=200)
_started = False


def _sender():
    while True:
        path, body = _Q.get()
        try:
            urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", data=body, method="POST"),
                                   timeout=1.0).read()
        except OSError:
            pass                                       # ページ（cam_view）が無い時は捨てる
        _Q.task_done()


def _drain():
    """終わる前に、送り残しを送る（最大 2 秒）。"""
    t0 = time.time()
    while _Q.unfinished_tasks and time.time() - t0 < 2.0:
        time.sleep(0.05)


def _put(path, body):
    global _started
    if not _started:
        import atexit
        threading.Thread(target=_sender, daemon=True).start()
        atexit.register(_drain)
        _started = True
    try:
        _Q.put_nowait((path, body))
    except queue.Full:
        pass


def push_log(s):
    _put("/push_log", s.encode("utf-8"))


def push_image(name, bgr, width=960):
    _put("/push_img?name=" + quote(name), _jpg(bgr, width))


class _Tee(io.TextIOBase):
    def __init__(self, orig):
        self.orig, self.buf = orig, ""

    def write(self, s):
        self.orig.write(s)
        self.buf += s
        if "\n" in self.buf:
            i = self.buf.rindex("\n") + 1
            push_log(self.buf[:i])
            self.buf = self.buf[i:]
        return len(s)

    def flush(self):
        self.orig.flush()


def tee_stdout():
    """print の文字を、画面とページの両方に出す。"""
    if not isinstance(sys.stdout, _Tee):
        sys.stdout = _Tee(sys.stdout)
        push_log(time.strftime("---- %H:%M:%S ") + os.path.basename(sys.argv[0]) + " を始めた\n")

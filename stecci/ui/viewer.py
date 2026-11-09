"""OpenGL viewer for the scan: textured mesh, per-face overlay (classes, cracks, dimmed ground), segment
outlines as 3D lines, face picking through an id buffer, orbit camera and brush cursors.

Per-face state lives in a texture buffer (draw order), segment colours in a second one, so assigning a class
only updates a few bytes. Picking renders face ids (gl_PrimitiveID) into an integer framebuffer that is read
back once per camera position.

Mouse (all tools):  middle-drag / Alt+drag / drag on empty space = orbit  ·  Shift+middle-drag = pan
                    wheel = zoom  ·  double-click = orbit around that point
"""
from __future__ import annotations

import ctypes
import math

import numpy as np
from OpenGL import GL
from PySide6.QtCore import QPointF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QPainter, QPen, QSurfaceFormat
from PySide6.QtOpenGLWidgets import QOpenGLWidget

from . import theme

CRACK_BIT = 1 << 28
SEG_MASK = CRACK_BIT - 1
FOV = 35.0

VS = """
#version 330 core
layout(location=0) in vec3 pos;
layout(location=1) in vec2 uv;
layout(location=2) in vec3 col;
uniform mat4 mvp;
out vec2 vUV; out vec3 vCol; out vec3 vPos;
void main(){ vUV = uv; vCol = col; vPos = pos; gl_Position = mvp * vec4(pos, 1.0); }
"""
FS = """
#version 330 core
in vec2 vUV; in vec3 vCol; in vec3 vPos;
uniform sampler2D tex; uniform bool useTex;
uniform isamplerBuffer faceData; uniform samplerBuffer segColor; uniform usamplerBuffer segFlag;
uniform int base; uniform float opacity; uniform float dim; uniform bool showFill; uniform bool showCrack;
uniform vec3 crackColor; uniform vec3 eye; uniform float shade;
out vec4 frag;
void main(){
  vec3 c = useTex ? texture(tex, vUV).rgb : vCol;
  vec3 n = normalize(cross(dFdx(vPos), dFdy(vPos)));
  float l = abs(dot(n, normalize(eye - vPos)));
  c *= mix(1.0, 0.6 + 0.4 * l, shade);
  int v = texelFetch(faceData, base + gl_PrimitiveID).r;
  if (v < 0) { c *= (1.0 - dim); }
  else {
    int s = v & %d;
    vec4 sc = texelFetch(segColor, s);
    if (showFill) c = mix(c, sc.rgb, sc.a * opacity);
    uint fl = texelFetch(segFlag, s).r;
    if ((fl & 1u) != 0u) {
      // selected: black diagonal crosshatch in screen pixels, like the photo workspace
      vec2 f = gl_FragCoord.xy;
      if (mod(f.x + f.y, 9.0) < 2.0 || mod(f.x - f.y, 9.0) < 2.0) c = mix(c, vec3(0.0), 0.82);
    }
    else if ((fl & 2u) != 0u) c = mix(c, vec3(1.0), 0.18);
    if (showCrack && (v & %d) != 0) c = crackColor;
  }
  frag = vec4(c, 1.0);
}
""" % (SEG_MASK, CRACK_BIT)
PICK_FS = """
#version 330 core
uniform int base;
out uint id;
void main(){ id = uint(base + gl_PrimitiveID + 1); }
"""
LINE_VS = """
#version 330 core
layout(location=0) in vec3 pos;
uniform mat4 mvp;
void main(){ gl_Position = mvp * vec4(pos, 1.0); }
"""
LINE_FS = """
#version 330 core
uniform vec3 color;
out vec4 frag;
void main(){ frag = vec4(color, 1.0); }
"""


def set_default_format():
    fmt = QSurfaceFormat()
    fmt.setVersion(3, 3)
    fmt.setProfile(QSurfaceFormat.CompatibilityProfile)    # wide lines
    fmt.setDepthBufferSize(24)
    fmt.setSamples(4)
    QSurfaceFormat.setDefaultFormat(fmt)


def _program(vs, fs):
    def comp(src, kind):
        s = GL.glCreateShader(kind)
        GL.glShaderSource(s, src)
        GL.glCompileShader(s)
        if not GL.glGetShaderiv(s, GL.GL_COMPILE_STATUS):
            raise RuntimeError(GL.glGetShaderInfoLog(s).decode())
        return s
    p = GL.glCreateProgram()
    GL.glAttachShader(p, comp(vs, GL.GL_VERTEX_SHADER))
    GL.glAttachShader(p, comp(fs, GL.GL_FRAGMENT_SHADER))
    GL.glLinkProgram(p)
    if not GL.glGetProgramiv(p, GL.GL_LINK_STATUS):
        raise RuntimeError(GL.glGetProgramInfoLog(p).decode())
    return p


def look_at(eye, target, up):
    f = target - eye; f /= np.linalg.norm(f)
    s = np.cross(f, up); s /= np.linalg.norm(s)
    u = np.cross(s, f)
    M = np.eye(4)
    M[0, :3], M[1, :3], M[2, :3] = s, u, -f
    M[:3, 3] = -M[:3, :3] @ eye
    return M


def perspective(fovy, aspect, near, far):
    f = 1 / math.tan(math.radians(fovy) / 2)
    M = np.zeros((4, 4))
    M[0, 0], M[1, 1] = f / aspect, f
    M[2, 2], M[2, 3] = (far + near) / (near - far), 2 * far * near / (near - far)
    M[3, 2] = -1
    return M


class Camera:
    def __init__(self):
        self.target = np.zeros(3)
        self.dist = 1.0
        self.yaw, self.pitch = math.radians(-60), math.radians(25)
        self.up = np.array([0, 0, 1.0])
        self.e1 = np.array([1.0, 0, 0])
        self.radius = 1.0

    def set_frame(self, up, forward_hint=None):
        up = np.asarray(up, float) / np.linalg.norm(up)
        h = np.asarray(forward_hint if forward_hint is not None else [1.0, 0, 0], float)
        if abs(h @ up) > 0.95:
            h = np.array([0, 1.0, 0]) if abs(up[1]) < 0.9 else np.array([1.0, 0, 0])
        e1 = h - up * (h @ up)
        self.up, self.e1 = up, e1 / np.linalg.norm(e1)

    @property
    def e2(self):
        return np.cross(self.up, self.e1)

    def direction(self):
        cp = math.cos(self.pitch)
        return cp * (math.cos(self.yaw) * self.e1 + math.sin(self.yaw) * self.e2) + math.sin(self.pitch) * self.up

    def eye(self):
        return self.target + self.dist * self.direction()

    def matrices(self, aspect):
        eye = self.eye()
        d = self.direction()
        up = self.up if abs(d @ self.up) < 0.999 else self.e2
        near = max(self.dist - 2 * self.radius, self.dist * 0.02, 1e-4)
        far = self.dist + 3 * self.radius
        return perspective(FOV, aspect, near, far) @ look_at(eye, self.target, up), eye

    def to_dict(self):
        return dict(target=self.target.tolist(), dist=self.dist, yaw=self.yaw, pitch=self.pitch)

    def from_dict(self, d):
        self.target = np.array(d["target"], float)
        self.dist, self.yaw, self.pitch = float(d["dist"]), float(d["yaw"]), float(d["pitch"])


class Viewer(QOpenGLWidget):
    hovered = Signal(int)                              # face id or -1
    face_press = Signal(int, object, bool)            # face, button, ctrl
    face_drag = Signal(int, object)
    face_release = Signal()
    background_click = Signal()                       # click (no drag) beside the stone
    stroke = Signal(object, object, object, str)      # points (k,3), radii (k,), button, kind
    brush_wheel = Signal(str, int)                    # kind, +1 / -1
    camera_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        theme.manager().changed.connect(self.update)       # background colour
        self.cam = Camera()
        self.tool = "assign"                          # assign | refine | split | crack
        self.brush_px = {"paint": 50, "crack": 5, "split": 3}
        self.scan = None
        self._pending = None                          # data waiting for the GL context
        self._gl_ready = False
        self._progs = None
        self._pages = []                              # (texture id or 0, base face, count)
        self._order = None                            # draw index -> face id
        self._rank = None                             # face id -> draw index
        self._n_lines = {"seg": 0, "outline": 0, "sel": 0}
        self.opts = dict(opacity=0.5, dim=1.0, show_fill=True, show_crack=True, show_bound=True,
                         shade=0.35, crack_color=(230, 30, 30), bound_color=(57, 255, 20),
                         outline_color=(255, 20, 200), bound_width=2.0, show_outline=True)
        self._ids = None                              # cached id-buffer read-back (H,W) uint32
        self._mouse = None
        self._drag = None                             # orbit | pan | assign | brush
        self._stroke_pts = []
        self._last_hover = -2
        self._hover_timer = QTimer(self, singleShot=True, interval=25, timeout=self._do_hover)
        self.centroids = None                         # (m,3) face centroids (for brush hits)
        self._mods = Qt.NoModifier
        self._queued = []

    # ── data ─────────────────────────────────────────────────────────────────
    def set_scan(self, scan, max_tex: int):
        self.scan = scan
        T = scan.V[scan.F]
        self.centroids = T.mean(1)
        self._pending = (scan, max_tex)
        self._ids = None
        lo, hi = scan.V.min(0), scan.V.max(0)
        self.cam.radius = float(np.linalg.norm(hi - lo)) / 2
        self.update()

    def clear(self):
        """Drop the scan (document closed): frees its GPU textures and the per-face data."""
        self.scan = None
        self.centroids = None
        self._pending = None
        self._queued = []
        self._ids = None
        self._stroke_pts = []
        self._drag = None
        if self._gl_ready and self._pages:
            self.makeCurrent()
            for t, _, _ in self._pages:
                if t:
                    GL.glDeleteTextures([t])
            self.doneCurrent()
        self._pages = []
        self._n_lines = {"seg": 0, "outline": 0, "sel": 0}
        self.update()

    def _upload_scan(self):
        scan, max_tex = self._pending
        self._pending = None
        import cv2
        GL.glBindVertexArray(self._vao)
        for loc, arr, size, typ, norm in ((0, scan.V.astype(np.float32), 3, GL.GL_FLOAT, False),
                                          (1, scan.UV.astype(np.float32), 2, GL.GL_FLOAT, False),
                                          (2, scan.C.astype(np.uint8), 3, GL.GL_UNSIGNED_BYTE, True)):
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._vbo[loc])
            GL.glBufferData(GL.GL_ARRAY_BUFFER, arr.nbytes, np.ascontiguousarray(arr), GL.GL_STATIC_DRAW)
            GL.glEnableVertexAttribArray(loc)
            GL.glVertexAttribPointer(loc, size, typ, norm, 0, None)
        order = np.argsort(scan.FT, kind="stable")
        self._order = order
        self._rank = np.empty_like(order)
        self._rank[order] = np.arange(len(order))
        idx = np.ascontiguousarray(scan.F[order].astype(np.uint32))
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self._ebo)
        GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, idx.nbytes, idx, GL.GL_STATIC_DRAW)
        GL.glBindVertexArray(0)
        # textures
        for t, _, _ in self._pages:
            if t:
                GL.glDeleteTextures([t])
        self._pages = []
        maxsz = min(max_tex, int(GL.glGetIntegerv(GL.GL_MAX_TEXTURE_SIZE)))
        ft = scan.FT[order]
        for page in np.unique(ft):
            start = int(np.searchsorted(ft, page, "left"))
            count = int(np.searchsorted(ft, page, "right")) - start
            tid = 0
            if page >= 0:
                img = scan.textures[page]
                if max(img.shape[:2]) > maxsz:
                    s = maxsz / max(img.shape[:2])
                    img = cv2.resize(img, (round(img.shape[1] * s), round(img.shape[0] * s)),
                                     interpolation=cv2.INTER_AREA)
                img = np.ascontiguousarray(img[::-1])             # GL: first row = bottom (v = 0)
                tid = GL.glGenTextures(1)
                GL.glBindTexture(GL.GL_TEXTURE_2D, tid)
                GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
                GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGB8, img.shape[1], img.shape[0], 0, GL.GL_RGB,
                                GL.GL_UNSIGNED_BYTE, img)
                GL.glGenerateMipmap(GL.GL_TEXTURE_2D)
                GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR_MIPMAP_LINEAR)
                GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
                GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
                GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
                try:
                    GL.glTexParameterf(GL.GL_TEXTURE_2D, 0x84FE, 8.0)      # anisotropic filtering
                except Exception:
                    pass
            self._pages.append((tid, start, count))
        self._set_buffer(self._face_tbo, np.full(scan.n_faces, -1, np.int32))
        self._set_buffer(self._seg_tbo, np.zeros((1, 4), np.uint8))
        self._set_buffer(self._flag_tbo, np.zeros(1, np.uint8))
        self._n_lines = {"seg": 0, "outline": 0, "sel": 0}

    def _set_buffer(self, tbo, data: np.ndarray):
        buf, tex, fmt = tbo
        data = np.ascontiguousarray(data)
        GL.glBindBuffer(GL.GL_TEXTURE_BUFFER, buf)
        GL.glBufferData(GL.GL_TEXTURE_BUFFER, max(data.nbytes, 4), data if data.nbytes else None, GL.GL_DYNAMIC_DRAW)
        GL.glBindTexture(GL.GL_TEXTURE_BUFFER, tex)
        GL.glTexBuffer(GL.GL_TEXTURE_BUFFER, fmt, buf)

    def set_face_state(self, stone, seg, crack):
        """Per-face overlay state: ground (-1), segment id, crack bit."""
        if self.scan is None:
            return
        # before isolation nothing is ground: the whole scan is shown undimmed
        v = np.full(self.scan.n_faces, -1 if stone is not None else 0, np.int32)
        if stone is not None:
            s = seg if seg is not None else np.zeros(self.scan.n_faces, np.int32)
            v[stone] = s[stone]
            if crack is not None:
                v[stone & crack] |= CRACK_BIT
        elif crack is not None:
            v[crack] = CRACK_BIT
        self._run_gl(lambda: self._set_buffer(self._face_tbo, v[self._order]))

    def set_segment_colors(self, rgba: np.ndarray, flags: np.ndarray):
        """rgba (n,4) uint8 per segment id (a = 0: no colour), flags (n,) uint8: 1 selected, 2 hover."""
        self._run_gl(lambda: (self._set_buffer(self._seg_tbo, rgba.astype(np.uint8)),
                              self._set_buffer(self._flag_tbo, flags.astype(np.uint8))))

    def set_lines(self, which: str, P: np.ndarray, edges: np.ndarray):
        pts = np.ascontiguousarray(P[edges.ravel()].astype(np.float32)) if len(edges) else np.zeros((0, 3), np.float32)

        def up():
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._line_vbo[which])
            GL.glBufferData(GL.GL_ARRAY_BUFFER, max(pts.nbytes, 12), pts if pts.nbytes else None, GL.GL_DYNAMIC_DRAW)
            self._n_lines[which] = len(pts)
        self._run_gl(up)

    def _run_gl(self, fn):
        if not self._gl_ready or self._pending is not None:
            # context or scan not uploaded yet: run after the next upload
            self._queued.append(fn)
            self.update()
            return
        self.makeCurrent()
        fn()
        self.doneCurrent()
        self.update()

    # ── camera ───────────────────────────────────────────────────────────────
    def set_up(self, up, forward_hint=None):
        self.cam.set_frame(up, forward_hint)
        self._camera_moved()

    def fit(self, mask=None):
        if self.scan is None:
            return
        V = self.scan.V
        if mask is not None and mask.any():
            V = V[np.unique(self.scan.F[mask])]
        lo, hi = V.min(0), V.max(0)
        self.cam.target = ((lo + hi) / 2).astype(float)
        r = float(np.linalg.norm(hi - lo)) / 2
        self.cam.radius = max(r, 1e-6)
        aspect = max(self.width(), 1) / max(self.height(), 1)
        half = math.radians(FOV) / 2
        fit_angle = half if aspect >= 1 else math.atan(math.tan(half) * aspect)
        self.cam.dist = r / math.sin(fit_angle) * 1.05
        self._camera_moved()

    def view_preset(self, name: str):
        presets = {"top": (-90, 89.9), "front": (90, 8), "back": (-90, 8), "right": (0, 8), "left": (180, 8),
                   "iso": (55, 28)}
        y, p = presets[name]
        self.cam.yaw, self.cam.pitch = math.radians(y), math.radians(p)
        self._camera_moved()

    def _camera_moved(self):
        self._ids = None
        self.camera_changed.emit()
        self.update()

    # ── GL ───────────────────────────────────────────────────────────────────
    def initializeGL(self):
        self._prog = _program(VS, FS)
        self._pick = _program(VS, PICK_FS)
        self._line = _program(LINE_VS, LINE_FS)
        self._vao = GL.glGenVertexArrays(1)
        self._vbo = list(GL.glGenBuffers(3))
        self._ebo = GL.glGenBuffers(1)
        self._line_vao = GL.glGenVertexArrays(1)
        self._line_vbo = {"seg": GL.glGenBuffers(1), "outline": GL.glGenBuffers(1), "sel": GL.glGenBuffers(1)}

        def tbo(fmt):
            return (GL.glGenBuffers(1), GL.glGenTextures(1), fmt)
        self._face_tbo = tbo(GL.GL_R32I)
        self._seg_tbo = tbo(GL.GL_RGBA8)
        self._flag_tbo = tbo(GL.GL_R8UI)
        self._fbo = None
        self._gl_ready = True

    def _uniforms(self, prog):
        return {n: GL.glGetUniformLocation(prog, n) for n in
                ("mvp", "tex", "useTex", "faceData", "segColor", "segFlag", "base", "opacity", "dim", "showFill",
                 "showCrack", "crackColor", "eye", "shade", "color")}

    def _draw_mesh(self, prog, mvp, eye, pick=False):
        u = self._uniforms(prog)
        GL.glUseProgram(prog)
        GL.glUniformMatrix4fv(u["mvp"], 1, GL.GL_TRUE, mvp.astype(np.float32))
        if not pick:
            o = self.opts
            GL.glUniform1f(u["opacity"], o["opacity"])
            GL.glUniform1f(u["dim"], o["dim"])
            GL.glUniform1i(u["showFill"], int(o["show_fill"]))
            GL.glUniform1i(u["showCrack"], int(o["show_crack"]))
            GL.glUniform3f(u["crackColor"], *[c / 255 for c in o["crack_color"]])
            GL.glUniform3f(u["eye"], *eye.astype(np.float32))
            GL.glUniform1f(u["shade"], o["shade"])
            for unit, (name, tbo) in enumerate((("faceData", self._face_tbo), ("segColor", self._seg_tbo),
                                                ("segFlag", self._flag_tbo)), start=1):
                GL.glActiveTexture(GL.GL_TEXTURE0 + unit)
                GL.glBindTexture(GL.GL_TEXTURE_BUFFER, tbo[1])
                GL.glUniform1i(u[name], unit)
            GL.glUniform1i(u["tex"], 0)
        GL.glBindVertexArray(self._vao)
        for tid, start, count in self._pages:
            if not pick:
                GL.glActiveTexture(GL.GL_TEXTURE0)
                GL.glBindTexture(GL.GL_TEXTURE_2D, tid)
                GL.glUniform1i(u["useTex"], int(tid != 0))
            GL.glUniform1i(u["base"], start)
            GL.glDrawElements(GL.GL_TRIANGLES, 3 * count, GL.GL_UNSIGNED_INT, ctypes.c_void_p(12 * start))
        GL.glBindVertexArray(0)

    def paintGL(self):
        if self._pending is not None and self._gl_ready:
            self._upload_scan()
            for fn in self._queued:
                fn()
            self._queued = []
        bg = QColor(theme.current().viewer)
        GL.glClearColor(bg.redF(), bg.greenF(), bg.blueF(), 1.0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        if self.scan is not None and self._pages:
            dpr = self.devicePixelRatioF()
            W, H = int(self.width() * dpr), int(self.height() * dpr)
            GL.glViewport(0, 0, W, H)
            mvp, eye = self.cam.matrices(W / max(H, 1))
            GL.glEnable(GL.GL_DEPTH_TEST)
            GL.glDepthFunc(GL.GL_LEQUAL)
            GL.glEnable(GL.GL_POLYGON_OFFSET_FILL)
            GL.glPolygonOffset(1.0, 2.0)
            self._draw_mesh(self._prog, mvp, eye)
            GL.glDisable(GL.GL_POLYGON_OFFSET_FILL)
            o = self.opts
            GL.glUseProgram(self._line)
            u = self._uniforms(self._line)
            GL.glUniformMatrix4fv(u["mvp"], 1, GL.GL_TRUE, mvp.astype(np.float32))
            GL.glBindVertexArray(self._line_vao)
            for which, show, colour, width in (("seg", o["show_bound"], o["bound_color"], o["bound_width"]),
                                               ("outline", o["show_outline"], o["outline_color"], 2.0),
                                               ("sel", True, (0, 0, 0), 3.0)):     # selection outline
                n = self._n_lines[which]
                if not show or n == 0:
                    continue
                GL.glUniform3f(u["color"], *[c / 255 for c in colour])
                GL.glLineWidth(float(width) * dpr)
                GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._line_vbo[which])
                GL.glEnableVertexAttribArray(0)
                GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, False, 0, None)
                GL.glDrawArrays(GL.GL_LINES, 0, n)
            GL.glBindVertexArray(0)
            GL.glUseProgram(0)
            GL.glDisable(GL.GL_DEPTH_TEST)
        self._paint_overlay()

    def _paint_overlay(self):
        """Brush cursor and the stroke being drawn (QPainter on top of the GL image)."""
        kind = self._brush_kind()
        if kind is None or self._mouse is None or self.scan is None:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = self.brush_px[kind] / 2
        if len(self._stroke_pts) > 1:
            pen = QPen(QColor(255, 255, 255, 110), 2 * r, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
            p.setPen(pen)
            for a, b in zip(self._stroke_pts[:-1], self._stroke_pts[1:]):
                p.drawLine(a, b)
        p.setPen(QPen(QColor(255, 255, 255, 220), 1.5))
        p.setBrush(Qt.NoBrush)
        p.drawEllipse(self._mouse, r, r)
        p.end()

    def _brush_kind(self):
        mods = self._mods
        if self.tool == "crack":
            return "crack"
        if self.tool == "split":
            return "split"
        if self.tool == "refine" or (self.tool == "assign" and mods & Qt.ShiftModifier):
            return "paint"
        return None

    # ── picking ──────────────────────────────────────────────────────────────
    def _id_buffer(self):
        if self._ids is not None:
            return self._ids
        if self.scan is None or not self._pages:
            return None
        self.makeCurrent()
        dpr = self.devicePixelRatioF()
        W, H = int(self.width() * dpr), int(self.height() * dpr)
        if self._fbo is None or self._fbo[3] != (W, H):
            if self._fbo is not None:
                GL.glDeleteFramebuffers(1, [self._fbo[0]])
                GL.glDeleteRenderbuffers(2, [self._fbo[1], self._fbo[2]])
            fbo = GL.glGenFramebuffers(1)
            rb_id, rb_d = GL.glGenRenderbuffers(2)
            GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, rb_id)
            GL.glRenderbufferStorage(GL.GL_RENDERBUFFER, GL.GL_R32UI, W, H)
            GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, rb_d)
            GL.glRenderbufferStorage(GL.GL_RENDERBUFFER, GL.GL_DEPTH_COMPONENT24, W, H)
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fbo)
            GL.glFramebufferRenderbuffer(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_RENDERBUFFER, rb_id)
            GL.glFramebufferRenderbuffer(GL.GL_FRAMEBUFFER, GL.GL_DEPTH_ATTACHMENT, GL.GL_RENDERBUFFER, rb_d)
            self._fbo = (fbo, rb_id, rb_d, (W, H))
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._fbo[0])
        GL.glViewport(0, 0, W, H)
        GL.glClearBufferuiv(GL.GL_COLOR, 0, np.zeros(4, np.uint32))
        GL.glClear(GL.GL_DEPTH_BUFFER_BIT)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDisable(GL.GL_MULTISAMPLE)
        mvp, eye = self.cam.matrices(W / max(H, 1))
        self._draw_mesh(self._pick, mvp, eye, pick=True)
        GL.glPixelStorei(GL.GL_PACK_ALIGNMENT, 4)
        data = GL.glReadPixels(0, 0, W, H, GL.GL_RED_INTEGER, GL.GL_UNSIGNED_INT)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.defaultFramebufferObject())
        GL.glEnable(GL.GL_MULTISAMPLE)
        self.doneCurrent()
        ids = np.frombuffer(data, np.uint32).reshape(H, W)[::-1]
        self._ids = ids
        return ids

    def face_at(self, pos: QPointF) -> int:
        ids = self._id_buffer()
        if ids is None:
            return -1
        dpr = self.devicePixelRatioF()
        x, y = int(pos.x() * dpr), int(pos.y() * dpr)
        if not (0 <= y < ids.shape[0] and 0 <= x < ids.shape[1]):
            return -1
        v = int(ids[y, x])
        return int(self._order[v - 1]) if v else -1

    def world_per_px(self, point) -> float:
        d = float(np.linalg.norm(self.cam.eye() - point))
        return 2 * d * math.tan(math.radians(FOV) / 2) / max(self.height(), 1)

    # ── mouse ────────────────────────────────────────────────────────────────
    def _nav_button(self, e) -> str | None:
        b, mods = e.button(), e.modifiers()
        if b == Qt.MiddleButton:
            return "pan" if mods & Qt.ShiftModifier else "orbit"
        if b == Qt.LeftButton and mods & Qt.AltModifier:
            return "orbit"
        return None

    def mousePressEvent(self, e):
        self._mods = e.modifiers()
        self._press = e.position()
        self._mouse = e.position()
        nav = self._nav_button(e)
        if nav:
            self._drag = nav
            return
        if e.button() not in (Qt.LeftButton, Qt.RightButton):
            return
        face = self.face_at(e.position())
        kind = self._brush_kind()
        if kind is not None:
            self._drag = "brush"
            self._stroke_btn = e.button()
            self._stroke_pts = [e.position()]
            self._stroke_hits = []
            self._add_hit(e.position())
            self.update()
            return
        if face < 0:
            self._drag = "orbit" if e.button() == Qt.LeftButton else "pan"
            self._bg_press = True
            return
        self._drag = "assign"
        self.face_press.emit(face, e.button(), bool(e.modifiers() & Qt.ControlModifier))

    def _add_hit(self, pos):
        f = self.face_at(pos)
        if f >= 0:
            c = self.centroids[f]
            self._stroke_hits.append((c, self.world_per_px(c) * self.brush_px[self._brush_kind() or "paint"] / 2))

    def mouseMoveEvent(self, e):
        self._mods = e.modifiers()
        pos = e.position()
        last = self._mouse or pos
        self._mouse = pos
        if self._drag in ("orbit", "pan"):
            dx, dy = pos.x() - last.x(), pos.y() - last.y()
            if self._drag == "orbit":
                self.cam.yaw -= dx * 0.008
                self.cam.pitch = float(np.clip(self.cam.pitch + dy * 0.008, -1.55, 1.55))
            else:
                k = self.world_per_px(self.cam.target)
                d = self.cam.direction()
                right = np.cross(self.cam.up, d)
                right /= max(np.linalg.norm(right), 1e-12)
                upv = np.cross(d, right)
                self.cam.target = self.cam.target + (-dx * right + dy * upv) * k
            self._camera_moved()
            return
        if self._drag == "brush":
            # sample along the mouse path so fast strokes have no gaps
            r = max(self.brush_px[self._brush_kind() or "paint"] / 4, 1.0)
            dist = math.hypot(pos.x() - last.x(), pos.y() - last.y())
            n = max(1, int(dist / r))
            for i in range(1, n + 1):
                t = i / n
                self._add_hit(QPointF(last.x() + (pos.x() - last.x()) * t, last.y() + (pos.y() - last.y()) * t))
            self._stroke_pts.append(pos)
            self.update()
            return
        if self._drag == "assign":
            f = self.face_at(pos)
            if f >= 0:
                self.face_drag.emit(f, e.buttons() & (Qt.LeftButton | Qt.RightButton))
            return
        self._hover_timer.start()
        if self._brush_kind() is not None:
            self.update()

    def _do_hover(self):
        if self._mouse is None or self._drag is not None:
            return
        f = self.face_at(self._mouse)
        if f != self._last_hover:
            self._last_hover = f
            self.hovered.emit(f)

    def mouseReleaseEvent(self, e):
        drag = self._drag
        self._drag = None
        bg, self._bg_press = getattr(self, "_bg_press", False), False
        if bg and self._press is not None:
            moved = math.hypot(e.position().x() - self._press.x(), e.position().y() - self._press.y())
            if moved < 4:
                self.background_click.emit()
            return
        if drag == "brush":
            hits = self._stroke_hits
            self._stroke_pts = []
            self.update()
            if hits:
                pts = np.array([h[0] for h in hits])
                radii = np.array([h[1] for h in hits])
                self.stroke.emit(pts, radii, self._stroke_btn, self._brush_kind() or "paint")
        elif drag == "assign":
            self.face_release.emit()

    def mouseDoubleClickEvent(self, e):
        """Double-click: orbit around the clicked point."""
        if e.button() == Qt.LeftButton:
            f = self.face_at(e.position())
            if f >= 0:
                self.cam.target = self.centroids[f].astype(float)
                self._camera_moved()

    def wheelEvent(self, e):
        steps = e.angleDelta().y() / 120
        mods = e.modifiers()
        if mods & Qt.ShiftModifier:
            self.brush_wheel.emit("paint", 1 if steps > 0 else -1)
            return
        if mods & Qt.ControlModifier:
            self.brush_wheel.emit("crack", 1 if steps > 0 else -1)
            return
        self.cam.dist *= 0.87 ** steps
        self.cam.dist = float(np.clip(self.cam.dist, self.cam.radius * 0.02, self.cam.radius * 20))
        self._camera_moved()

    def keyPressEvent(self, e):
        self._mods = e.modifiers()
        self.update()
        super().keyPressEvent(e)

    def keyReleaseEvent(self, e):
        self._mods = e.modifiers()
        self.update()
        super().keyReleaseEvent(e)

    def leaveEvent(self, e):
        self._mouse = None
        self.hovered.emit(-1)
        self._last_hover = -2
        self.update()

    def resizeGL(self, w, h):
        self._ids = None

from __future__ import annotations

import base64
import json
import math
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent
OLLAMA = "http://127.0.0.1:11434"
MAX_BODY = 16 * 1024 * 1024
MAX_CONTEXT_TOKENS = 32768
MESH_SCHEMA = {
    "type": "object",
    "properties": {
        "description": {"type": "string"},
        "vertices": {
            "type": "array",
            "items": {
                "type": "array",
                "items": {"type": "number"},
                "minItems": 3,
                "maxItems": 3,
            },
        },
        "faces": {
            "type": "array",
            "items": {
                "type": "array",
                "items": {"type": "integer"},
                "minItems": 3,
            },
        },
    },
    "required": ["description", "vertices", "faces"],
}


def ollama_request(path: str, payload: dict | None = None) -> dict:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(
        f"{OLLAMA}{path}",
        data=body,
        headers={"Content-Type": "application/json"} if body is not None else {},
        method="POST" if body is not None else "GET",
    )
    with urlopen(request, timeout=600 if path == "/api/chat" else 4) as response:
        return json.loads(response.read().decode("utf-8"))


def ollama_stream(payload: dict):
    request = Request(
        f"{OLLAMA}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    return urlopen(request, timeout=None)


def validate_mesh(data: object) -> tuple[str, str, int, int]:
    if not isinstance(data, dict):
        raise ValueError("模型返回的网格不是 JSON 对象。")
    description = data.get("description")
    vertices = data.get("vertices")
    faces = data.get("faces")
    if not isinstance(description, str) or not isinstance(vertices, list) or not isinstance(faces, list):
        raise ValueError("模型返回缺少 description、vertices 或 faces 字段。")
    if len(vertices) < 4:
        raise ValueError("顶点数量至少需要 4 个。")
    if len(faces) < 4:
        raise ValueError("面数量至少需要 4 个。")

    coordinates = []
    for vertex in vertices:
        if not isinstance(vertex, list) or len(vertex) != 3:
            raise ValueError("每个顶点必须包含 3 个坐标。")
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) for value in vertex):
            raise ValueError("顶点坐标必须是有限数字。")
        coordinates.append(vertex)

    minimum = [min(vertex[axis] for vertex in coordinates) for axis in range(3)]
    maximum = [max(vertex[axis] for vertex in coordinates) for axis in range(3)]
    center = [(minimum[axis] + maximum[axis]) / 2 for axis in range(3)]
    scale = max(maximum[axis] - minimum[axis] for axis in range(3)) / 2
    if not math.isfinite(scale) or scale == 0:
        raise ValueError("模型顶点没有有效的空间尺寸。")

    lines = ["# Generated locally from a reference image with Ollama", "o LocalReferenceMesh"]
    for vertex in coordinates:
        normalized = [(vertex[axis] - center[axis]) / scale for axis in range(3)]
        lines.append("v " + " ".join(f"{value:.6f}" for value in normalized))

    one_based = all(
        isinstance(face, list)
        and all(
            isinstance(index, int)
            and not isinstance(index, bool)
            and 1 <= index <= len(vertices)
            for index in face
        )
        for face in faces
    ) and any(index == len(vertices) for face in faces for index in face)

    valid_faces = []
    for face in faces:
        if not isinstance(face, list) or not 3 <= len(face) <= 32:
            raise ValueError("每个面必须包含 3 到 32 个顶点索引。")
        if one_based:
            face = [index - 1 for index in face]
        if any(isinstance(index, bool) or not isinstance(index, int) or index < 0 or index >= len(vertices) for index in face):
            continue
        if len(set(face)) < 3:
            continue
        valid_faces.append(face)

    if len(valid_faces) < 4:
        raise ValueError("模型生成的有效面不足 4 个，请重试或换用其他视觉模型。")
    for face in valid_faces:
        lines.append("f " + " ".join(str(index + 1) for index in face))

    return description.strip()[:1000], "\n".join(lines) + "\n", len(vertices), len(valid_faces)


class LocalHandler(BaseHTTPRequestHandler):
    server_version = "ImageToModelLocal/1.0"

    def end_headers(self) -> None:
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self' data: blob:; style-src 'self'; script-src 'self'; connect-src 'self'; object-src 'none'; base-uri 'none'")
        super().end_headers()

    def send_json(self, status: int, data: dict) -> None:
        encoded = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def start_event_stream(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Cache-Control", "no-cache, no-transform")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        self._streaming_response = True

    def send_event(self, data: dict) -> None:
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode("utf-8") + b"\n")
        self.wfile.flush()

    def send_failure(self, status: int, message: str) -> None:
        if getattr(self, "_streaming_response", False):
            self.send_event({"type": "error", "error": message})
        else:
            self.send_json(status, {"error": message})

    def read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > MAX_BODY:
            raise ValueError("请求为空或超过 16 MB 限制。")
        result = json.loads(self.rfile.read(length).decode("utf-8"))
        if not isinstance(result, dict):
            raise ValueError("请求格式无效。")
        return result

    def do_GET(self) -> None:
        if not self.client_address[0].startswith("127.") and self.client_address[0] != "::1":
            self.send_error(403)
            return
        if self.path == "/api/status":
            try:
                data = ollama_request("/api/tags")
                self.send_json(200, {"ollama": True, "models": len(data.get("models", []))})
            except (URLError, TimeoutError, OSError, ValueError):
                self.send_json(200, {"ollama": False, "models": 0})
            return
        if self.path == "/api/models":
            try:
                data = ollama_request("/api/tags")
                models = []
                for item in data.get("models", []):
                    name = item.get("name")
                    if not name:
                        continue
                    details = ollama_request("/api/show", {"model": name})
                    capabilities = details.get("capabilities", [])
                    models.append({
                        "name": name,
                        "size": item.get("size", 0),
                        "vision": "vision" in capabilities,
                    })
                self.send_json(200, {"models": models})
            except (HTTPError, URLError, TimeoutError, OSError, ValueError) as error:
                self.send_json(503, {"error": f"无法连接本机 Ollama：{error}"})
            return
        files = {"/": ("index.html", "text/html; charset=utf-8"), "/app.js": ("app.js", "text/javascript; charset=utf-8"), "/styles.css": ("styles.css", "text/css; charset=utf-8"), "/favicon.ico": (None, "")}
        if self.path not in files:
            self.send_error(404)
            return
        filename, content_type = files[self.path]
        if filename is None:
            self.send_response(204)
            self.end_headers()
            return
        content = (ROOT / filename).read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def do_POST(self) -> None:
        if self.path != "/api/generate":
            self.send_error(404)
            return
        self._streaming_response = False
        try:
            payload = self.read_json()
            model = payload.get("model")
            image = payload.get("image")
            mime = payload.get("mime", "image/png")
            if not isinstance(model, str) or not re.fullmatch(r"[\w.:/-]{1,120}", model):
                raise ValueError("请选择一个有效的本地模型。")
            if mime not in {"image/png", "image/jpeg", "image/webp", "image/gif"}:
                raise ValueError("图片格式仅支持 PNG、JPG、WebP 或 GIF。")
            if not isinstance(image, str) or len(image) > 15 * 1024 * 1024:
                raise ValueError("图片为空或超过大小限制。")
            try:
                base64.b64decode(image, validate=True)
            except ValueError as error:
                raise ValueError("图片数据无效。") from error

            prompt = (
                "Reconstruct and COMPLETE the full 3D scene suggested by the reference image, not only its central subject. "
                "Return JSON matching the requested schema only. Use X for image left/right, Z for up, and Y for depth; place the camera at negative Y looking toward positive Y. "
                "For an interior, build a coherent room with floor, visible walls, plausible ceiling and corners, openings, and the major furniture and objects. "
                "Continue partially cropped furniture beyond the image edge. Infer occluded and off-camera areas using perspective, visible edges, symmetry, materials, "
                "and ordinary room layouts; make reasonable complete guesses instead of leaving cut-off or hidden parts open. For other subjects, complete their hidden backs "
                "and cropped geometry in the same way. Keep dimensions and perspective coherent, and spend the available output detail on the whole scene rather than one small object. "
                "Build real surfaces and volume, not a point cloud. A scene may contain multiple separate but spatially consistent components. "
                "Use zero-based vertex indices, consistent face winding, and 3-6 indices per face. Avoid degenerate faces. "
                "Coordinates may use any finite scale; the application will center and normalize them. In description, distinguish visible evidence from inferred details "
                "and briefly note uncertainty inherent in a single image."
            )
            request_payload = {
                "model": model,
                "stream": True,
                "format": MESH_SCHEMA,
                "options": {"temperature": 0.2, "num_ctx": 32768, "num_predict": -1},
                "messages": [{"role": "user", "content": prompt, "images": [image]}],
            }

            self.start_event_stream()
            self.send_event({"type": "progress", "phase": "loading", "message": "正在读取模型上下文容量…"})
            details = ollama_request("/api/show", {"model": model})
            model_info = details.get("model_info", {})
            context_lengths = [
                value for key, value in model_info.items()
                if key.endswith(".context_length") and isinstance(value, int) and value > 0
            ]
            if context_lengths:
                request_payload["options"]["num_ctx"] = min(MAX_CONTEXT_TOKENS, max(context_lengths))
            self.send_event({
                "type": "progress",
                "phase": "waiting",
                "message": f"上下文窗口 {request_payload['options']['num_ctx']:,} tokens，正在等待本地模型…",
            })
            content_parts = []
            character_count = 0
            with ollama_stream(request_payload) as response:
                self.send_event({"type": "progress", "phase": "generating", "message": "模型正在生成网格数据…"})
                for line in response:
                    if not line.strip():
                        continue
                    chunk = json.loads(line.decode("utf-8"))
                    text = chunk.get("message", {}).get("content", "")
                    if text:
                        content_parts.append(text)
                        character_count += len(text)
                        self.send_event({
                            "type": "progress",
                            "phase": "generating",
                            "characters": character_count,
                            "message": f"正在生成网格数据，已接收 {character_count:,} 个字符…",
                        })
                    if chunk.get("done"):
                        break
            content = "".join(content_parts)
            mesh_data = json.loads(content)
            self.send_event({"type": "progress", "phase": "processing", "message": "模型输出已完成，正在校验并整理网格…"})
            description, obj, vertices, faces = validate_mesh(mesh_data)
            self.send_event({"type": "complete", "description": description, "obj": obj, "vertices": vertices, "faces": faces})
        except ValueError as error:
            self.send_failure(400, str(error))
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")[:800]
            self.send_failure(502, f"Ollama 请求失败 ({error.code})：{detail or error.reason}")
        except (URLError, TimeoutError, OSError) as error:
            reason = getattr(error, "reason", error)
            if isinstance(reason, TimeoutError):
                self.send_failure(504, "本地视觉模型连接失败。应用没有设置推理超时，请检查 Ollama 是否仍在运行。")
            else:
                self.send_failure(503, f"无法连接本机 Ollama：{error}")
        except (json.JSONDecodeError, KeyError, TypeError) as error:
            self.send_failure(502, f"模型返回的数据无法解析，请尝试其他视觉模型。({error})")

    def log_message(self, format: str, *args: object) -> None:
        print(f"[{self.log_date_time_string()}] {args[0]}")


def main() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 8765), LocalHandler)
    print("图片变模型已启动：http://127.0.0.1:8765")
    print("仅监听本机；Ollama 地址固定为 http://127.0.0.1:11434")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n服务已停止。")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

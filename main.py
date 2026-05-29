import os
import uuid
import json
import threading
import subprocess
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import google.generativeai as genai
from faster_whisper import WhisperModel
import httpx

# ── Init ────────────────────────────────────────────────────────────────────
genai.configure(api_key=os.environ.get("GOOGLE_AI_API_KEY", ""))

_whisper = None

def get_whisper():
    global _whisper
    if _whisper is None:
        print("Loading Whisper model...")
        _whisper = WhisperModel("base", device="cpu", compute_type="int8")
        print("Whisper ready.")
    return _whisper

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

jobs: dict = {}  # job_id → {status, progress, result_path, error}

# ── Helpers ──────────────────────────────────────────────────────────────────

def upd(job_id: str, **kwargs):
    if job_id in jobs:
        jobs[job_id].update(kwargs)

def fmt_srt(t: float) -> str:
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    ms = int((t % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def run(cmd: list, **kwargs):
    result = subprocess.run(cmd, capture_output=True, text=True, **kwargs)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{result.stderr}")
    return result

# ── Main processing pipeline ─────────────────────────────────────────────────

def process_job(job_id: str, input_path: str, download_url: str | None = None):
    tmp = Path(f"/tmp/{job_id}")
    tmp.mkdir(exist_ok=True)

    try:
        # 0. Download from URL if needed
        if download_url:
            upd(job_id, status="영상 다운로드 중", progress=3)
            with httpx.stream("GET", download_url, timeout=600, follow_redirects=True) as r:
                r.raise_for_status()
                with open(input_path, "wb") as f:
                    for chunk in r.iter_bytes(chunk_size=1024 * 1024):
                        f.write(chunk)

        # 1. Extract audio
        upd(job_id, status="음성 추출 중", progress=10)
        audio_path = str(tmp / "audio.wav")
        run(["ffmpeg", "-i", input_path,
             "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
             audio_path, "-y"])

        # 2. Whisper STT
        upd(job_id, status="음성 인식 중 (Whisper)", progress=25)
        segs, _ = get_whisper().transcribe(audio_path, language="ko")
        transcript = [
            {"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip()}
            for s in segs if s.text.strip()
        ]

        if not transcript:
            raise ValueError("음성이 감지되지 않았습니다.")

        # 3. Gemini cut planning
        upd(job_id, status="AI 스토리 분석 중", progress=50)
        model = genai.GenerativeModel("gemini-2.5-flash")
        prompt = f"""영상 음성 인식 결과입니다. 핵심 구간만 남기고 불필요한 구간을 제거하세요.

제거 대상: 침묵, "음...", "어...", 말 더듬기, 반복 설명, 군더더기
유지 대상: 핵심 설명, 행동 묘사, 결론, 정보 전달

원본의 60~70% 길이를 목표로 하세요.

음성 인식 결과:
{json.dumps(transcript, ensure_ascii=False)}

JSON만 출력 (마크다운 없이):
{{
  "keep": [{{"start": 0.0, "end": 3.5}}, ...],
  "subtitles": [{{"start": 0.0, "end": 3.5, "text": "자막 내용"}}, ...]
}}"""

        resp = model.generate_content(prompt)
        raw = resp.text.strip().strip("```").lstrip("json").strip()
        plan = json.loads(raw)
        keep = plan.get("keep", [])
        subs = plan.get("subtitles", [])

        if not keep:
            keep = [{"start": transcript[0]["start"], "end": transcript[-1]["end"]}]

        # 4. Get video duration
        probe = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", input_path],
            capture_output=True, text=True
        )
        duration = float(json.loads(probe.stdout)["format"]["duration"])

        # 5. Cut clips
        upd(job_id, status="컷편집 중", progress=65)
        clip_paths = []
        for i, seg in enumerate(keep):
            cp = str(tmp / f"clip_{i:03d}.mp4")
            run(["ffmpeg",
                 "-ss", str(seg["start"]),
                 "-to", str(min(seg["end"], duration)),
                 "-i", input_path,
                 "-c:v", "libx264", "-c:a", "aac",
                 "-avoid_negative_ts", "make_zero",
                 cp, "-y"])
            clip_paths.append(cp)

        # 6. Concat
        upd(job_id, status="영상 합치는 중", progress=75)
        concat_txt = str(tmp / "concat.txt")
        with open(concat_txt, "w") as f:
            for cp in clip_paths:
                f.write(f"file '{cp}'\n")

        merged = str(tmp / "merged.mp4")
        run(["ffmpeg", "-f", "concat", "-safe", "0",
             "-i", concat_txt, "-c", "copy", merged, "-y"])

        # 7. Build adjusted SRT
        upd(job_id, status="자막 생성 중", progress=82)
        merged_t = 0.0
        time_map = []
        for seg in keep:
            d = seg["end"] - seg["start"]
            time_map.append((seg["start"], seg["end"], merged_t))
            merged_t += d

        def to_merged(orig):
            for o_s, o_e, m_s in time_map:
                if o_s <= orig <= o_e:
                    return m_s + (orig - o_s)
            return None

        srt_path = str(tmp / "subs.srt")
        lines = []
        idx = 1
        for sub in subs:
            ms = to_merged(sub["start"])
            me = to_merged(sub["end"])
            if ms is None or me is None or me <= ms:
                continue
            text = sub["text"]
            if len(text) > 28:
                mid = len(text) // 2
                sp = text.rfind(" ", 0, mid + 6)
                if sp > 0:
                    text = text[:sp] + "\n" + text[sp + 1:]
            lines += [str(idx), f"{fmt_srt(ms)} --> {fmt_srt(me)}", text, ""]
            idx += 1

        with open(srt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        # 8. Burn subtitles
        upd(job_id, status="자막 입히는 중", progress=90)
        output = str(tmp / "result.mp4")
        run(["ffmpeg", "-i", merged,
             "-vf", f"subtitles={srt_path}:force_style='FontName=NanumGothic,FontSize=20,PrimaryColour=&Hffffff,OutlineColour=&H000000,Outline=2,Alignment=2'",
             "-c:v", "libx264", "-c:a", "copy",
             output, "-y"])

        os.remove(input_path)
        upd(job_id, status="done", progress=100, result_path=output)

    except Exception as e:
        upd(job_id, status="error", progress=0, error=str(e))

# ── Routes ───────────────────────────────────────────────────────────────────

class UrlRequest(BaseModel):
    url: str
    filename: str = "video.mp4"
    job_id: str | None = None

@app.get("/")
def root():
    return {"status": "autocut api running"}

@app.post("/process")
async def process(file: UploadFile = File(...)):
    job_id = str(uuid.uuid4())
    ext = Path(file.filename or "video.mp4").suffix or ".mp4"
    input_path = f"/tmp/{job_id}_input{ext}"

    with open(input_path, "wb") as f:
        f.write(await file.read())

    jobs[job_id] = {"status": "처리 시작", "progress": 5, "result_path": None, "error": None}

    t = threading.Thread(target=process_job, args=(job_id, input_path))
    t.daemon = True
    t.start()

    return {"job_id": job_id}

@app.post("/process-url")
async def process_from_url(body: UrlRequest):
    job_id = body.job_id or str(uuid.uuid4())
    ext = Path(body.filename).suffix or ".mp4"
    input_path = f"/tmp/{job_id}_input{ext}"

    jobs[job_id] = {"status": "다운로드 대기", "progress": 2, "result_path": None, "error": None}

    t = threading.Thread(target=process_job, args=(job_id, input_path, body.url))
    t.daemon = True
    t.start()

    return {"job_id": job_id}

@app.get("/status/{job_id}")
def status(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        "status": job["status"],
        "progress": job["progress"],
        "done": job["status"] == "done",
        "error": job.get("error"),
    }

@app.get("/download/{job_id}")
def download(job_id: str):
    job = jobs.get(job_id)
    if not job or job["status"] != "done":
        raise HTTPException(status_code=400, detail="Not ready")
    return FileResponse(job["result_path"], media_type="video/mp4", filename="autocut_result.mp4")

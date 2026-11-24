"""Generate a modern-TTS fake-speech dataset (Microsoft neural voices + Gemini TTS), Hindi and English.

Fully automated and resumable: rerun the same command after an interruption, quota stop or
Colab disconnect, and it continues where it stopped. Every clip gets a row in manifest.csv.

  # Microsoft only (free, no key): about 1 hour for ~2,500 clips
  python -m detector.datagen.tts_dataset --out DATA/selfgen --providers microsoft

  # Add Gemini (needs GEMINI_API_KEY in the environment / Colab Secrets; never put keys in code)
  export GEMINI_API_KEY=...
  python -m detector.datagen.tts_dataset --out DATA/selfgen --providers microsoft gemini

  # Plan only: counts, voices, splits and the estimated Gemini cost, nothing generated
  python -m detector.datagen.tts_dataset --out DATA/selfgen --providers microsoft gemini --dry-run

Splits (fixed by --seed, written to manifest.csv):
  * voices: a few voices per provider are TEST-ONLY (never in train/dev), so a model can be
    tested on voices it has never heard;
  * sentences: every sentence belongs to exactly one of train/dev/test, so no text is shared
    across splits.
Texts: built-in Hindi and English templates (everyday, WhatsApp-style, numbers, names, money
and scam-like requests), or your own files with --texts-hi / --texts-en (one sentence per line,
e.g. transcripts of the matched real corpora). Real speech is NOT generated here; pair this
data with real Hindi/English recordings of similar amount (Kathbath, IndicVoices, M-AILABS,
IndieFake real, your own recordings) before training.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import io
import itertools
import os
import random
import sys
import time
import wave
from dataclasses import dataclass
from pathlib import Path

GEMINI_VOICES = [  # prebuilt Gemini TTS voices
    "Zephyr", "Puck", "Charon", "Kore", "Fenrir", "Leda", "Orus", "Aoede", "Callirrhoe", "Autonoe",
    "Enceladus", "Iapetus", "Umbriel", "Algieba", "Despina", "Erinome", "Algenib", "Rasalgethi",
    "Laomedeia", "Achernar", "Alnilam", "Schedar", "Gacrux", "Pulcherrima", "Achird",
    "Zubenelgenubi", "Vindemiatrix", "Sadachbia", "Sadaltager", "Sulafat",
]
MS_LOCALES = ("hi-IN", "en-IN", "en-US", "en-GB")
FIELDS = ["clip_id", "provider", "model", "voice", "voice_split", "language", "text_id", "text_split",
          "split", "text", "style", "path", "format", "sample_rate", "duration_s", "created", "status"]

# ---------------------------------------------------------------- built-in texts
HI_NAMES = ["राहुल", "प्रिया", "अमित", "नेहा", "सुनीता", "विकास", "अंकित", "पूजा", "रोहन", "कविता", "अर्जुन", "मीना"]
HI_PLACES = ["बाज़ार", "स्टेशन", "ऑफिस", "कॉलेज", "अस्पताल", "बैंक", "मंदिर", "एयरपोर्ट", "स्कूल", "मॉल"]
HI_DAYS = ["सोमवार", "मंगलवार", "बुधवार", "गुरुवार", "शुक्रवार", "शनिवार", "रविवार"]
HI_TEMPLATES = [
    "{name}, कृपया मुझे {hour} बजे फ़ोन करना।",
    "क्या तुम {place} पहुँच गए हो? मैं बाहर इंतज़ार कर रहा हूँ।",
    "आज {place} में बहुत भीड़ है, थोड़ी देर से आना।",
    "मेरा नया नंबर {digits} है, इसे सेव कर लो।",
    "{amount} रुपये अभी इस नंबर पर भेज दो, बहुत ज़रूरी है।",
    "कल की मीटिंग {hour} बजे होगी, समय पर आ जाना।",
    "{name} का जन्मदिन {day} को है, केक मैं ले आऊँगा।",
    "बारिश की वजह से {place} वाली सड़क बंद है।",
    "डॉक्टर ने कहा है कि दवा दिन में {n} बार लेनी है।",
    "हैलो, मैं बैंक से बोल रहा हूँ, आपका खाता बंद होने वाला है।",
    "माँ, मेरा फ़ोन खो गया है, इस नंबर पर {amount} रुपये भेज दो।",
    "{name}, तुम्हारा पार्सल {place} पर आ गया है।",
    "ट्रेन {minutes} मिनट लेट है, मैं {hour} बजे तक पहुँचूँगा।",
    "आज शाम को {place} चलें? {name} भी आ रहा है।",
    "आपका ओटीपी {otp} है, इसे किसी के साथ साझा न करें।",
    "{day} को छुट्टी है, हम सब {place} घूमने जाएँगे।",
    "भाई, {amount} रुपये उधार दे दो, {day} तक लौटा दूँगा।",
    "मैंने {name} को {place} के पास देखा था।",
    "कृपया ध्यान दें, गाड़ी संख्या {digits4} प्लेटफ़ॉर्म नंबर {n} पर आ रही है।",
    "सुनो, आज खाना मत बनाना, हम बाहर खाएँगे।",
]
EN_NAMES = ["Rahul", "Priya", "Amit", "Neha", "Sunita", "Vikas", "Ankit", "Pooja", "John", "Sarah", "David", "Emma"]
EN_PLACES = ["the market", "the station", "the office", "college", "the hospital", "the bank", "the airport", "school", "the mall", "the temple"]
EN_DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
EN_TEMPLATES = [
    "{name}, please call me back at {hour} o'clock.",
    "Have you reached {place}? I'm waiting outside.",
    "There's a huge crowd at {place} today, come a little late.",
    "My new number is {digits}, please save it.",
    "Please send {amount} rupees to this number right now, it's urgent.",
    "Tomorrow's meeting is at {hour}, don't be late.",
    "{name}'s birthday is on {day}, I'll bring the cake.",
    "The road near {place} is closed because of the rain.",
    "The doctor said to take the medicine {n} times a day.",
    "Hello, I'm calling from your bank, your account will be blocked today.",
    "Mom, I lost my phone, please send {amount} rupees to this number.",
    "{name}, your parcel has arrived at {place}.",
    "The train is {minutes} minutes late, I'll reach by {hour}.",
    "Shall we go to {place} this evening? {name} is coming too.",
    "Your one time password is {otp}, do not share it with anyone.",
    "We have a holiday on {day}, let's all visit {place}.",
    "Can you lend me {amount} rupees? I'll return it by {day}.",
    "I saw {name} near {place} yesterday.",
    "Attention please, train number {digits4} is arriving on platform {n}.",
    "Listen, don't cook today, we'll eat out.",
]


def builtin_sentences(lang: str, n: int, rng: random.Random) -> list[str]:
    names, places, days, templates = ((HI_NAMES, HI_PLACES, HI_DAYS, HI_TEMPLATES) if lang == "hi"
                                      else (EN_NAMES, EN_PLACES, EN_DAYS, EN_TEMPLATES))
    out, seen = [], set()
    for _ in range(n * 20):
        if len(out) >= n:
            break
        slots = dict(name=rng.choice(names), place=rng.choice(places), day=rng.choice(days),
                     hour=rng.randint(1, 12), n=rng.randint(2, 4), minutes=rng.choice([10, 15, 20, 30, 45]),
                     amount=rng.choice([500, 1000, 2000, 5000, 10000, 15000, 25000]),
                     digits=f"{rng.randint(6, 9)}{rng.randint(0, 999_999_999):09d}", digits4=f"{rng.randint(1000, 9999)}",
                     otp=f"{rng.randint(0, 999_999):06d}")
        k = rng.choice([1, 1, 2])  # some two-sentence clips for 6-15 s of speech
        text = " ".join(t.format(**slots) for t in rng.sample(templates, k))
        if text not in seen:
            seen.add(text)
            out.append(text)
    return out


def load_texts(path: str | None, lang: str, n: int, rng: random.Random) -> list[tuple[str, str]]:
    if path:
        lines = [l.strip() for l in Path(path).read_text(encoding="utf-8").splitlines() if 3 <= len(l.strip()) <= 300]
        rng.shuffle(lines)
        texts = list(dict.fromkeys(lines))[:n]
    else:
        texts = builtin_sentences(lang, n, rng)
    return [(f"{lang}-{hashlib.sha1(t.encode()).hexdigest()[:10]}", t) for t in texts]


def text_split(text_id: str, dev=0.1, test=0.15) -> str:
    h = int(hashlib.sha1(text_id.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "test" if h < test else ("dev" if h < test + dev else "train")


@dataclass
class Job:
    provider: str
    model: str
    voice: str
    voice_split: str
    language: str
    text_id: str
    text: str
    style: str
    path: Path

    @property
    def split(self):
        # test-only voices are test whatever the sentence; otherwise the sentence decides
        return "test" if self.voice_split == "test" else text_split(self.text_id)


# ---------------------------------------------------------------- planning
def ms_voices() -> list[dict]:
    import edge_tts
    return [v for v in asyncio.run(edge_tts.list_voices()) if v["Locale"] in MS_LOCALES]


def assign_voice_splits(voices: list[str], n_test: int, seed: int) -> dict[str, str]:
    order = sorted(voices)
    random.Random(seed).shuffle(order)
    return {v: ("test" if i < n_test else "trainable") for i, v in enumerate(order)}


def plan(args) -> list[Job]:
    rng = random.Random(args.seed)
    texts = {"hi": load_texts(args.texts_hi, "hi", args.sentences, rng),
             "en": load_texts(args.texts_en, "en", args.sentences, rng)}
    jobs = []
    if "microsoft" in args.providers:
        voices = ms_voices()
        multi = [v["ShortName"] for v in voices if "Multilingual" in v["ShortName"]]
        by_lang = {"hi": [v["ShortName"] for v in voices if v["Locale"] == "hi-IN"] + multi,  # multilingual voices speak Hindi
                   "en": [v["ShortName"] for v in voices if v["Locale"] in ("en-IN", "en-US", "en-GB")]}
        # Test-only voices: one multilingual (so Hindi also has an unseen voice) and one English-only voice.
        order = sorted(by_lang["en"])
        random.Random(args.seed).shuffle(order)
        held = {next(v for v in order if v in multi), next(v for v in order if v not in multi)}
        splits = {v: ("test" if v in held else "trainable") for v in set(by_lang["hi"]) | set(by_lang["en"])}
        for lang, n in (("hi", args.ms_per_lang), ("en", args.ms_per_lang)):
            pool = by_lang[lang]
            for i, (tid, text) in enumerate(itertools.islice(itertools.cycle(texts[lang]), n)):
                voice = pool[i % len(pool)]
                rate, pitch = rng.choice(["-10%", "-5%", "+0%", "+5%", "+10%"]), rng.choice(["-5Hz", "+0Hz", "+5Hz"])
                cid = hashlib.sha1(f"ms|{voice}|{tid}|{rate}|{pitch}".encode()).hexdigest()[:16]
                jobs.append(Job("microsoft", "edge-tts", voice, splits[voice], lang, tid, text, f"rate{rate} pitch{pitch}",
                                Path(args.out) / "microsoft" / voice / f"{cid}.mp3"))
    if "gemini" in args.providers:
        splits = assign_voice_splits(GEMINI_VOICES, 4, args.seed)
        model = args.gemini_model or "auto"
        for lang in ("hi", "en"):
            for i, (tid, text) in enumerate(itertools.islice(itertools.cycle(texts[lang]), args.gemini_per_lang)):
                voice = GEMINI_VOICES[(i * 7 + (lang == "hi")) % len(GEMINI_VOICES)]
                cid = hashlib.sha1(f"gm|{voice}|{tid}".encode()).hexdigest()[:16]
                jobs.append(Job("gemini", model, voice, splits[voice], lang, tid, text, "default",
                                Path(args.out) / "gemini" / voice / f"{cid}.wav"))
    return jobs


# ---------------------------------------------------------------- manifest
def done_ids(manifest: Path) -> set[str]:
    if not manifest.exists():
        return set()
    with manifest.open(encoding="utf-8") as f:
        return {r["clip_id"] for r in csv.DictReader(f) if r["status"] == "ok"}


def append_row(manifest: Path, row: dict):
    new = not manifest.exists()
    with manifest.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def row_for(job: Job, fmt: str, sr: int, dur: float, status="ok") -> dict:
    return dict(clip_id=job.path.stem, provider=job.provider, model=job.model, voice=job.voice, voice_split=job.voice_split,
                language=job.language, text_id=job.text_id, text_split=text_split(job.text_id), split=job.split,
                text=job.text, style=job.style, path=str(job.path), format=fmt, sample_rate=sr,
                duration_s=round(dur, 3), created=time.strftime("%Y-%m-%d %H:%M:%S"), status=status)


def probe(path: Path) -> tuple[int, float]:
    import soundfile as sf
    try:
        info = sf.info(str(path))
        return info.samplerate, info.frames / info.samplerate
    except Exception:  # MP3 support depends on the libsndfile build; fall back to ffprobe
        import subprocess
        out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=sample_rate:format=duration",
                              "-of", "default=nw=1:nk=1", str(path)], capture_output=True, text=True).stdout.split()
        return int(out[0]), float(out[1])


# ---------------------------------------------------------------- Microsoft (edge-tts)
async def run_microsoft(jobs: list[Job], manifest: Path, concurrency: int):
    import edge_tts
    sem = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()
    done = 0

    async def one(job: Job):
        nonlocal done
        rate, pitch = job.style.split()[0][4:], job.style.split()[1][5:]
        job.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = job.path.with_suffix(".part")
        async with sem:
            for attempt in range(5):
                try:
                    await edge_tts.Communicate(job.text, job.voice, rate=rate, pitch=pitch).save(str(tmp))
                    tmp.rename(job.path)
                    break
                except Exception as exc:  # network hiccups: back off and retry
                    if attempt == 4:
                        print(f"  failed {job.voice}: {exc}", file=sys.stderr)
                        return
                    await asyncio.sleep(2 ** attempt)
        sr, dur = probe(job.path)
        async with lock:
            append_row(manifest, row_for(job, "mp3", sr, dur))
            done += 1
            if done % 100 == 0:
                print(f"  microsoft: {done}/{len(jobs)}", flush=True)

    await asyncio.gather(*(one(j) for j in jobs))


# ---------------------------------------------------------------- Gemini
def gemini_model_name(client, wanted: str) -> str:
    if wanted and wanted != "auto":
        return wanted
    names = [m.name.split("/")[-1] for m in client.models.list() if "tts" in m.name.lower()]
    if not names:
        sys.exit("No Gemini TTS model is available to this API key.")
    flash = sorted((n for n in names if "flash" in n), reverse=True)
    return (flash or sorted(names, reverse=True))[0]


def run_gemini(jobs: list[Job], manifest: Path, rpm: float, model_wanted: str):
    from google import genai
    from google.genai import types
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        sys.exit("Set GEMINI_API_KEY (Colab: Secrets panel, then os.environ['GEMINI_API_KEY'] = userdata.get(...)).")
    client = genai.Client(api_key=key)
    model = gemini_model_name(client, model_wanted)
    print(f"  gemini model: {model}")
    gap = 60.0 / rpm
    for i, job in enumerate(jobs, 1):
        t0 = time.monotonic()
        job.path.parent.mkdir(parents=True, exist_ok=True)
        prompt = job.text if job.language == "en" else f"Say in natural Hindi: {job.text}"
        for attempt in range(8):
            try:
                resp = client.models.generate_content(
                    model=model, contents=prompt,
                    config=types.GenerateContentConfig(
                        response_modalities=["AUDIO"],
                        speech_config=types.SpeechConfig(voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=job.voice)))))
                pcm = resp.candidates[0].content.parts[0].inline_data.data
                break
            except Exception as exc:
                msg = str(exc)
                if "RESOURCE_EXHAUSTED" in msg or "429" in msg:
                    if "per day" in msg.lower() or "daily" in msg.lower():
                        print("  Gemini daily quota reached; rerun tomorrow (or enable billing) to continue.")
                        return
                    time.sleep(min(300, 15 * 2 ** attempt))
                    continue
                if attempt >= 3:
                    print(f"  failed {job.voice}: {msg[:160]}", file=sys.stderr)
                    pcm = None
                    break
                time.sleep(5 * 2 ** attempt)
        if not pcm:
            continue
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:  # Gemini returns 16-bit mono PCM at 24 kHz
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(24000)
            w.writeframes(pcm)
        tmp = job.path.with_suffix(".part")
        tmp.write_bytes(buf.getvalue())
        tmp.rename(job.path)
        job.model = model
        append_row(manifest, row_for(job, "wav", 24000, len(pcm) / 2 / 24000))
        if i % 50 == 0:
            print(f"  gemini: {i}/{len(jobs)}", flush=True)
        time.sleep(max(0.0, gap - (time.monotonic() - t0)))


# ---------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, help="output folder (e.g. on Google Drive or the HDD)")
    p.add_argument("--providers", nargs="+", choices=["microsoft", "gemini"], default=["microsoft"])
    p.add_argument("--sentences", type=int, default=2000, help="unique sentences per language")
    p.add_argument("--ms-per-lang", type=int, default=1250, help="Microsoft clips per language")
    p.add_argument("--gemini-per-lang", type=int, default=750, help="Gemini clips per language")
    p.add_argument("--gemini-model", default="auto", help="Gemini TTS model id; 'auto' picks the newest flash TTS model")
    p.add_argument("--gemini-rpm", type=float, default=8, help="Gemini requests per minute (free tier is low)")
    p.add_argument("--concurrency", type=int, default=4, help="parallel Microsoft requests")
    p.add_argument("--texts-hi", help="optional file: one Hindi sentence per line")
    p.add_argument("--texts-en", help="optional file: one English sentence per line")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / "manifest.csv"
    jobs = plan(args)
    have = done_ids(manifest)
    todo = [j for j in jobs if j.path.stem not in have]
    by = lambda prov, key: sorted({getattr(j, key) for j in jobs if j.provider == prov})
    for prov in args.providers:
        pj = [j for j in jobs if j.provider == prov]
        test_voices = sorted({j.voice for j in pj if j.voice_split == "test"})
        print(f"{prov}: {len(pj)} clips ({sum(j.path.stem in have for j in pj)} done), {len(by(prov, 'voice'))} voices, "
              f"test-only voices: {test_voices}")
        print(f"   splits: " + ", ".join(f"{s} {sum(j.split == s for j in pj)}" for s in ("train", "dev", "test")))
    gem = [j for j in todo if j.provider == "gemini"]
    if gem:
        secs = sum(len(j.text) for j in gem) / 14.0          # ~14 characters per second of speech
        tokens = secs * 32                                    # audio tokens, conservative rate
        print(f"Gemini estimate: ~{secs / 3600:.1f} h of audio, ~{tokens / 1e6:.2f} M audio tokens -> about "
              f"${tokens / 1e6 * 20:.2f} at $20 / M output tokens (check the current price); "
              f"free tier: ~{len(gem) / args.gemini_rpm / 60:.1f} h at {args.gemini_rpm} requests/min, within daily caps")
    if args.dry_run:
        return
    ms = [j for j in todo if j.provider == "microsoft"]
    if ms:
        print(f"Generating {len(ms)} Microsoft clips ...")
        asyncio.run(run_microsoft(ms, manifest, args.concurrency))
    if gem:
        print(f"Generating {len(gem)} Gemini clips ...")
        run_gemini(gem, manifest, args.gemini_rpm, args.gemini_model)
    print(f"Done. Manifest: {manifest} ({len(done_ids(manifest))} clips ok). Rerun the same command to resume.")


if __name__ == "__main__":
    main()

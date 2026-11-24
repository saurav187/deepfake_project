"""Owner-invoked synthesis. Supply consent-reviewed development jobs as CSV."""

import argparse
import asyncio
import csv
import hashlib
import json
import os
import subprocess
import tempfile
import wave
from pathlib import Path

from .data import safe_path


def generated_path(job):
    key = hashlib.sha256(json.dumps(job, sort_keys=True).encode()).hexdigest()[:12]
    ext = ".mp3" if job["engine"] == "edge" else ".wav"
    return Path("generated") / (job["id"] + "-" + key + ext)


def run_job(job, path, root):
    engine, text, voice, lang = [
        job[k] for k in ("engine", "text", "voice", "language")
    ]
    if engine == "edge":
        import edge_tts

        asyncio.run(edge_tts.Communicate(text, voice).save(str(path)))
    elif engine == "azure":
        from azure.cognitiveservices import speech

        config = speech.SpeechConfig(
            subscription=os.environ["SPEECH_KEY"], region=os.environ["SPEECH_REGION"]
        )
        config.speech_synthesis_voice_name = voice
        config.set_speech_synthesis_output_format(
            speech.SpeechSynthesisOutputFormat.Riff24Khz16BitMonoPcm
        )
        result = (
            speech.SpeechSynthesizer(speech_config=config, audio_config=None)
            .speak_text_async(text)
            .get()
        )
        if result.reason != speech.ResultReason.SynthesizingAudioCompleted:
            raise RuntimeError("Azure synthesis failed")
        path.write_bytes(result.audio_data)
    elif engine == "gemini":
        from google import genai
        from google.genai import types

        result = genai.Client(
            api_key=os.environ["GEMINI_API_KEY"]
        ).models.generate_content(
            model=job["model"],
            contents=text,
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name=voice
                        )
                    )
                ),
            ),
        )
        pcm = result.candidates[0].content.parts[0].inline_data
        if not ("pcm" in pcm.mime_type.lower() or "l16" in pcm.mime_type.lower()):
            raise ValueError("Unexpected Gemini audio format")
        with wave.open(str(path), "wb") as w:
            w.setparams((1, 2, 24000, 0, "NONE", ""))
            w.writeframes(pcm.data)
    elif engine == "xtts":
        subprocess.run(
            [
                "tts",
                "--model_name",
                "tts_models/multilingual/multi-dataset/xtts_v2",
                "--speaker_wav",
                str(safe_path(root, job["reference"])),
                "--language_idx",
                lang,
                "--text",
                text,
                "--out_path",
                str(path),
            ],
            check=True,
        )
    elif engine == "chatterbox":
        import torch
        import torchaudio
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS

        model = ChatterboxMultilingualTTS.from_pretrained(
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        wav = model.generate(
            text,
            language_id=lang,
            audio_prompt_path=str(safe_path(root, job["reference"])),
        )
        torchaudio.save(str(path), wav.cpu(), model.sr)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--jobs", required=True)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = p.parse_args()
    root = args.data / "development"
    with safe_path(root, args.jobs).open() as f:
        jobs = list(csv.DictReader(f))
    totals = {}
    unique = set()
    pending = []
    all_characters = {}
    for j in jobs:
        if (
            j["engine"] not in {"azure", "gemini", "edge", "xtts", "chatterbox"}
            or not j["text"].strip()
        ):
            raise ValueError("Invalid synthesis job")
        if j["engine"] in {"xtts", "chatterbox"} and (
            not j["consent_id"].strip() or not safe_path(root, j["reference"]).is_file()
        ):
            raise ValueError("Need consent and a development reference")
        if j["id"] in unique or not j["id"].replace("-", "").isalnum():
            raise ValueError("Invalid or repeated job id")
        unique.add(j["id"])
        all_characters[j["engine"]] = all_characters.get(j["engine"], 0) + len(
            j["text"]
        )
        path = root / generated_path(j)
        if path.is_file() and path.stat().st_size:
            continue
        pending.append(j)
        totals[j["engine"]] = totals.get(j["engine"], 0) + len(j["text"])
    seconds = sum(float(j["seconds"]) for j in pending if j["engine"] == "gemini")
    print(
        json.dumps(
            {
                "jobs": len(jobs),
                "pending_jobs": len(pending),
                "pending_characters": totals,
                "characters": all_characters,
                "gemini_output_usd_at_20_per_million": seconds * 0.0005,
                "gemini_input_usd_estimate": totals.get("gemini", 0) / 2 / 1e6,
                "local_gpu_hours_estimate": [0.5, 2]
                if set(totals) & {"xtts", "chatterbox"}
                else [0, 0],
            }
        )
    )
    if args.dry_run:
        return
    out = root / "generated"
    out.mkdir(exist_ok=True)
    for j in pending:
        path = root / generated_path(j)
        with tempfile.TemporaryDirectory(dir=out) as stage:
            tmp = Path(stage) / path.name
            run_job(j, tmp, root)
            if not tmp.stat().st_size:
                raise ValueError("Empty generated recording")
            tmp.replace(path)


if __name__ == "__main__":
    main()

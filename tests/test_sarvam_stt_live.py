"""
Live Sarvam STT diagnostic.

Normal unit test runs skip this file. To run a real provider check:

    $env:RUN_SARVAM_LIVE_TEST="1"
    $env:SARVAM_LIVE_AUDIO="C:\\path\\to\\short.wav"
    python -m unittest tests.test_sarvam_stt_live

The script prints only safe key diagnostics and does not touch Qdrant, Gemini,
or the /voice RAG pipeline.
"""

import os
import unittest
from pathlib import Path

from app.voice import stt


class LiveSarvamSTTTest(unittest.TestCase):
    @unittest.skipUnless(
        os.getenv("RUN_SARVAM_LIVE_TEST") == "1",
        "set RUN_SARVAM_LIVE_TEST=1 to call the live Sarvam STT API",
    )
    def test_live_sarvam_stt(self):
        status = stt.stt_status()
        print(f"Provider: {status['provider']}")
        print(f"Configured: {status['configured']}")
        print(f"Key length: {status['key_length']}")
        print(f"Key prefix: {status['key_prefix']}")
        print(f"Endpoint: {status['endpoint']}")
        print(f"Model: {status['model']}")
        print(f"Mode: {status['mode']}")

        self.assertTrue(status["configured"], "SARVAM_API_KEY is missing")

        audio_path = os.getenv("SARVAM_LIVE_AUDIO")
        self.assertTrue(audio_path, "SARVAM_LIVE_AUDIO must point to a short audio file")

        path = Path(audio_path)
        self.assertTrue(path.exists(), f"audio file not found: {path}")

        try:
            result = stt.transcribe_audio(path.read_bytes(), os.getenv("SARVAM_LIVE_LANGUAGE", "hi-IN"))
        except stt.STTError as exc:
            print(f"HTTP: {exc.provider_status_code}")
            print(f"Error category: {exc.category}")
            print(f"Provider error code: {exc.provider_error_code}")
            print(f"Provider message: {exc.provider_message}")
            raise

        print("HTTP: 200")
        print(f"Transcript: {result['transcript']}")
        print(f"Language: {result['language_code']}")
        print(f"Latency ms: {result['latency_ms']}")

        self.assertEqual(result["provider"], "sarvam")
        self.assertTrue(result["transcript"].strip())


if __name__ == "__main__":
    unittest.main()

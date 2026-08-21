"""
Live /voice endpoint diagnostic.

This is skipped unless RUN_VOICE_LIVE_TEST=1. For local SQLite-backed Qdrant,
set QDRANT_PATH to an unlocked copy of data/qdrant if another server is running.
"""

import os
import json
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from app import api as api_module

app = api_module.app


class LiveVoiceEndpointTest(unittest.TestCase):
    @unittest.skipUnless(
        os.getenv("RUN_VOICE_LIVE_TEST") == "1",
        "set RUN_VOICE_LIVE_TEST=1 to call /voice with real Sarvam STT",
    )
    def test_live_voice_endpoint(self):
        audio_path = os.getenv("SARVAM_LIVE_AUDIO")
        self.assertTrue(audio_path, "SARVAM_LIVE_AUDIO must point to a short audio file")

        path = Path(audio_path)
        self.assertTrue(path.exists(), f"audio file not found: {path}")

        with TestClient(app) as client:
            response = client.post(
                "/voice",
                files={"file": (path.name, path.read_bytes(), "audio/wav")},
                data={"language": os.getenv("SARVAM_LIVE_LANGUAGE", "en-IN")},
            )

        print(f"POST /voice HTTP: {response.status_code}")
        print(json.dumps(response.json(), ensure_ascii=True)[:1000])
        self.assertEqual(response.status_code, 200)

        payload = response.json()
        print(f"Transcript: {payload['transcript']}")
        print(f"STT provider: {payload['stt_provider']}")
        print(f"STT latency ms: {payload['stt_latency_ms']}")
        print(f"Retrieved chunks: {payload['retrieved_chunks']}")

        self.assertEqual(payload["stt_provider"], "sarvam")
        self.assertTrue(payload["transcript"].strip())
        self.assertIn("answer", payload)
        self.assertIn("rag_timings", payload)


if __name__ == "__main__":
    unittest.main()

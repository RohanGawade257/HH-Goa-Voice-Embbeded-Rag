import sys
sys.stdout.reconfigure(encoding="utf-8")

print("Testing API imports...")
try:
    from app import api
    print("app/api.py: OK")
except Exception as e:
    print(f"app/api.py: FAILED - {e}")
    import traceback; traceback.print_exc()

print()
print("Testing STT imports...")
try:
    from app.voice import stt
    status = stt.stt_status()
    print("app/voice/stt.py: OK")
    provider = status["provider"]
    configured = status["configured"]
    print(f"  Provider: {provider}")
    print(f"  Configured: {configured}")
    result = stt.transcribe_audio(b"fake")
    transcript = result["transcript"]
    mock_provider = result["provider"]
    print(f"  Mock transcript: {transcript}")
    print(f"  Mock provider: {mock_provider}")
except Exception as e:
    print(f"app/voice/stt.py: FAILED - {e}")
    import traceback; traceback.print_exc()

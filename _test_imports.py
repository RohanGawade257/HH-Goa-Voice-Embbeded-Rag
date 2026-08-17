import sys
sys.stdout.reconfigure(encoding="utf-8")

print("Testing API imports...")
try:
    import api
    print("api.py: OK")
except Exception as e:
    print(f"api.py: FAILED - {e}")
    import traceback; traceback.print_exc()

print()
print("Testing STT imports...")
try:
    import stt
    status = stt.stt_status()
    print("stt.py: OK")
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
    print(f"stt.py: FAILED - {e}")
    import traceback; traceback.print_exc()

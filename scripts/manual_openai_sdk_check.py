"""Manual smoke check: transcribe one local file through the OpenAI SDK
against a running server. Not part of the test suite.

    pip install openai
    python scripts/manual_openai_sdk_check.py path/to/audio.wav \
        [--base-url http://localhost:8000/v1] [--api-key KEY]

The API key defaults to $WHISPER_API_KEY; a server with any admin key
configured rejects an empty or dummy key with 401 (open mode only applies
before the first key exists, and only from an allowlisted host).
"""
import argparse
import os
import sys
import time

from openai import OpenAI


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("audio_path")
    ap.add_argument("--base-url", default=os.environ.get("WHISPER_BASE_URL", "http://localhost:8000/v1"))
    ap.add_argument("--api-key", default=os.environ.get("WHISPER_API_KEY", "not-needed"))
    args = ap.parse_args()

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    print(f"Starting SpeechPulse compatibility test: {args.audio_path}")
    print("-" * 50)

    start_time = time.time()
    ok = False
    try:
        with open(args.audio_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="verbose_json",
                timestamp_granularities=["word"],
            )

        if hasattr(transcription, "words"):
            print("\n✅ SUCCESS: 'words' attribute found in response!")

            if transcription.words is None:
                print("❌ BUT: 'words' is None! (API returned null instead of [])")
            else:
                ok = True
                word_count = len(transcription.words)
                print(f"📊 Received {word_count} words.")

                if word_count > 0:
                    print("\nFirst 5 words:")
                    for w in transcription.words[:5]:
                        print(f"  - '{w.word}' ({w.start:.2f}s - {w.end:.2f}s)")
        else:
            print("\n❌ FAILURE: 'words' attribute MISSING from response!")
            print("This will cause SpeechPulse to crash.")

        print("\n📝 Full Text:")
        print(transcription.text)

    except Exception as e:
        print(f"\n❌ ERROR during request: {e}")

    elapsed_time = time.time() - start_time
    print("-" * 50)
    print(f"⏱️  Processing time: {elapsed_time:.2f} seconds")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

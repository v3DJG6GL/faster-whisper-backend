import time

from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

audio_path = r"Z:\SpeechPulse_Test\2025-12-18T17_55_36.wav"

print(f"Starting SpeechPulse compatibility test: {audio_path}")
print("-" * 50)

start_time = time.time()

try:
    with open(audio_path, "rb") as audio_file:
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

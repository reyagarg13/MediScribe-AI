import requests

# Step 1: Upload audio → transcription
transcribe_url = "http://127.0.0.1:8000/transcribe"
file_path = "patient_statement.wav"   # your test file

with open(file_path, "rb") as f:
    files = {"file": f}
    transcribe_response = requests.post(transcribe_url, files=files)

transcription = transcribe_response.json().get("transcription")
print("Transcript:", transcription)

# Step 2: Send transcript → summarizer
summarize_url = "http://127.0.0.1:8000/summarize"
data = {"transcription": transcription}

summary_response = requests.post(summarize_url, json=data)
summary = summary_response.json().get("summary")
print("Summary:", summary)

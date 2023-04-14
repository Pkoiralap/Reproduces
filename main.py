from pyannote.audio import Pipeline

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization@2.1",
    use_auth_token="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"
)

diarization = pipeline("test.mp3", num_speakers=2)
for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(turn, speaker)

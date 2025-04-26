import sounddevice as sd
import numpy as np  
import rich.console as Console
import whisper
from queue import Queue
import threading
import time
from tts import TextToSpeechService

console = Console()
stt = whisper.load_model("base.en")
tts = TextToSpeechService()


def process_audio():
    data_queue = Queue()  # type: ignore[var-annotated]
    stop_event = threading.Event()
    recording_thread = threading.Thread(
        target=record_audio,
        args=(stop_event, data_queue),
    )
    recording_thread.start()
    input()
    stop_event.set()
    recording_thread.join()
    audio_data = b"".join(list(data_queue.queue))
    audio_np = (
        np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
    )

    if audio_np.size > 0:
        with console.status("Transcribing...", spinner="earth"):
            text = transcribe(audio_np)
        console.print(f"[yellow]You: {text}")
        with console.status("Generating response...", spinner="earth"):
            response = get_llm_response(text)
            sample_rate, audio_array = tts.long_form_synthesize(response)
        console.print(f"[cyan]Assistant: {response}")
        play_audio(sample_rate, audio_array)
    else:
        console.print(
            "[red]No audio recorded. Please ensure your microphone is working."
        )
    
    return audio_np, text, response, sample_rate, audio_array



# ---- Function to Speech input ---- #
def record_audio(stop_event, data_queue):
    """
    Captures audio data from the user's microphone and adds it to a queue for further processing.

    Args:
        stop_event (threading.Event): An event that, when set, signals the function to stop recording.
        data_queue (queue.Queue): A queue to which the recorded audio data will be added.

    Returns:
        None
    """
    def callback(indata, frames, time, status):
        if status:
            console.print(status)
        data_queue.put(bytes(indata))

    with sd.RawInputStream(
        samplerate=16000, dtype="int16", channels=1, callback=callback
    ):
        while not stop_event.is_set():
            time.sleep(0.1)
    
    return data_queue

# ---- Function to Transcribe Audio ---- #
def transcribe(audio_np: np.ndarray) -> str:
    """
    Transcribes the given audio data using the Whisper speech recognition model.

    Args:
        audio_np (numpy.ndarray): The audio data to be transcribed.

    Returns:
        str: The transcribed text.
    """
    result = stt.transcribe(audio_np, fp16=True)  # Set fp16=True if using a GPU
    text = result["text"].strip()
    return text

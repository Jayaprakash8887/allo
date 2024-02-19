import json
import os
import time

import requests
from pydub import AudioSegment

from config_util import get_config_value
from utils import *


asr_mapping = {
    "bn": "ai4bharat/conformer-multilingual-indo_aryan-gpu--t4",
    "en": "ai4bharat/whisper-medium-en--gpu--t4",
    "gu": "ai4bharat/conformer-multilingual-indo_aryan-gpu--t4",
    "hi": "ai4bharat/conformer-hi-gpu--t4",
    "kn": "ai4bharat/conformer-multilingual-dravidian-gpu--t4",
    "ml": "ai4bharat/conformer-multilingual-dravidian-gpu--t4",
    "mr": "ai4bharat/conformer-multilingual-indo_aryan-gpu--t4",
    "or": "ai4bharat/conformer-multilingual-indo_aryan-gpu--t4",
    "pa": "ai4bharat/conformer-multilingual-indo_aryan-gpu--t4",
    "sa": "ai4bharat/conformer-multilingual-indo_aryan-gpu--t4",
    "ta": "ai4bharat/conformer-multilingual-dravidian-gpu--t4",
    "te": "ai4bharat/conformer-multilingual-dravidian-gpu--t4",
    "ur": "ai4bharat/conformer-multilingual-indo_aryan-gpu--t4"
}

translation_serviceId = "ai4bharat/indictrans-v2-all-gpu--t4"

tts_mapping = {
    "as": "ai4bharat/indic-tts-coqui-indo_aryan-gpu--t4",
    "bn": "ai4bharat/indic-tts-coqui-indo_aryan-gpu--t4",
    "brx": "ai4bharat/indic-tts-coqui-misc-gpu--t4",
    "en": "ai4bharat/indic-tts-coqui-misc-gpu--t4",
    "gu": "ai4bharat/indic-tts-coqui-indo_aryan-gpu--t4",
    "hi": "ai4bharat/indic-tts-coqui-indo_aryan-gpu--t4",
    "kn": "ai4bharat/indic-tts-coqui-dravidian-gpu--t4",
    "ml": "ai4bharat/indic-tts-coqui-dravidian-gpu--t4",
    "mni": "ai4bharat/indic-tts-coqui-misc-gpu--t4",
    "mr": "ai4bharat/indic-tts-coqui-indo_aryan-gpu--t4",
    "or": "ai4bharat/indic-tts-coqui-indo_aryan-gpu--t4",
    "pa": "ai4bharat/indic-tts-coqui-indo_aryan-gpu--t4",
    "raj": "ai4bharat/indic-tts-coqui-indo_aryan-gpu--t4",
    "ta": "ai4bharat/indic-tts-coqui-dravidian-gpu--t4",
    "te": "ai4bharat/indic-tts-coqui-dravidian-gpu--t4"
}


class RequestError(Exception):
    def __init__(self, response):
        self.response = response


def get_encoded_string(audio):
    if is_url(audio):
        local_filename = generate_temp_filename("mp3")
        with requests.get(audio) as r:
            with open(local_filename, 'wb') as f:
                f.write(r.content)
    elif is_base64(audio):
        local_filename = generate_temp_filename("mp3")
        decoded_audio_content = base64.b64decode(audio)
        output_mp3_file = open(local_filename, "wb")
        output_mp3_file.write(decoded_audio_content)
        output_mp3_file.close()
    else:
        local_filename = audio

    output_file = AudioSegment.from_file(local_filename)
    mp3_output_file = output_file.export(local_filename, format="mp3")
    given_audio = AudioSegment.from_file(mp3_output_file)
    given_audio = given_audio.set_frame_rate(16000)
    given_audio = given_audio.set_channels(1)
    tmp_wav_filename = generate_temp_filename("wav")
    given_audio.export(tmp_wav_filename, format="wav", codec="pcm_s16le")
    with open(tmp_wav_filename, "rb") as wav_file:
        wav_file_content = wav_file.read()
    encoded_string = base64.b64encode(wav_file_content)
    encoded_string = str(encoded_string, 'ascii', 'ignore')
    os.remove(local_filename)
    os.remove(tmp_wav_filename)
    return encoded_string, wav_file_content

def speech_to_text(encoded_string, input_language):
    start_time = time.time()
    url = get_config_value('translator', 'BHASHINI_ENDPOINT_URL', None)
    payload = {
        "pipelineTasks": [
            {
                "taskType": "asr",
                "config": {
                    "language": {
                        "sourceLanguage": input_language
                    },
                    "serviceId": asr_mapping[input_language]
                }
            }
        ],
        "inputData": {
            "audio": [
                {
                    "audioContent": encoded_string
                }
            ]
        }
    }
    headers = {
        'Authorization': get_config_value('translator', 'BHASHINI_API_KEY', None),
        'Content-Type': 'application/json'
    }

    try:
        response = requests.request("POST", url, headers=headers, data=json.dumps(payload))
        process_time = time.time() - start_time
        response.raise_for_status()
        text = json.loads(response.text)[
            "pipelineResponse"][0]["output"][0]["source"]
        return text
    except requests.exceptions.RequestException as e:
        process_time = time.time() - start_time
        raise RequestError(e.response) from e

def indic_translation(text, source, destination):
    if source == destination:
        return text
    try:
        start_time = time.time()
        url = get_config_value('translator', 'BHASHINI_ENDPOINT_URL', None)
        payload = {
            "pipelineTasks": [
                {
                    "taskType": "translation",
                    "config": {
                        "language": {
                            "sourceLanguage": source,
                            "targetLanguage": destination
                        },
                        "serviceId": translation_serviceId
                    }
                }
            ],
            "inputData": {
                "input": [
                    {
                        "source": text
                    }
                ]
            }
        }
        headers = {
            'Authorization': get_config_value('translator', 'BHASHINI_API_KEY', None),
            'Content-Type': 'application/json'
        }

        response = requests.request("POST", url, headers=headers, data=json.dumps(payload))
        process_time = time.time() - start_time
        response.raise_for_status()
        indic_text = json.loads(response.text)["pipelineResponse"][0]["output"][0]["target"]
    except requests.exceptions.RequestException as e:
        process_time = time.time() - start_time
        raise RequestError(e.response) from e
        # indic_text = google_translate_text(text, source, destination)
    return indic_text

def text_to_speech(language, text, gender='female'):
    try:
        start_time = time.time()
        url = get_config_value('translator', 'BHASHINI_ENDPOINT_URL', None)
        payload = {
            "pipelineTasks": [
                {
                    "taskType": "tts",
                    "config": {
                        "language": {
                            "sourceLanguage": language
                        },
                        "serviceId": tts_mapping[language],
                        "gender": gender
                    }
                }
            ],
            "inputData": {
                "input": [
                    {
                        "source": text
                    }
                ],
                "audio": [
                    {
                        "audioContent": None
                    }
                ]
            }
        }
        headers = {
            'Authorization': get_config_value('translator', 'BHASHINI_API_KEY', None),
            'Content-Type': 'application/json'
        }
        response = requests.request("POST", url, headers=headers, data=json.dumps(payload))
        process_time = time.time() - start_time
        response.raise_for_status()
        audio_content = response.json()["pipelineResponse"][0]['audio'][0]['audioContent']
        audio_content = base64.b64decode(audio_content)
    except requests.exceptions.RequestException as e:
        process_time = time.time() - start_time
        audio_content = None
    return audio_content


def audio_input_to_text(audio_file, input_language):
    encoded_string, wav_file_content = get_encoded_string(audio_file)
    try:
        indic_text = speech_to_text(encoded_string, input_language)
    except:
        indic_text = None
    return indic_text


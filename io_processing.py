from fastapi import HTTPException
from langchain.tools.base import ToolException

from cloud_storage_oci import give_public_url, upload_file_object
from logger import logger
from translator import *
import time


def process_incoming_voice(file_url: str, input_language: str):
    error_message = None
    try:
        regional_text = audio_input_to_text(file_url, input_language)
        print("\nprocess_incoming_voice:: regional_text:: ", regional_text)
        try:
            english_text = indic_translation(text=regional_text, source=input_language, destination='en')
            print("\nprocess_incoming_voice:: english_text:: ", english_text)
        except Exception as e:
            error_message = "Indic translation to English failed"
            logger.error(f"Exception occurred: {e}", exc_info=True)
            english_text = None
    except Exception as e:
        error_message = "Speech to text conversion API failed"
        logger.error(f"Exception occurred: {e}", exc_info=True)
        regional_text = None
        english_text = None
    return regional_text, english_text, error_message


def process_incoming_text(regional_text, input_language):
    error_message = None
    try:
        english_text = indic_translation(text=regional_text, source=input_language, destination='en')
    except Exception as e:
        error_message = "Indic translation to English failed"
        english_text = None
        logger.error(f"Exception occurred: {e}", exc_info=True)
    return english_text, error_message


def process_outgoing_text(english_text, input_language):
    regional_text = indic_translation(text=english_text, source='en', destination=input_language)
    return regional_text


def process_outgoing_voice(message: str, input_language: str):
    print("process_outgoing_voice:: message:: ", message)
    translated_text = process_outgoing_text(message, input_language)
    print("process_outgoing_voice:: translated_text:: ", translated_text)
    decoded_audio_content = text_to_speech(language=input_language, text=translated_text)
    if decoded_audio_content is not None:
        time_stamp = time.strftime("%Y%m%d-%H%M%S")
        filename = "audio-output-" + time_stamp + ".mp3"
        output_mp3_file = open(filename, "wb")
        output_mp3_file.write(decoded_audio_content)
        print(upload_file_object(output_mp3_file.name))
        return {"audio": give_public_url(output_mp3_file.name), "eng_text": message, "reg_text": translated_text}
    else:
        raise ToolException("The voice conversion tool is not available")


def process_outgoing_voice_manual(message: str, input_language: str):
    print("process_outgoing_voice_manual:: message:: ", message)
    translated_text = process_outgoing_text(message, input_language)
    print("process_outgoing_voice_manual:: translated_text:: ", translated_text)
    decoded_audio_content = text_to_speech(language=input_language, text=translated_text)
    if decoded_audio_content is not None:
        time_stamp = time.strftime("%Y%m%d-%H%M%S")
        filename = "audio-output-" + time_stamp + ".mp3"
        output_mp3_file = open(filename, "wb")
        output_mp3_file.write(decoded_audio_content)
        print(upload_file_object(output_mp3_file.name))
        return give_public_url(output_mp3_file.name), translated_text
    else:
        raise HTTPException(500, "The voice conversion tool is not available")

from collections import defaultdict
import gradio as gr
import json
import os
import pycountry
import requests

DEFAULT_VOICE = "en_US-libritts-high"
REPO_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main/"
VOICES_JSON = "voices.json"
DATA_DIR = '.'
params = {
    "activate": True,
    "selected_voice": "es_ES-sharvard-medium",
}
voices_data = None
voices_by_lang = None
languages = None


def input_modifier(string):
    """
    This function is applied to your text inputs before
    they are fed into the model.
    """
    if not params['activate']:
        return string

    return string


def output_modifier(string):
    """
    This function is applied to the model outputs.
    """
    if not params['activate']:
        return string

    return string


def bot_prefix_modifier(string):
    """
    This function is only applied in chat mode. It modifies
    the prefix text for the Bot and can be used to bias its
    behavior.
    """
    return string


def refresh_voices(force_dl=False):
    voices_data = {}
    file_data = os.path.join(DATA_DIR, VOICES_JSON)
    if os.path.isfile(file_data) and not force_dl:
        # Load cached values
        with open(file_data, 'rt') as f:
            jdata = json.load(f)
    else:
        # Download new values
        result = None
        try:
            result = requests.get(url=REPO_URL + VOICES_JSON)
        except HTTPError as http_err:
            print(f'HTTP error occurred: {http_err}')
        except Exception as err:
            print(f'Other error occurred: {err}')
        if result is None:
            return
        if result.status_code != 200:
            print(f'Wong status {result.status_code}')
            return
        jdata = result.json()
        with open(file_data, 'wt') as f:
            f.write(json.dumps(jdata, indent=2))
    voices_data = jdata
    voices_by_lang = defaultdict(list)
    languages = set()
    total = 0
    for id, data in voices_data.items():
        ln = data['language']
        ln_code_2 = ln[:2].upper()
        co_code_2 = ln[3:]
        # Patch errors
        if co_code_2 == 'UK':
            co_code_2 = 'UA'
        lang = (pycountry.languages.get(alpha_2=ln_code_2).name + ' (' +
                pycountry.countries.get(alpha_2=co_code_2).name + ')')
        voices_by_lang[lang].append(id)
        languages.add(lang)
        total += data['num_speakers']
    languages = sorted(list(languages))
    print(f'Languages: {len(languages)}')
    print(f'Models: {len(voices_data)}')
    print(f'Voices: {total}')


def ui():
    global voices_data
    if not voices_data:
        refresh_voices()
        selected = params['selected_voice']
        if selected == 'None':
            # TODO: select a voice according to the locale language
            params['selected_voice'] = DEFAULT_VOICE
        elif selected not in voices:
            logger.error(f'Selected voice {selected} not available, switching to {DEFAULT_VOICE}')
            params['selected_voice'] = DEFAULT_VOICE

    # Estoy acá hay que armar unos dropdown al estilo de los que hay en https://rhasspy.github.io/piper-samples/
    # Language Voice Quality Speaker
    # Estaría impresionante si hubiese un botón que al elegirlo reproduce el sample del elegido

    # Finding the language name from the language code to use as the default value
    language_name = list(language_codes.keys())[list(language_codes.values()).index(params['language string'])]

    # Gradio elements
    with gr.Row():
        activate = gr.Checkbox(value=params['activate'], label='Activate translation')

    with gr.Row():
        language = gr.Dropdown(value=language_name, choices=[k for k in language_codes], label='Language')

    # Event functions to update the parameters in the backend
    activate.change(lambda x: params.update({"activate": x}), activate, None)
    language.change(lambda x: params.update({"language string": language_codes[x]}), language, None)


refresh_voices()

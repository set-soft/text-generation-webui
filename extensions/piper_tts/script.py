from collections import defaultdict
import hashlib
import gradio as gr
import json
import os
from pathlib import Path
from extensions.piper_tts.piper import Piper
import pycountry
import requests
from modules.logging_colors import logger
from modules import shared
from pprint import pprint

DEFAULT_VOICE = "en_US-libritts-high"
REPO_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main/"
VOICES_JSON = "voices.json"
DATA_DIR = os.path.dirname(__file__)
VOICE_TYPE = ['Primary', 'Secondary']
MODELS_PATH = os.path.join("models", "piper")
# Only one speaker available
ONLY_ONE = "Only one"
params = {
    "activate": True,
    "enable_0": True,
    "selected_voice_0": DEFAULT_VOICE,  # "es_ES-sharvard-medium",
    "speaker_0": '',  # "F",
    "enable_1": True,
    "selected_voice_1": DEFAULT_VOICE,  # "es_ES-sharvard-medium",
    "speaker_1": '',  # "M",
    "autoplay": False,
    "show_text": True,
}
# Data as in the JSON (model_id -> full_data)
voices_data = None
# Classified by language (language -> [model_id ...]
voices_by_lang = None
# language -> {Visible name -> full_data}
name_to_voice = None
# A list containing the available languages, using the visible name
languages = None
wav_idx = 0
cfg0 = None
cfg0_name = ''
cfg1 = None
cfg1_name = ''


def input_modifier(string):
    """
    This function is applied to your text inputs before
    they are fed into the model.
    """
    if not params['activate']:
        return string

    shared.processing_message = "*Is recording a voice message...*"
    return string


def output_modifier(string):
    """
    This function is applied to the model outputs.
    """
    global params, wav_idx

    if not params['activate']:
        return string

    original_string = string
    string = string.replace('"', '')
    string = string.replace('“', '')
    string = string.replace('\n', ' ')
    string = string.strip()
    if string == '':
        string = 'empty reply, try regenerating'

    output_file = Path(f'extensions/piper_tts/outputs/{wav_idx:06d}.wav'.format(wav_idx))
    logger.debug(f'Outputting audio to {str(output_file)}')

    gen_audio_file(string, str(output_file))

    autoplay = 'autoplay' if params['autoplay'] else ''
    string = f'<audio src="file/{output_file.as_posix()}" controls {autoplay}></audio>'
    wav_idx += 1

    if params['show_text']:
        string += f'\n\n{original_string}'

    shared.processing_message = "*Is typing...*"
    return string


def bot_prefix_modifier(string):
    """
    This function is only applied in chat mode. It modifies
    the prefix text for the Bot and can be used to bias its
    behavior.
    """
    return string


def download_model_file(f_name, name, info):
    d_name = os.path.dirname(f_name)
    os.makedirs(d_name, exist_ok=True)
    url = REPO_URL + name
    logger.debug(f'Downloading {url}')
    result = requests.get(url=url)
    size = len(result.content)
    esize = info['size_bytes']
    if size != esize:
        raise ValueError(f'Downloaded size mismatch ({url}) {size} vs {esize}')
    h = hashlib.md5()
    h.update(result.content)
    res_md5 = h.hexdigest()
    e_md5 = info['md5_digest']
    if res_md5 != e_md5:
        raise ValueError(f'Downloaded integrity fail ({url}) {res_md5} vs {e_md5}')
    with open(f_name, 'wb') as f:
        f.write(result.content)


def gen_audio_file(string, out_file):
    # Check if configured
    global cfg0_name, cfg0

    sel = params['selected_voice_0']
    data = voices_data[sel]
    if sel != cfg0_name:
        # Not configured for this voice
        # Check the model
        for name, info in data['files'].items():
            f_name = os.path.join(MODELS_PATH, name)
            if name.endswith('.onnx'):
                model_name = f_name
            if not os.path.isfile(f_name):
                download_model_file(f_name, name, info)
        # Create a Piper object
        cfg0 = Piper(model_name)
        cfg0_name = sel
        # Ensure the output dir exists
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
    sel_speaker = params['speaker_0']
    if sel_speaker:
        speaker_id = data['speaker_id_map'][sel_speaker]
    else:
        speaker_id = None
    wav_bytes = cfg0.synthesize(string, speaker_id=speaker_id)
    with open(out_file, 'wb') as f:
        f.write(wav_bytes)


def lang_code_to_name(ln):
    ln_code_2 = ln[:2].upper()
    co_code_2 = ln[3:]
    # Patch errors
    if co_code_2 == 'UK':
        co_code_2 = 'UA'
    return (pycountry.languages.get(alpha_2=ln_code_2).name + ' (' +
            pycountry.countries.get(alpha_2=co_code_2).name + ')')


def get_voice_name(v):
    return f"{v['name']} ({v['quality']})"


def refresh_voices(force_dl=False):
    global voices_data
    global voices_by_lang
    global languages
    global name_to_voice
    if voices_data is None:
        voices_data = {}
        voices_by_lang = defaultdict(list)
        languages = set()
        name_to_voice = defaultdict(dict)

    file_data = os.path.join(DATA_DIR, VOICES_JSON)
    if os.path.isfile(file_data) and not force_dl:
        # Load cached values
        logger.debug(f"Loading {file_data}")
        with open(file_data, 'rt') as f:
            jdata = json.load(f)
#     else:
#         # Load debug test sample
#         with open(file_data+'.reload', 'rt') as f:
#             jdata = json.load(f)
    else:
        # Download new values
        result = None
        url = REPO_URL + VOICES_JSON
        logger.debug(f"Downloading {url}")
        try:
            result = requests.get(url=url)
        except HTTPError as http_err:
            logger.error(f'HTTP error occurred: {http_err}')
        except Exception as err:
            logger.error(f'Other error occurred: {err}')
        if result is None:
            return
        if result.status_code != 200:
            logger.error(f'Wong status {result.status_code}')
            return
        jdata = result.json()
        logger.debug(f"Saving {file_data}")
        with open(file_data, 'wt') as f:
            f.write(json.dumps(jdata, indent=2))
    voices_data = jdata
    voices_by_lang = defaultdict(list)
    languages = set()
    name_to_voice = defaultdict(dict)
    total = 0
    for id, data in voices_data.items():
        if id != f"{data['language']}-{data['name']}-{data['quality']}":
            logger.warning(f"Piper id {id} doesn't match data")
            continue
        lang = lang_code_to_name(data['language'])
        voices_by_lang[lang].append(id)
        vis_name = get_voice_name(data)
        name_to_voice[lang][vis_name] = data
        languages.add(lang)
        total += data['num_speakers']
        voices_data[id]['lang_name'] = lang
        voices_data[id]['vis_name'] = vis_name
        voices_data[id]['id'] = id
    languages = sorted(list(languages))
    logger.debug(f'Piper available languages: {len(languages)}')
    logger.debug(f'Piper available models: {len(voices_data)}')
    logger.debug(f'Piper available voices: {total}')


def refresh_voices_dd_1(id):
    """ Ensure the selected voice/speaker are available """
    # Helpers
    no_change = gr.Dropdown.update()
    no_changes = (no_change, no_change, no_change)

    selected = params[f'selected_voice_{int(id)}']
    speaker = params[f'speaker_{int(id)}']
    sel_data = voices_data.get(selected)
    if sel_data is not None:
        # The selected voice is there
        id_map = sel_data['speaker_id_map']
        if speaker in id_map:
            # All ok!
            return no_changes
        logger.warning('Selected speaker no longer exists')
        avail_speakers, new_speaker, interactive = get_new_speaker(sel_data, speaker)
        return (no_change, no_change, gr.Dropdown.update(choices=avail_speakers, value=new_speaker, interactive=interactive))
    # The selected voice went away
    # Check the language exits
    logger.warning(f'Previously selected voice `{selected}` is no longer available')
    cur_lang = lang_code_to_name(selected[:5])
    if cur_lang not in languages:
        logger.warning(f'Previously selected language `{cur_lang}` is no longer available')
        new_lang = lang_code_to_name(DEFAULT_VOICE[:5])
        return (gr.Dropdown.update(choices=languages, value=new_lang), no_change, no_change)
    # Check the model exists
    avail_names = get_voices_for_lang(cur_lang)
    # We know the exact model+quality isn't there, what about with other quality?
    new_name = avail_names[0]
    cur_name = selected[6:].split('-')[0]
    for n in avail_names:
        if n.startswith(cur_name):
            new_name = n
            break
    sel_data = name_to_voice[new_name]
    # Check for the speaker
    id_map = sel_data['speaker_id_map']
    if speaker in id_map:
        # The speaker is there
        speaker_upd = no_change
    else:
        avail_speakers, new_speaker, interactive = get_new_speaker(sel_data, speaker, inform=True)
        speaker_upd = gr.Dropdown.update(choices=avail_speakers, value=new_speaker, interactive=interactive)
    return (no_change, gr.Dropdown.update(choices=avail_names, value=new_name), speaker_upd)


def refresh_voices_dd():
    """ Force a network refresh """
    refresh_voices(force_dl=True)
    return refresh_voices_dd_1(0) + refresh_voices_dd_1(1)


def get_voices_for_lang(lang):
    return sorted([v['vis_name'] for v in name_to_voice[lang].values()])


def change_lang(lang, name):
    """ Update the voice name, use the first for this language if the previous isn't available """
    avail_names = get_voices_for_lang(lang)
    if name not in avail_names:
       name = avail_names[0]
    return gr.Dropdown.update(choices=avail_names, value=name)


def change_voice_name(id, lang, name, speaker):
    """ Update the speaker list and selection """
    id = int(id)
    sel_v = f'selected_voice_{id}'
    current_voice_data = name_to_voice[lang][name]
    params[sel_v] = current_voice_data['id']
    logger.debug(f"Selecting piper voice id {id}: `{params[sel_v]}`")
    avail_speakers, new_speaker, interactive = get_new_speaker(current_voice_data, speaker)
    return (gr.Dropdown.update(choices=avail_speakers, value=new_speaker, interactive=interactive),
            gr.HTML.update(value=get_sample_html(id)),
            gr.Accordion.update(label=voice_accordion_label(id)))


def get_new_speaker(data, speaker, inform=False):
    avail_speakers = list(data['speaker_id_map'].keys())
    if inform and speaker and speaker not in avail_speakers:
        logger.warning(f'Selected speaker `{speaker}` not available')
    interactive = len(avail_speakers) > 0
    new_speaker = (speaker if speaker in avail_speakers else avail_speakers[0]) if interactive else ONLY_ONE
    return (avail_speakers, new_speaker, interactive)


def change_speaker(id, speaker):
    id = int(id)
    speaker_v = f'speaker_{id}'
    params[speaker_v] = speaker if speaker != ONLY_ONE else ''
    logger.debug(f"Selecting piper speaker {id}: `{params[speaker_v]}`")
    return (gr.HTML.update(value=get_sample_html(id)), gr.Accordion.update(label=voice_accordion_label(id)))


def get_sample_url(id, data=None, speaker=None):
    if data is None:
        data = voices_data[params[f'selected_voice_{int(id)}']]
    if speaker is None:
        speaker = params[f'speaker_{int(id)}']
    # Find the dir
    dir_name = os.path.dirname(next(iter(data['files'])))
    speaker = str(data['speaker_id_map'].get(speaker, 0))
    file = f'samples/speaker_{speaker}.mp3'
    return os.path.join(REPO_URL, dir_name, file)


def get_sample_html(id, data=None, speaker=None):
    return f'<audio src="{get_sample_url(id, data, speaker)}" controls></audio>'


def voice_accordion_label(id):
    id = int(id)
    sel_voice = params[f'selected_voice_{id}']
    speaker_v = params[f'speaker_{id}'] if params[f'speaker_{id}'] else ONLY_ONE
    enable_v = 'enabled' if params[f'enable_{id}'] else 'disabled'
    return f'{VOICE_TYPE[id]} voice parameters ({sel_voice}/{speaker_v}/{enable_v})'


def change_enabled(id, activate):
    params[f'enable_{int(id)}'] = activate
    return gr.Accordion.update(label=voice_accordion_label(id))


def add_voice_ui(id):
    # Parameters with the id
    sel_v = f'selected_voice_{id}'
    speaker_v = f'speaker_{id}'
    enable_v = f'enable_{id}'

    selected = params[sel_v]
    if selected == 'None' or selected is None:
        # TODO: select a voice according to the locale language
        selected = params[sel_v] = DEFAULT_VOICE
    elif selected not in voices_data:
        logger.warning(f'Selected voice `{selected}` not available, switching to {DEFAULT_VOICE}')
        selected = params[sel_v] = DEFAULT_VOICE

    current_voice_data = voices_data[selected]
    sel_lang = current_voice_data['lang_name']
    names = get_voices_for_lang(sel_lang)
    sel_name = current_voice_data['vis_name']
    speakers, sel_speaker, interactive_sp = get_new_speaker(current_voice_data, params[speaker_v], inform=True)
    params[speaker_v] = sel_speaker if interactive_sp else ''

    with gr.Accordion(voice_accordion_label(id), open=False) as accordion:
        with gr.Row():
            language = gr.Dropdown(value=sel_lang, choices=languages, label='Language')
            name = gr.Dropdown(value=sel_name, choices=names, label='Voice name (quality)')
            voice_id = gr.Number(value=id, visible=False)

        with gr.Row():
            speaker = gr.Dropdown(value=sel_speaker, choices=speakers, label='Speaker', interactive=interactive_sp)
            with gr.Column():
                activate = gr.Checkbox(value=params[enable_v], label='Enabled')
                audio_player = gr.HTML(value=get_sample_html(id, current_voice_data, sel_speaker))

    # Event functions to update the parameters in the backend
    language.change(change_lang, inputs=[language, name], outputs=[name])
    name.change(change_voice_name, inputs=[voice_id, language, name, speaker], outputs=[speaker, audio_player, accordion])
    speaker.change(change_speaker, inputs=[voice_id, speaker], outputs=[audio_player, accordion])
    activate.change(change_enabled, inputs=[voice_id, activate], outputs=[accordion])
    return [language, name, speaker]


def ui():
    if not voices_data:
        refresh_voices()

    # Gradio elements
    with gr.Row():
        activate = gr.Checkbox(value=params['activate'], label='Activate Text To Speach')
        refresh = gr.Button(value='Reload available voices')

    # Add controls for the primary and secondary voices
    outs_0 = add_voice_ui(0)
    outs_1 = add_voice_ui(1)

    # Event functions to update the parameters in the backend
    activate.change(lambda x: params.update({"activate": x}), activate, None)
    refresh.click(refresh_voices_dd, inputs=[], outputs=outs_0 + outs_1)

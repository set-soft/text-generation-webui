import glob
from collections import defaultdict
import hashlib
import gradio as gr
import json
import numpy as np
import os
from pathlib import Path
import re
import requests
import time
import yaml

from extensions.piper_tts.piper import Piper
from extensions.piper_tts.piper.samples import text_samples
from modules.logging_colors import logger
from modules.ui import create_refresh_button
from modules.utils import gradio
from modules import shared, chat
# from pprint import pprint

Piper._LOGGER = logger
DEFAULT_VOICE = "en_US-libritts-high"
REPO_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main/"
VOICES_JSON = "voices.json"
DATA_DIR = os.path.dirname(__file__)
VOICE_TYPE = ['Primary', 'Secondary']
MODELS_PATH = os.path.join("models", "piper")
PRESETS_PATH = os.path.join(MODELS_PATH, "presets")
PRESETS_P = Path(PRESETS_PATH)
DEF_PRESET_NAME = 'en_US_amy_and_libritts_p922_(default)'
USER_SAMPLE_TEXTS = os.path.join(MODELS_PATH, 'user_sample_texts.json')
OUTPUTS_PATH = os.path.join("extensions", "piper_tts", "outputs")
HIDE_WIDGET = gr.update(visible=False)
SHOW_WIDGET = gr.update(visible=True)
# Only one speaker available
ONLY_ONE = "Only one"
params = {
    "activate": True,
    "enable_0": True,
    "selected_voice_0": "en_US-amy-medium",  # "es_ES-sharvard-medium",
    "speaker_0": '',  # "F",
    "noise_scale_0": 0.667,
    "length_scale_0": 1,
    "noise_w_0": 0.8,
    "use_model_params_0": True,
    "show_sample_phonems_0": True,
    "enable_1": True,
    "selected_voice_1": DEFAULT_VOICE,  # "es_ES-sharvard-medium",
    "speaker_1": 'p922',  # "M",
    "noise_scale_1": 0.667,
    "length_scale_1": 1,
    "noise_w_1": 0.8,
    "use_model_params_1": True,
    "show_sample_phonems_1": True,
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
cfg = [None, None]
cfg_name = ['', '']
speaker_id = [0, 0]
# Text entered by the user
user_text_samples = {}
try:
    with open(USER_SAMPLE_TEXTS, 'rt') as f:
        user_text_samples = json.load(f)
except Exception:
    pass


def input_modifier(string):
    """
    This function is applied to your text inputs before
    they are fed into the model.
    """
    if not params['activate']:
        return string

    shared.processing_message = "*Is recording a voice message...*"
    return string


def create_output_file_name(base, ext):
    # The browser will cache the file, so we change its name
    # But we must delete old files to avoid filling the disk ;-)
    for f in glob.glob(os.path.join(OUTPUTS_PATH, base + '*' + ext)):
        os.remove(f)
    return os.path.join(OUTPUTS_PATH, base + f'_{time.time()}' + ext)


def output_modifier(string):
    """
    This function is applied to the model outputs.
    """
    global params, wav_idx

    if not params['activate']:
        return string

    logger.debug(f'Input: `{string}`')
    original_string = string
    # string = string.replace('"', '')
    # string = string.replace('“', '')
    # string = string.replace('\n', ' ')
    string = string.strip()
    if string == '':
        string = 'empty reply, try regenerating'

    output_file = create_output_file_name(f'{wav_idx:06d}', '.wav')
    logger.debug(f'Outputting audio to {output_file}')

    gen_audio_file(string, output_file)

    autoplay = 'autoplay' if params['autoplay'] else ''
    string = f'<audio src="file/{output_file}" controls {autoplay}></audio>'
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


def download_model_file(f_name, name, info, progress=None, downloaded=None, total=None, desc=None):
    d_name = os.path.dirname(f_name)
    os.makedirs(d_name, exist_ok=True)
    url = REPO_URL + name
    logger.debug(f'Downloading {url}')
    result = requests.get(url=url, stream=True)
    size = int(result.headers.get('content-length', 0))
    esize = info['size_bytes']
    if size != esize:
        raise ValueError(f'Downloaded size mismatch ({url}) {size} vs {esize}')

    block_size = 64 * 1024
    content = b''
    for data in result.iter_content(block_size):
        if progress is not None:
            downloaded += len(data)
            progress(downloaded/total, desc=desc)
        content += data

    h = hashlib.md5()
    h.update(content)
    res_md5 = h.hexdigest()
    e_md5 = info['md5_digest']
    if res_md5 != e_md5:
        raise ValueError(f'Downloaded integrity fail ({url}) {res_md5} vs {e_md5}')
    with open(f_name, 'wb') as f:
        f.write(content)

    return downloaded


def check_downloaded(data):
    for name in data['files'].keys():
        f_name = os.path.join(MODELS_PATH, name)
        if not os.path.isfile(f_name):
            return False
    return True


def download_voice(id, progress=gr.Progress()):
    files = voices_data[params[f'selected_voice_{int(id)}']]['files']
    yield (gr.Markdown.update("Downloading voice", visible=True), gr.Button.update(visible=False))
    progress(0.0, desc="Downloading files")
    yield (gr.Markdown.update(""), gr.Button.update(visible=False))
    total = 0
    for info in files.values():
        total += info['size_bytes']
    total_files = len(files)
    downloaded = 0
    for c, (name, info) in enumerate(files.items()):
        f_name = os.path.join(MODELS_PATH, name)
        if not os.path.isfile(f_name):
            desc = f"Downloading {name} ({c+1} of {total_files})"
            progress(downloaded/total, desc=desc)
            downloaded = download_model_file(f_name, name, info, progress, downloaded, total, desc)
        else:
            downloaded += info['size_bytes']
            progress(downloaded/total, desc=f"Skipping {name}")
    progress(1.0, desc="Finished")
    yield (gr.Markdown.update(value="", visible=False), gr.Button.update(visible=False))


def check_configured(voice, out_file):
    global cfg_name, cfg, speaker_id

    s_voice = str(voice)
    sel = params['selected_voice_' + s_voice]
    logger.debug(f'sel {sel}')
    data = voices_data[sel]
    logger.debug(f'cfg_name[voice] {cfg_name[voice]}')
    loaded = False
    if sel != cfg_name[voice]:
        logger.debug(f'Creating session for voice {voice}')
        # Not configured for this voice
        # Check the model files
        files = data['files']
        for c, (name, info) in enumerate(files.items()):
            f_name = os.path.join(MODELS_PATH, name)
            if name.endswith('.onnx'):
                model_name = f_name
            if not os.path.isfile(f_name):
                download_model_file(f_name, name, info)
        # Create a Piper object
        cfg[voice] = Piper(model_name)
        loaded = True
        cfg_name[voice] = sel
        # Ensure the output dir exists
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
    sel_speaker = params['speaker_' + s_voice]
    if sel_speaker:
        speaker_id[voice] = data['speaker_id_map'][sel_speaker]
    else:
        speaker_id[voice] = None
    return loaded


def gen_audio_file(string, out_file):
    global cfg, speaker_id

    # Check if configured
    start_time = time.time()
    loaded = False
    for v in range(2):
        loaded |= check_configured(v, out_file)
    elapsed_time = time.time() - start_time
    if loaded and elapsed_time:
        logger.debug(f'Models load time: {elapsed_time:5.2f} seconds')
    # Separate description from dialog
    sections = string.split('*')
    voice = 1
    audios = []
    silence = np.array(11000*[0.0], dtype=np.float32)
    enabled = (params['enable_0'], params['enable_1'])
    # Select the generation parameters
    noise_scale = (None, None)
    length_scale = (None, None)
    noise_w = (None, None)
    if not params['use_model_params_0']:
        noise_scale[0] = params['noise_scale_0']
        length_scale[0] = params['length_scale_0']
        noise_w[0] = params['noise_w_0']
    if not params['use_model_params_1']:
        noise_scale[1] = params['noise_scale_1']
        length_scale[1] = params['length_scale_1']
        noise_w[1] = params['noise_w_1']

    total_phonems = 0
    start_time = time.time()
    for s in sections:
        # Alternate voices
        voice = 1 - voice
        s = s.strip()
        logger.debug(f'{voice}: {s}')
        if not s or not enabled[voice]:
            # Skip empty sections
            continue
        audio, ps = cfg[voice].synthesize_partial(s, speaker_id=speaker_id[voice], length_scale=length_scale[voice],
                                                  noise_scale=noise_scale[voice], noise_w=noise_w[voice])
        # Show phonemizer result
        for p in ps:
            logger.debug('> ' + p)
            total_phonems += len(p)
        # Add this portion of audio and a silence to seprate from the next voice
        audios.append(audio)
        audios.append(silence)
    elapsed_time = time.time() - start_time
    if elapsed_time:
        logger.debug(f'{total_phonems} phonems in {elapsed_time:5.2f} seconds {round(total_phonems/elapsed_time)} p/s')

    with open(out_file, 'wb') as f:
        f.write(cfg[0].audios_to_wav(audios))


def lang_code_to_name(ln):
    return f"{ln['name_native']} ({ln['name_english']}) [{ln['country_english']}]"


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
        except requests.HTTPError as http_err:
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
        language = data['language']
        if id != f"{language['code']}-{data['name']}-{data['quality']}":
            logger.warning(f"Piper id {id} doesn't match data")
            continue
        lang = lang_code_to_name(language)
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


def get_sample_text(data):
    return text_samples.get(data['language']['family'], '???')


def change_voice_name(id, lang, name, speaker):
    """ Update the speaker list and selection """
    sel_v = f'selected_voice_{id}'
    current_voice_data = name_to_voice[lang][name]
    voice_id = current_voice_data['id']
    params[sel_v] = voice_id
    is_downloaded = check_downloaded(current_voice_data)
    logger.debug(f"Selecting piper voice id {id}: `{params[sel_v]}`")
    avail_speakers, new_speaker, interactive = get_new_speaker(current_voice_data, speaker)
    # speaker, audio_player, sample_txt, accordion, download, activate, edit_sample, hint_edit_sample, phonems
    return (gr.Dropdown.update(choices=avail_speakers, value=new_speaker, visible=interactive),
            gr.HTML.update(value=get_sample_html(id)),
            gr.Textbox.update(value=get_sample_text(current_voice_data)),
            gr.Accordion.update(label=voice_accordion_label(id)),
            gr.Button.update(visible=not is_downloaded),
            gr.Checkbox.update(value=is_downloaded, visible=is_downloaded),
            gr.Button.update(visible=is_downloaded),
            gr.Markdown.update(visible=not is_downloaded),
            gr.Textbox.update(visible=False, value=''))


def get_new_speaker(data, speaker, inform=False):
    avail_speakers = list(data['speaker_id_map'].keys())
    if inform and speaker and speaker not in avail_speakers:
        logger.warning(f'Selected speaker `{speaker}` not available')
    interactive = len(avail_speakers) > 0
    new_speaker = (speaker if speaker in avail_speakers else avail_speakers[0]) if interactive else ONLY_ONE
    return (avail_speakers, new_speaker, interactive)


def change_speaker(id, speaker):
    speaker_v = f'speaker_{id}'
    only_one = speaker != ONLY_ONE
    params[speaker_v] = speaker if only_one else ''
    logger.debug(f"Selecting piper speaker {id}: `{params[speaker_v]}`")
    return (gr.HTML.update(value=get_sample_html(id)),
            gr.Accordion.update(label=voice_accordion_label(id)))


def get_sample_url(id, data=None, speaker=None):
    if data is None:
        data = voices_data[params[f'selected_voice_{id}']]
    if speaker is None:
        speaker = params[f'speaker_{id}']
    # Find the dir
    dir_name = os.path.dirname(next(iter(data['files'])))
    speaker = str(data['speaker_id_map'].get(speaker, 0))
    file = f'samples/speaker_{speaker}.mp3'
    return os.path.join(REPO_URL, dir_name, file)


def get_sample_html(id, data=None, speaker=None):
    return f'<audio src="{get_sample_url(id, data, speaker)}" controls></audio>'


def voice_accordion_label(id):
    sel_voice = params[f'selected_voice_{id}']
    speaker_v = params[f'speaker_{id}'] if params[f'speaker_{id}'] else ONLY_ONE
    enable_v = 'enabled' if params[f'enable_{id}'] else 'disabled'
    return f'{VOICE_TYPE[id]} voice parameters ({sel_voice}/{speaker_v}/{enable_v})'


def change_enabled(id, activate):
    params[f'enable_{id}'] = activate
    return gr.Accordion.update(label=voice_accordion_label(id))


def change_status(str):
    """ This is used to enable the checkbox after download """
    # activate, edit_sample, hint_edit_sample
    return (gr.Checkbox.update(value=True, visible=True),
            gr.Button.update(visible=True),
            gr.Markdown.update(visible=False))


def change_use_model_params(id, enable):
    params[f'use_model_params_{id}'] = enable
    upd = gr.Slider.update(visible=not enable)
    return (upd, upd, upd)


def edit_sample_text(id):
    """ When we edit the text we disable the audio player and the edit button.
        We also enable the generate button and allow to enter new text. """
    lang = params[f'selected_voice_{id}'][:2]
    new_text = user_text_samples.get(lang, text_samples.get(lang, '???'))
    # sample_txt, edit_sample, audio_player, generate, phonems
    return (gr.Textbox.update(interactive=True, value=new_text),
            gr.Button.update(visible=False),
            gr.HTML.update(visible=False),
            gr.Button.update(visible=True),
            gr.Textbox.update(visible=False))


def save_user_model_params():
    global user_params
    try:
        with open(USER_PARAMS, 'wt') as f:
            json.dump(user_params, f, indent=2)
    except Exception as e:
        logger.error(f'Failed to save JSON file {USER_PARAMS}: {e}')


def generate_sample(id, sample_txt):
    """ Locally generate a sample for the voice """
    logger.debug(f'Generating for voice {id}: `{sample_txt}`')

    sel = params[f'selected_voice_{id}']
    data = voices_data[sel]
    files = data['files']
    for c, (name, info) in enumerate(files.items()):
        f_name = os.path.join(MODELS_PATH, name)
        if name.endswith('.onnx'):
            model_name = f_name
            break
    p = Piper(model_name)

    use_model_params = params[f'use_model_params_{id}']
    if use_model_params:
        noise_scale = length_scale = noise_w = None
    else:
        noise_scale = params[f'noise_scale_{id}']
        length_scale = params[f'length_scale_{id}']
        noise_w = params[f'noise_w_{id}']
    sel_speaker = params[f'speaker_{id}']
    if sel_speaker:
        speaker_id = data['speaker_id_map'][sel_speaker]
    else:
        speaker_id = None

    audios, phonems = p.synthesize_partial(sample_txt, speaker_id=speaker_id, length_scale=length_scale,
                                           noise_scale=noise_scale, noise_w=noise_w)

    output_file = create_output_file_name(f'sample_{sel}', '.wav')
    with open(output_file, 'wb') as f:
        f.write(p.audios_to_wav(audios))

    phonems = '\n'.join(phonems)

    # Memorize the sample text for this language
    user_text_samples[sel[:2]] = sample_txt
    try:
        with open(USER_SAMPLE_TEXTS, 'wt') as f:
            json.dump(user_text_samples, f, indent=2)
    except Exception as e:
        logger.error(f'Failed to save JSON file {USER_SAMPLE_TEXTS}: {e}')

    # Memorize the parameters for this voice
    sel_w_id = f'{sel}_{id}'
    global user_params
    if use_model_params:
        # Just indicate we are using the model parameters
        if sel in user_params:
            user_params[sel_w_id]['use_model_params'] = True
        else:
            user_params[sel_w_id] = {'use_model_params': True}
    else:
        # Custom values
        user_params[sel_w_id] = {'use_model_params': False,
                                 'noise_scale': noise_scale,
                                 'noise_w': noise_w,
                                 'length_scale': length_scale}
    save_user_model_params()

    # sample_txt, audio_player, edit_sample, generate, phonems
    return (gr.Textbox.update(interactive=False),
            gr.HTML.update(visible=True, value=f'<audio src="file/{output_file}" controls autoplay></audio>'),
            gr.Button.update(visible=True),
            gr.Button.update(visible=False),
            gr.Textbox.update(value=phonems, visible=params[f'show_sample_phonems_{id}']))


def change_show_sample_phonems(id, show, phonems):
    params[f'show_sample_phonems_{id}'] = show
    return gr.Textbox.update(visible=show and bool(phonems))


def add_voice_ui(id):
    # Parameters with the id
    id_str = f'_{id}'
    sel_v = 'selected_voice' + id_str
    speaker_v = 'speaker' + id_str
    enable_v = 'enable' + id_str
    noise_scale_v = 'noise_scale' + id_str
    length_scale_v = 'length_scale' + id_str
    noise_w_v = 'noise_w' + id_str
    use_model_params_v = 'use_model_params' + id_str
    show_sample_phonems_v = 'show_sample_phonems' + id_str

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
    is_downloaded = check_downloaded(current_voice_data)
    if not is_downloaded:
        # Disable it if the files aren't there
        params[enable_v] = False

    with gr.Accordion(voice_accordion_label(id), open=False) as accordion:
        with gr.Box():
            with gr.Row():
                with gr.Column():
                    language = gr.Dropdown(value=sel_lang, choices=languages, label='Language')
                    speaker = gr.Dropdown(value=sel_speaker, choices=speakers, label='Speaker', visible=interactive_sp)

                with gr.Column():
                    name = gr.Dropdown(value=sel_name, choices=names, label='Voice name (quality)')
                    voice_id = gr.Number(value=id, visible=False, precision=0)
                    activate = gr.Checkbox(value=params[enable_v], label='Enabled', visible=is_downloaded)

        with gr.Box():
            sample_txt = gr.Textbox(value=get_sample_text(current_voice_data), interactive=False, label='Sample text', lines=2)

            phonems = gr.TextArea(label='Generated phonems', visible=False, interactive=False)

            with gr.Row(variant='panel'):
                # Sample player: from internet or using a custom text, after downloading the model
                audio_player = gr.HTML(value=get_sample_html(id, current_voice_data, sel_speaker))
                # This button is enabled when the user edits the text
                generate = gr.Button(value='Generate sample', visible=False)
                # Allow to edit the sample text, only if we have the model
                edit_sample = gr.Button(value='Edit sample text', visible=is_downloaded)
                # Help hint to know when the text can be edited
                hint_edit_sample = gr.Markdown(value='Hint:\n\nDownload the voice to test it using custom text\n',
                                               visible=not is_downloaded)

            show_sample_phonems = gr.Checkbox(value=params[show_sample_phonems_v],
                                              label='Show phonemizer output for generated sample')

        with gr.Box():
            use_mp = params[use_model_params_v]
            use_model_params = gr.Checkbox(value=use_mp, label='Use model parameters',
                                           info="Disable it to finetune the voice generation. "
                                                "Note that they don't affect the default voice sample")
            with gr.Row():
                noise_scale = gr.Slider(value=params[noise_scale_v], label='Noise scale (variability)', minimum=0, maximum=2,
                                        visible=not use_mp)
                noise_w = gr.Slider(value=params[noise_w_v], label='Noise W (cadence)', minimum=0, maximum=2,
                                    visible=not use_mp)

            with gr.Row():
                length_scale = gr.Slider(value=params[length_scale_v], label='Length scale (speed)', minimum=0, maximum=2,
                                         visible=not use_mp)

        with gr.Row():
            download = gr.Button(value='Download voice', visible=not is_downloaded)
            download_status = gr.Markdown(visible=False)

    # Event functions to update the parameters in the backend
    language.change(change_lang, inputs=[language, name], outputs=[name])
    name.change(change_voice_name, inputs=[voice_id, language, name, speaker],
                outputs=[speaker, audio_player, sample_txt, accordion, download, activate, edit_sample, hint_edit_sample,
                         phonems])
    speaker.change(change_speaker, inputs=[voice_id, speaker], outputs=[audio_player, accordion])
    activate.change(change_enabled, inputs=[voice_id, activate], outputs=[accordion])
    length_scale.change(lambda x: params.update({length_scale_v: x}), length_scale, None)
    noise_scale.change(lambda x: params.update({noise_scale_v: x}), noise_scale, None)
    noise_w.change(lambda x: params.update({noise_w_v: x}), noise_w, None)
    use_model_params.change(change_use_model_params, [voice_id, use_model_params], [noise_scale, noise_w, length_scale])
    download.click(download_voice, voice_id, [download_status, download])
    download_status.change(change_status, download_status, [activate, edit_sample, hint_edit_sample])
    # Enable the textbox, disable the button and enable the generate button
    edit_sample.click(edit_sample_text, voice_id, [sample_txt, edit_sample, audio_player, generate, phonems])
    generate.click(generate_sample, [voice_id, sample_txt], [sample_txt, audio_player, edit_sample, generate, phonems])
    show_sample_phonems.change(change_show_sample_phonems, [voice_id, show_sample_phonems, phonems], phonems)

    return [language, name, speaker]


def remove_tts_from_history(history):
    """ Removes all the audio from the chat history """
    visible = history['visible']  # Contains pairs User/Character
    for i, entry in enumerate(history['internal']):
        visible[i] = [visible[i][0], entry[1]]

    return history


def toggle_text_in_history(history):
    """ Shows or hides the text in the history """
    visible = history['visible']  # Contains pairs User/Character
    for i, entry in enumerate(history['visible']):
        visible_reply = entry[1]
        if visible_reply.startswith('<audio'):
            if params['show_text']:
                # Audio + Text
                reply = history['internal'][i][1]
                visible[i] = [visible[i][0], f"{visible_reply.split('</audio>')[0]}</audio>\n\n{reply}"]
            else:
                # Just the audio
                visible[i] = [visible[i][0], f"{visible_reply.split('</audio>')[0]}</audio>"]

    return history


def atoi(text):
    return int(text) if text.isdigit() else text.lower()


def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]


def get_available_presets():
    return sorted(set((k.stem for k in Path(PRESETS_PATH).glob('*.yaml'))), key=natural_keys)


def get_default_preset():
    presets_dir = Path(PRESETS_PATH)
    presets_dir.mkdir(parents=True, exist_ok=True)
    default_preset = None
    # Look for the last one selected
    last_preset = presets_dir/'last_preset.txt'
    if last_preset.is_file():
        with last_preset.open() as f:
            sel_preset = f.readline().strip()
        if (presets_dir/(sel_preset + '.yaml')).is_file():
            default_preset = sel_preset
    # Try to peek one
    if default_preset is None:
        first = next(presets_dir.glob('*.yaml'), None)
        if first is not None:
            default_preset = first.stem
    # Nothing there, create one
    def_preset = presets_dir/(DEF_PRESET_NAME+'.yaml')
    if not def_preset.exists():
        with def_preset.open('wt') as f:
            yaml.dump(params, f, sort_keys=False)
    if default_preset is None:
        default_preset = def_preset.stem
        with last_preset.open('wt') as f:
            f.write(default_preset)
    return default_preset


def start_save_preset():
    new_name = 'new_preset_'
    n = 1
    while (PRESETS_P/f'user_preset_{n}.yaml').is_file():
        n += 1
    return (gr.Textbox.update(value=f'user_preset_{n}'),
            gr.update(visible=True))


def check_preset_name(name):
    """ Check that this name can be used """
    if not name:
        return (HIDE_WIDGET, gr.update(value='Empty name'))
    res = re.search(r'[^\w\d _\(\),]', name)
    if res:
        return (HIDE_WIDGET, gr.Markdown.update(value='Invalid chars in name'))
    if (PRESETS_P/(name+'.yaml')).exists():
        return (HIDE_WIDGET, gr.update(value='Name already used'))
    return (SHOW_WIDGET, gr.update(value='Name is OK'))


def do_save_preset(name):
    """ Save current options as a preset """
    logger.info(f'Saving preset {name}')
    with (PRESETS_P/(name+'.yaml')).open(mode='wt') as f:
        f.write(yaml.dump(params, sort_keys=False))
    return (HIDE_WIDGET, name)


def do_del_preset(name):
    """ Delete selected preset """
    logger.info(f'Removing preset {name}')
    (PRESETS_P/(name+'.yaml')).unlink()
    # Hide the confirm dialog and select the default
    return (HIDE_WIDGET, DEF_PRESET_NAME)


def do_load_preset(new_preset, old_preset):
    result = []
    if new_preset != DEF_PRESET_NAME:
        result.append(SHOW_WIDGET)
        result.append(SHOW_WIDGET)
    else:
        result.append(HIDE_WIDGET)
        result.append(HIDE_WIDGET)
    result.append(new_preset)
    if new_preset == old_preset:
        # Save changes the selection, but isn't a real change
        logger.warning('No se lee')
        return result
    logger.warning('Implementar lectura')
    return result


def show_remove_options(name):
    """ Make the confirmation visible and update the confirmation text """
    return (SHOW_WIDGET, gr.update(value=f'Are you sure you want to delete the _{name}_ preset?\n\n.'))


def show_save_options(name):
    """ Make the confirmation visible and update the confirmation text """
    return (SHOW_WIDGET, gr.update(value=f'Are you sure you want to overwrite the _{name}_ preset?\n\n.'))


def ui():
    if not voices_data:
        refresh_voices()

    cur_preset_name = get_default_preset()

    # Gradio elements
    with gr.Row():
        activate = gr.Checkbox(value=params['activate'], label='Activate Text To Speach')
        refresh = gr.Button(value='Reload available voices')

    with gr.Row():
        cur_is_default = cur_preset_name == DEF_PRESET_NAME
        preset_menu = gr.Dropdown(choices=get_available_presets(), value=cur_preset_name,
                                  label='Configuration presets', elem_classes='slim-dropdown')
        cur_preset = gr.Textbox(value=cur_preset_name, visible=False)
        create_refresh_button(preset_menu, lambda: None, lambda: {'choices': get_available_presets()}, 'refresh-button')
        add_preset = gr.Button('+', elem_classes='refresh-button')
        save_preset = gr.Button('💾', elem_classes='refresh-button', visible=not cur_is_default)
        delete_preset = gr.Button('🗑️', elem_classes='refresh-button', visible=not cur_is_default)

    with gr.Row():
        autoplay = gr.Checkbox(value=params['autoplay'], label='Play TTS automatically')
        show_text = gr.Checkbox(value=params['show_text'], label='Show message text under audio player')

    if shared.is_chat():
        with gr.Row():
            convert = gr.Button('Permanently replace audios with the message texts')
            convert_cancel = gr.Button('Cancel', visible=False)
            convert_confirm = gr.Button('Confirm (cannot be undone)', variant="stop", visible=False)

        # Convert history with confirmation
        convert_arr = [convert_confirm, convert, convert_cancel]
        confirm_cancel = [gr.update(visible=True), gr.update(visible=False), gr.update(visible=True)]
        only_convert = [gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)]
        # On "convert" click change to confirm/cancel
        convert.click(lambda: confirm_cancel, None, convert_arr)
        # On "confirm" click remove audios
        convert_confirm.click(lambda: only_convert, None, convert_arr).\
            then(remove_tts_from_history, gradio('history'), gradio('history')).\
            then(chat.save_persistent_history, gradio('history', 'character_menu', 'mode'), None).\
            then(chat.redraw_html, shared.reload_inputs, gradio('display'))
        # On "cancel" just change to the convert button
        convert_cancel.click(lambda: only_convert, None, convert_arr)

        # Toggle message text in history
        show_text.change(lambda x: params.update({"show_text": x}), show_text, None).\
            then(toggle_text_in_history, gradio('history'), gradio('history')).\
            then(chat.save_persistent_history, gradio('history', 'character_menu', 'mode'), None).\
            then(chat.redraw_html, shared.reload_inputs, gradio('display'))

    # Add controls for the primary and secondary voices
    outs_0 = add_voice_ui(0)
    outs_1 = add_voice_ui(1)

    # Event functions to update the parameters in the backend
    activate.change(lambda x: params.update({"activate": x}), activate, None)
    refresh.click(refresh_voices_dd, inputs=[], outputs=outs_0 + outs_1)
    autoplay.change(lambda x: params.update({"autoplay": x}), autoplay, None)

    # #################
    # Add preset dialog
    with gr.Box(visible=False, elem_classes='file-saver') as file_adder:
        add_filename = gr.Textbox(lines=1, label='New name')
        name_status = gr.Textbox(value='Name is OK', show_label=False, interactive=False)
        with gr.Row():
            add_confirm = gr.Button('Save', elem_classes="small-button")
            add_cancel = gr.Button('Cancel', elem_classes="small-button")

    # Action for the button to add the preset
    add_preset.click(start_save_preset, None, [add_filename, file_adder])

    # Add actions
    add_filename.input(check_preset_name, add_filename, [add_confirm, name_status], show_progress=False)
    add_confirm.click(do_save_preset, add_filename, [file_adder, cur_preset], show_progress=False).\
        then(lambda x: gr.Dropdown.update(choices=get_available_presets(), value=x), cur_preset, preset_menu, show_progress=False)
    add_cancel.click(lambda: HIDE_WIDGET, None, file_adder)

    # ####################
    # Delete preset dialog
    with gr.Box(visible=False, elem_classes='file-saver') as file_remover:
        del_msg = gr.Markdown()
        with gr.Row():
            del_confirm = gr.Button('Delete', elem_classes="small-button", variant='stop')
            del_cancel = gr.Button('Cancel', elem_classes="small-button")

    # Action for the button to remove a preset
    delete_preset.click(show_remove_options, preset_menu, [file_remover, del_msg], show_progress=False)

    # Delete actions
    del_cancel.click(lambda: HIDE_WIDGET, None, file_remover, show_progress=False)
    del_confirm.click(do_del_preset, cur_preset, [file_remover, cur_preset], show_progress=False).\
        then(lambda x: gr.Dropdown.update(choices=get_available_presets(), value=x), cur_preset, preset_menu, show_progress=False)

    # ##################
    # Save preset dialog
    with gr.Box(visible=False, elem_classes='file-saver') as file_saver:
        save_msg = gr.Markdown()
        with gr.Row():
            save_confirm = gr.Button('Replace', elem_classes="small-button", variant='stop')
            save_cancel = gr.Button('Cancel', elem_classes="small-button")

    # Action for the button to save a preset (overwrite)
    save_preset.click(show_save_options, preset_menu, [file_saver, save_msg], show_progress=False)

    # Save actions
    save_cancel.click(lambda: HIDE_WIDGET, None, file_saver, show_progress=False)
    save_confirm.click(do_save_preset, cur_preset, [file_saver, cur_preset], show_progress=False).\
        then(lambda x: gr.Dropdown.update(choices=get_available_presets(), value=x), cur_preset, preset_menu, show_progress=False)

    preset_menu.change(do_load_preset, [preset_menu, cur_preset], [save_preset, delete_preset, cur_preset])

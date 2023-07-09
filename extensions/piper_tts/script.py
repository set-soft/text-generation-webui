from collections import defaultdict
import gradio as gr
import json
import os
import pycountry
import requests
from modules.logging_colors import logger

DEFAULT_VOICE = "en_US-libritts-high"
REPO_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main/"
VOICES_JSON = "voices.json"
DATA_DIR = os.path.dirname(__file__)
# Only one speaker available
ONLY_ONE = "Only one"
params = {
    "activate": True,
    "selected_voice": "es_ES-sharvard-medium",
    "speaker": "F",
}
voices_data = None
voices_by_lang = None
name_to_voice = None
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


def lang_code_to_name(ln):
    ln_code_2 = ln[:2].upper()
    co_code_2 = ln[3:]
    # Patch errors
    if co_code_2 == 'UK':
        co_code_2 = 'UA'
    return (pycountry.languages.get(alpha_2=ln_code_2).name + ' (' +
            pycountry.countries.get(alpha_2=co_code_2).name + ')')


def refresh_voices(force_dl=False):
    global voices_data
    global voices_by_lang
    global languages
    if voices_data is None:
        voices_data = {}
        voices_by_lang = defaultdict(list)
        languages = set()

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
            print(f'HTTP error occurred: {http_err}')
        except Exception as err:
            print(f'Other error occurred: {err}')
        if result is None:
            return
        if result.status_code != 200:
            print(f'Wong status {result.status_code}')
            return
        jdata = result.json()
        logger.debug(f"Saving {file_data}")
        with open(file_data, 'wt') as f:
            f.write(json.dumps(jdata, indent=2))
    voices_data = jdata
    voices_by_lang = defaultdict(list)
    languages = set()
    total = 0
    for id, data in voices_data.items():
        if id != f"{data['language']}-{data['name']}-{data['quality']}":
            logger.warning(f"Piper id {id} doesn't match data")
            continue
        lang = lang_code_to_name(data['language'])
        voices_by_lang[lang].append(id)
        languages.add(lang)
        total += data['num_speakers']
        voices_data[id]['lang_name'] = lang
        voices_data[id]['id'] = id
    languages = sorted(list(languages))
    logger.debug(f'Piper available languages: {len(languages)}')
    logger.debug(f'Piper available models: {len(voices_data)}')
    logger.debug(f'Piper available voices: {total}')


def refresh_voices_dd():
    """ Force a network refresh """
    no_change = gr.Dropdown.update()
    no_changes = (no_change, no_change, no_change)

    refresh_voices(force_dl=True)
    selected = params['selected_voice']
    speaker = params['speaker']
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


def get_voice_name(v):
    return f"{v['name']} ({v['quality']})"


def get_voices_for_lang(lang):
    global name_to_voice
    names = []
    name_to_voice = {}
    for v in voices_by_lang[lang]:
        data = voices_data[v]
        name = get_voice_name(data)
        names.append(name)
        name_to_voice[name] = data
    return names


def change_lang(lang, name):
    # Update the voice name, use the first for this language
    print(f"change_lang `{lang}`")
    avail_names = get_voices_for_lang(lang)
    if name not in avail_names:
       name = avail_names[0]
    return gr.Dropdown.update(choices=avail_names, value=name)


def change_voice_name(name, speaker):
    # Update the speaker list and selection
    print(f"change_voice_name `{name}`")
    current_voice_data = name_to_voice[name]
    params['selected_voice'] = current_voice_data['id']
    logger.debug(f"Selecting piper voice id: {params['selected_voice']}")
    avail_speakers, new_speaker, interactive = get_new_speaker(current_voice_data, speaker)
    return (gr.Dropdown.update(choices=avail_speakers, value=new_speaker, interactive=interactive),
            gr.HTML.update(value=get_sample_html()))


def get_new_speaker(data, speaker, inform=False):
    avail_speakers = list(data['speaker_id_map'].keys())
    if inform and speaker not in avail_speakers:
        logger.warning(f'Selected speaker `{speaker}` not available')
    interactive = len(avail_speakers) > 0
    new_speaker = (speaker if speaker in avail_speakers else avail_speakers[0]) if interactive else ONLY_ONE
    return (avail_speakers, new_speaker, interactive)


def change_speaker(speaker):
    params['speaker'] = speaker if speaker != ONLY_ONE else ''
    print(f"change_speaker `{params['speaker']}`")
    return gr.HTML.update(value=get_sample_html())


def get_sample_url(data=None, speaker=None):
    if data is None:
        data = voices_data[params['selected_voice']]
    if speaker is None:
        speaker = params['speaker']
    # Find the dir
    dir_name = os.path.dirname(next(iter(data['files'])))
    speaker = str(data['speaker_id_map'].get(speaker, 0))
    file = f'samples/speaker_{speaker}.mp3'
    return os.path.join(REPO_URL, dir_name, file)


def get_sample_html(data=None, speaker=None):
    return f'<audio src="{get_sample_url(data, speaker)}" controls></audio>'


def ui():
    if not voices_data:
        refresh_voices()
    selected = params['selected_voice']
    if selected == 'None' or selected is None:
        # TODO: select a voice according to the locale language
        selected = params['selected_voice'] = DEFAULT_VOICE
    elif selected not in voices_data:
        logger.warning(f'Selected voice `{selected}` not available, switching to {DEFAULT_VOICE}')
        selected = params['selected_voice'] = DEFAULT_VOICE

    current_voice_data = voices_data[selected]
    sel_lang = current_voice_data['lang_name']
    names = get_voices_for_lang(sel_lang)
    sel_name = get_voice_name(current_voice_data)
    speakers, sel_speaker, interactive_sp = get_new_speaker(current_voice_data, params['speaker'], inform=True)

    # Gradio elements
    with gr.Row():
        activate = gr.Checkbox(value=params['activate'], label='Activate speak')

    with gr.Accordion("Voice parameters", open=False):
        with gr.Row():
            language = gr.Dropdown(value=sel_lang, choices=languages, label='Language')
            refresh = gr.Button(value='Refresh')

        with gr.Row():
            name = gr.Dropdown(value=sel_name, choices=names, label='Name')
            speaker = gr.Dropdown(value=sel_speaker, choices=speakers, label='Speaker', interactive=interactive_sp)

        with gr.Row():
            audio_player = gr.HTML(value=get_sample_html(current_voice_data, sel_speaker))

    # Event functions to update the parameters in the backend
    activate.change(lambda x: params.update({"activate": x}), activate, None)
    language.change(change_lang, inputs=[language, name], outputs=[name])
    name.change(change_voice_name, inputs=[name, speaker], outputs=[speaker, audio_player])
    speaker.change(change_speaker, speaker, audio_player)
    refresh.click(refresh_voices_dd, inputs=[], outputs=[language, name, speaker])

"""
UI Gradio per inferenza con i modelli fine-tuned openai/privacy-filter.

Uso:
    source .venv/bin/activate
    pip install gradio
    python app.py

Apre automaticamente il browser su http://localhost:7860
"""

import os

# CRITICO: disabilita i kernel Triton (CUDA-only) prima di importare opf.
# Su Apple Silicon (MPS) e su CPU il path Triton non funziona.
os.environ['OPF_MOE_TRITON'] = '0'

import torch
import gradio as gr
from opf import OPF


# ═══════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════

ROOT = os.path.dirname(os.path.abspath(__file__))

MODELS = {
    # Modello v3: generico italiano (unico). Step legale rimosso perché overfittava.
    'Generico italiano (v3)':         os.path.join(ROOT, 'models/checkpoint_generic_italian_v3'),
    # Legacy v2: tenuti come fallback se vuoi confrontare
    'Legacy: documenti italiani (v2)': os.path.join(ROOT, 'models/checkpoint_step1_italian_docs_v2'),
    'Legacy: docs + legale (v2)':      os.path.join(ROOT, 'models/checkpoint_step2_legal_v2'),
}

DEVICE = ('mps' if torch.backends.mps.is_available()
          else 'cuda' if torch.cuda.is_available()
          else 'cpu')


# ═══════════════════════════════════════════════════════════════════════
# Lazy loading — tiene in RAM solo il modello selezionato
# ═══════════════════════════════════════════════════════════════════════

_cache = {'name': None, 'model': None}


def get_model(name):
    if _cache['name'] != name:
        path = MODELS[name]
        if not os.path.isdir(path):
            raise gr.Error(
                f'Checkpoint non ancora presente: {path}\n\n'
                f'Scompatta prima lo zip scaricato da Kaggle in:\n{os.path.dirname(path)}/\n\n'
                f'Oppure seleziona un altro modello dal dropdown.'
            )
        print(f'⚙️  Caricamento {name} da {path} (device={DEVICE})...')
        _cache['model'] = OPF(
            model=path,
            device=DEVICE,
            output_mode='typed',
            decode_mode='viterbi',
        )
        _cache['name'] = name
        print('✅ Caricato')
    return _cache['model']


# ═══════════════════════════════════════════════════════════════════════
# Logica di anonimizzazione
# ═══════════════════════════════════════════════════════════════════════

CATEGORY_LABELS = {
    'private_person':      '👤 Persona',
    'private_address':     '📍 Indirizzo',
    'private_email':       '✉️ Email',
    'private_phone':       '📞 Telefono',
    'private_url':         '🔗 URL',
    'private_date':        '📅 Data',
    'account_number':      '🔢 Num. conto',
    'secret':              '🔐 Secret',
    'codice_fiscale':      '🆔 Codice Fiscale',
    'carta_identita':      '🪪 Carta identità',
    'patente':             '🚗 Patente',
    'passaporto':          '✈️ Passaporto',
    'partita_iva':         '💼 Partita IVA',
    'iban':                '🏦 IBAN',
    'tessera_sanitaria':   '⚕️ Tessera sanitaria',
    'numero_procedimento': '⚖️ N. procedimento',
    'riferimento_catastale': '🏠 Rif. catastale',
    'parte_in_causa':      '👥 Parte in causa',
}


def redact(text, model_name):
    if not text or not text.strip():
        return '', [], '⚠️ Inserisci del testo da analizzare'

    model = get_model(model_name)
    result = model.redact(text)

    rows = []
    for s in result.detected_spans:
        pretty_label = CATEGORY_LABELS.get(s.label, s.label)
        score = getattr(s, 'score', None)
        rows.append([
            pretty_label,
            s.text,
            f'{s.start}:{s.end}',
            f'{score:.2f}' if score is not None else '—',
        ])

    from collections import Counter
    cnt = Counter(s.label for s in result.detected_spans)
    if cnt:
        summary = f'✅ Rilevate **{len(result.detected_spans)}** entità: '
        summary += ', '.join(f'{v}× {CATEGORY_LABELS.get(k, k)}' for k, v in cnt.most_common())
    else:
        summary = '✅ Nessuna entità sensibile rilevata'

    return result.redacted_text, rows, summary


# ═══════════════════════════════════════════════════════════════════════
# UI
# ═══════════════════════════════════════════════════════════════════════

DEFAULT_MODEL = 'Generico italiano (v3)'
EXAMPLES = [
    ['Ciao Marco, ti scrivo per conferma l\'appuntamento di domani. Chiamami al 333 1234567 se cambi idea. — Luigi Bianchi', DEFAULT_MODEL],
    ['Il sottoscritto Mario Rossi, CF RSSMRA80A01H501U, residente in Via Roma 10, Milano, presenta la seguente richiesta.', DEFAULT_MODEL],
    ['Per bonifici utilizzare IBAN IT60X0542811101000000123456 intestato a Luigi Bianchi. Contatti: luigi.bianchi@studio.it', DEFAULT_MODEL],
    ['Candidatura spontanea — Giulia Marino, nata a Bologna nel 1992, giulia.marino@email.it, 3382345678. CV allegato.', DEFAULT_MODEL],
    ['Oggi a Roma, Anna Conti ha inaugurato la nuova mostra presso il museo. L\'evento è stato seguito da oltre 200 persone.', DEFAULT_MODEL],
    ["Ordine #45821 confermato. Consegna a Roberto Costa, Via Manzoni 15, Torino. Grazie per il tuo acquisto!", DEFAULT_MODEL],
]

CSS = '''
.gradio-container {max-width: 1400px !important;}
#input_text textarea, #output_text textarea {font-family: 'SF Mono', Menlo, monospace; font-size: 14px;}
'''


def build_ui():
    with gr.Blocks(title='Privacy Filter IT') as demo:
        gr.Markdown(
            '# 🔒 Privacy Filter — Anonimizzazione testi italiani\n\n'
            'Modelli fine-tuned su documenti italiani e dominio legale. '
            f'Device: **{DEVICE}** — inferenza locale, i dati non lasciano la tua macchina.'
        )

        with gr.Row():
            model_choice = gr.Dropdown(
                choices=list(MODELS.keys()),
                value=DEFAULT_MODEL,
                label='🤖 Modello (v3 = generico, v2 = legacy)',
                scale=2,
            )
            btn = gr.Button('✨ Anonimizza', variant='primary', scale=1)

        with gr.Row():
            input_text = gr.Textbox(
                label='📝 Testo originale',
                placeholder='Incolla qui il testo da anonimizzare...',
                lines=18,
                elem_id='input_text',
            )
            output_text = gr.Textbox(
                label='🔒 Testo anonimizzato',
                lines=18,
                interactive=False,
                elem_id='output_text',
            )

        summary = gr.Markdown('')

        entities = gr.Dataframe(
            headers=['Categoria', 'Testo rilevato', 'Posizione', 'Confidence'],
            label='🔍 Dettaglio entità rilevate',
            wrap=True,
            interactive=False,
        )

        gr.Examples(
            examples=EXAMPLES,
            inputs=[input_text, model_choice],
            label='Esempi — clicca per caricare',
        )

        # Triggers
        btn.click(
            fn=redact,
            inputs=[input_text, model_choice],
            outputs=[output_text, entities, summary],
        )
        input_text.submit(
            fn=redact,
            inputs=[input_text, model_choice],
            outputs=[output_text, entities, summary],
        )

    return demo


if __name__ == '__main__':
    print(f'Device rilevato: {DEVICE}')
    demo = build_ui()
    demo.launch(
        server_name='127.0.0.1',
        server_port=7860,
        inbrowser=True,
        css=CSS,
        theme=gr.themes.Soft(),
    )

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


SHORT_NAMES = {
    'private_person':        'PERSONA',
    'private_address':       'INDIRIZZO',
    'private_email':         'EMAIL',
    'private_phone':         'TELEFONO',
    'private_url':           'URL',
    'private_date':          'DATA',
    'account_number':        'CONTO',
    'secret':                'SECRET',
    'codice_fiscale':        'CF',
    'carta_identita':        'CARTA_ID',
    'patente':               'PATENTE',
    'passaporto':            'PASSAPORTO',
    'partita_iva':           'PIVA',
    'iban':                  'IBAN',
    'tessera_sanitaria':     'TS',
    'numero_procedimento':   'PROC',
    'riferimento_catastale': 'CATASTO',
    'parte_in_causa':        'PARTE',
}


def _name_words(text):
    """Estrae i token significativi di un nome (lowercase, senza punteggiatura)."""
    import re
    # Mantieni apostrofi/trattini interni a parola (D'Angelo, De-Luca)
    cleaned = re.sub(r"[^\w\s'`\-]", ' ', text.lower())
    return [w for w in cleaned.split() if w and len(w) > 1]


def _cluster_persons(person_spans):
    """Cluster di coreference per persona.

    Logica:
    1) Nomi completi (≥2 parole significative) creano cluster.
       Cluster si fondono se condividono almeno una parola (Mario Rossi + Mario
       Giuseppe Rossi → stesso cluster).
    2) Nomi parziali (1 parola) si collegano al cluster di un nome completo che
       contiene quella parola. Se più cluster matchano, vince il più vicino in
       posizione (proximity).
    3) Nomi parziali senza match a nessun nome completo creano cluster propri,
       raggruppando le occorrenze identiche.

    Ritorna: dict (start, end) -> cluster_id (int da 1).
    """
    if not person_spans:
        return {}

    sorted_spans = sorted(person_spans, key=lambda s: s.start)
    # Pre-calcola word set per ogni span
    span_words = [(s, set(_name_words(s.text))) for s in sorted_spans]

    # ─ Pass 1: nomi completi (≥2 parole) ─
    # Strategia union-find semplice: lista di set di word
    clusters = []  # list of dict: {'words': set, 'spans': [(s, e), ...]}
    span_to_cluster_idx = {}

    for span, words in span_words:
        if len(words) < 2:
            continue
        merged_idx = None
        for i, c in enumerate(clusters):
            # Fonde solo se uno è subset dell'altro: "Mario Rossi" + "Mario
            # Giuseppe Rossi" sì, "Mario Rossi" + "Mario Bianchi" no.
            if words.issubset(c['words']) or c['words'].issubset(words):
                if merged_idx is None:
                    c['words'] |= words
                    c['spans'].append((span.start, span.end))
                    merged_idx = i
                else:
                    # Fonde clusters[merged_idx] e clusters[i]
                    clusters[merged_idx]['words'] |= c['words']
                    clusters[merged_idx]['spans'].extend(c['spans'])
                    clusters[i] = None  # marca per rimozione
        if merged_idx is None:
            clusters.append({'words': set(words), 'spans': [(span.start, span.end)]})
            merged_idx = len(clusters) - 1
        span_to_cluster_idx[(span.start, span.end)] = merged_idx

    # Compatta clusters (rimuovi None)
    new_clusters = []
    idx_map = {}
    for old_i, c in enumerate(clusters):
        if c is None:
            continue
        idx_map[old_i] = len(new_clusters)
        new_clusters.append(c)
    clusters = new_clusters
    span_to_cluster_idx = {k: idx_map[v] for k, v in span_to_cluster_idx.items() if v in idx_map}
    # Aggiorna anche le liste interne
    for c in clusters:
        c['spans'] = sorted(set(c['spans']))

    # ─ Pass 2: nomi parziali (1 parola) ─
    # Per ognuno, trova cluster compatibili (che contengono la parola).
    # Se 1 → assegna. Se >1 → proximity. Se 0 → nuovo cluster (eventualmente
    # raggruppato con altri short-name spans con stessa parola).
    short_only_clusters = {}  # word -> cluster_idx (solo per short-names orfani)

    for span, words in span_words:
        if len(words) != 1:
            continue
        word = next(iter(words))
        # Cluster dei full-name che contengono questa parola
        candidates = [i for i, c in enumerate(clusters) if word in c['words']]

        if len(candidates) == 1:
            chosen = candidates[0]
        elif len(candidates) > 1:
            # Proximity: cluster con uno span più vicino
            def min_dist(cidx):
                return min(abs(s - span.start) for s, _ in clusters[cidx]['spans'])
            chosen = min(candidates, key=min_dist)
        else:
            # Nessun full-name → cluster autonomo per questa parola
            if word in short_only_clusters:
                chosen = short_only_clusters[word]
            else:
                clusters.append({'words': {word}, 'spans': []})
                chosen = len(clusters) - 1
                short_only_clusters[word] = chosen

        clusters[chosen]['spans'].append((span.start, span.end))
        span_to_cluster_idx[(span.start, span.end)] = chosen

    # ─ Assegna ID finali in order of first appearance ─
    cluster_first_pos = {}
    for span_pos, cidx in span_to_cluster_idx.items():
        if cidx not in cluster_first_pos or span_pos[0] < cluster_first_pos[cidx]:
            cluster_first_pos[cidx] = span_pos[0]
    # Ordina cluster per prima posizione, assegna ID 1, 2, 3...
    sorted_cidx = sorted(cluster_first_pos, key=lambda i: cluster_first_pos[i])
    cidx_to_id = {cidx: i + 1 for i, cidx in enumerate(sorted_cidx)}

    return {sp: cidx_to_id[cidx] for sp, cidx in span_to_cluster_idx.items()}


def redact_with_progressive_ids(text, spans):
    """Anonimizza il testo con ID progressivi per categoria.

    Per `private_person`: cluster con coreference (Mario Rossi + Rossi → stesso ID).
    Per altre categorie: matching esatto sul testo normalizzato.

    Esempio:
      "Il sig. Mario Rossi e il dott. Bianchi. Poi Rossi ha aggiunto..."
      → "<PERSONA_1> e <PERSONA_2>. Poi <PERSONA_1> ha aggiunto..."
    """
    from collections import defaultdict

    person_spans = [s for s in spans if s.label == 'private_person']
    other_spans  = [s for s in spans if s.label != 'private_person']

    span_ids = {}

    # ─ Coreference clustering per persone ─
    person_ids = _cluster_persons(person_spans)
    span_ids.update(person_ids)

    # ─ Matching esatto per altre categorie ─
    counters = defaultdict(int)
    id_map = {}
    for s in sorted(other_spans, key=lambda x: x.start):
        norm = s.text.strip().lower()
        key = (s.label, norm)
        if key not in id_map:
            counters[s.label] += 1
            id_map[key] = counters[s.label]
        span_ids[(s.start, s.end)] = id_map[key]

    # ─ Sostituisci nel testo (da destra a sinistra) ─
    result = text
    sorted_all = sorted(spans, key=lambda s: s.start, reverse=True)
    for s in sorted_all:
        short = SHORT_NAMES.get(s.label, s.label.upper())
        sid = span_ids[(s.start, s.end)]
        tag = f'<{short}_{sid}>'
        result = result[:s.start] + tag + result[s.end:]

    return result, span_ids


def redact(text, model_name):
    if not text or not text.strip():
        return '', '', [], '⚠️ Inserisci del testo da analizzare'

    model = get_model(model_name)
    result = model.redact(text)

    # Testo redatto con ID progressivi
    redacted_ids, span_ids = redact_with_progressive_ids(text, result.detected_spans)

    rows = []
    for s in sorted(result.detected_spans, key=lambda x: x.start):
        pretty_label = CATEGORY_LABELS.get(s.label, s.label)
        score = getattr(s, 'score', None)
        sid = span_ids.get((s.start, s.end), '?')
        rows.append([
            f'{pretty_label} #{sid}',
            s.text,
            f'{s.start}:{s.end}',
            f'{score:.2f}' if score is not None else '—',
        ])

    from collections import Counter
    cnt = Counter(s.label for s in result.detected_spans)
    if cnt:
        # Conta anche entità uniche per categoria (coreference approssimata)
        unique_per_label = Counter()
        seen = set()
        for s in result.detected_spans:
            key = (s.label, s.text.strip().lower())
            if key not in seen:
                seen.add(key)
                unique_per_label[s.label] += 1
        parts = []
        for k, v in cnt.most_common():
            u = unique_per_label[k]
            label = CATEGORY_LABELS.get(k, k)
            if u == v:
                parts.append(f'{v}× {label}')
            else:
                parts.append(f'{v}× {label} ({u} unici)')
        summary = f'✅ Rilevate **{len(result.detected_spans)}** entità: ' + ', '.join(parts)
    else:
        summary = '✅ Nessuna entità sensibile rilevata'

    return result.redacted_text, redacted_ids, rows, summary


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
            with gr.Column():
                output_text = gr.Textbox(
                    label='🔒 Anonimizzato (standard)',
                    lines=8,
                    interactive=False,
                    elem_id='output_text',
                )
                output_text_ids = gr.Textbox(
                    label='🔢 Anonimizzato con ID progressivi (stessa entità → stesso ID)',
                    lines=8,
                    interactive=False,
                    elem_id='output_text_ids',
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
            outputs=[output_text, output_text_ids, entities, summary],
        )
        input_text.submit(
            fn=redact,
            inputs=[input_text, model_choice],
            outputs=[output_text, output_text_ids, entities, summary],
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

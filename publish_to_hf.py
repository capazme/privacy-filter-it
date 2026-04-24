"""
Pubblica un checkpoint fine-tuned di openai/privacy-filter su HuggingFace Hub.

Uso:
    # 1) Installa la dipendenza
    pip install -U huggingface_hub

    # 2) Login (una tantum — apre il browser per autorizzare)
    huggingface-cli login

    # 3) Pubblica
    python publish_to_hf.py \\
        --checkpoint models/checkpoint_step1_italian_docs_v2 \\
        --repo-id TUO_USERNAME/privacy-filter-it-docs-v2 \\
        --private

    # Dopo il push, il modello è accessibile su:
    #   https://huggingface.co/TUO_USERNAME/privacy-filter-it-docs-v2

Il modello usa un'architettura CUSTOM (model_type: privacy_filter) non
registrata in transformers. Gli utenti dovranno installare opf per usarlo:
    pip install git+https://github.com/openai/privacy-filter.git
"""

import argparse
import json
import os
import sys
from pathlib import Path


MODEL_CARD_TEMPLATE = """---
language:
- it
license: apache-2.0
library_name: opf
base_model: openai/privacy-filter
pipeline_tag: token-classification
tags:
- privacy-filter
- pii-detection
- italian
- anonymization
- ner
- opf
---

# {repo_name}

Fine-tuning di [openai/privacy-filter](https://huggingface.co/openai/privacy-filter) su documenti italiani sintetici per il riconoscimento di PII (Personally Identifiable Information).

{description}

## ⚠️ Come caricare il modello

Questo modello usa un'architettura **custom** (`model_type: privacy_filter`) **non** registrata in `transformers`. NON funziona con `AutoModel` / `transformers.pipeline`.

Per usarlo serve la libreria `opf`:

```bash
pip install git+https://github.com/openai/privacy-filter.git
```

```python
import os
os.environ['OPF_MOE_TRITON'] = '0'  # disabilita kernel CUDA-only su MPS/CPU

from opf import OPF
from huggingface_hub import snapshot_download

# Scarica il modello (viene messo in cache locale)
local_path = snapshot_download(repo_id='{repo_id}')

model = OPF(
    model=local_path,
    device='cuda',           # oppure 'mps' (Apple Silicon) o 'cpu'
    output_mode='typed',
    decode_mode='viterbi',
)

text = 'Il sottoscritto Mario Rossi, CF RSSMRA80A01H501U, residente in Via Roma 10, Milano.'
result = model.redact(text)

print(result.redacted_text)
# -> Il sottoscritto <PRIVATE_PERSON>, <CODICE_FISCALE>, residente in <PRIVATE_ADDRESS>.

for span in result.detected_spans:
    print(f'{{span.label:25s}} "{{span.text}}" [{{span.start}}:{{span.end}}]')
```

## 📋 Categorie riconosciute

Il modello riconosce **{n_labels}** categorie di PII italiane:

| Categoria | Descrizione |
|---|---|
{label_table}

## 📊 Dettagli training

- **Base model**: `openai/privacy-filter`
- **Dataset**: sintetico, generato dal modulo `dataset_builder.py` (vedi [repo GitHub]({github_url}))
- **Dati**: **{n_train}** esempi di training, **{n_val}** di validation, **{n_test}** di test (held-out)
- **Epoche**: {epochs}
- **Batch size**: {batch_size}
- **Grad accum steps**: {grad_accum}
- **Learning rate**: {lr}
- **Hardware training**: {device}
- **Best epoch**: {best_epoch} (validation loss: {best_loss:.4f})
- **Param dtype**: {param_dtype}

## 🎯 Metriche (validation set)

- **Token accuracy** (best): {token_acc:.4f}
- **Validation loss** (best): {best_loss:.4f}

## 🎨 Esempi di output

**Input**: `Per bonifici IBAN IT60X0542811101000000123456 intestato a Luigi Bianchi. luigi.bianchi@studio.it`

**Output**: `Per bonifici <IBAN> intestato a <PRIVATE_PERSON>. <PRIVATE_EMAIL>`

## ⚖️ Licenza & limitazioni

- **Licenza**: Apache 2.0 (ereditata dal base model)
- **Limiti**: il dataset è sintetico — il modello potrebbe avere pattern overfitted su formati tipici (es. "CF RSSMRA80A01H501U" preceduto da prefisso). Testa con i tuoi testi prima dell'uso in produzione.
- **Contesto**: addestrato su testo italiano generico (email, CV, news, chat, business). Non ottimizzato per domini specifici (medico, scientifico, etc.).
- **Dati sintetici**: nessun dato reale di terze parti usato nel training. Tutti gli esempi sono generati programmaticamente con formati italiani validi ma valori casuali.

## 📎 Citazione

Se usi questo modello, per favore cita il lavoro originale di OpenAI:

```
@misc{{openai-privacy-filter,
  title = {{Privacy Filter}},
  author = {{OpenAI}},
  year = {{2024}},
  url = {{https://github.com/openai/privacy-filter}}
}}
```
"""


CATEGORY_DESCRIPTIONS = {
    'O':                     'Outside — token non è una PII',
    'private_person':        'Nomi di persone fisiche',
    'private_address':       'Indirizzi (vie, città, numeri civici)',
    'private_email':         'Indirizzi email',
    'private_phone':         'Numeri di telefono italiani',
    'private_url':           'URL contenenti dati personali',
    'private_date':          'Date (nascita, scadenze, eventi)',
    'account_number':        'Numeri di conto (generici)',
    'secret':                'Credenziali, password, token',
    'codice_fiscale':        'Codice Fiscale italiano (16 caratteri)',
    'carta_identita':        "Numero Carta d'Identità italiana",
    'patente':               'Numero Patente di guida',
    'passaporto':            'Numero Passaporto',
    'partita_iva':           'Partita IVA italiana (11 cifre)',
    'iban':                  'IBAN italiano (27 caratteri)',
    'tessera_sanitaria':     'Tessera Sanitaria',
    'numero_procedimento':   'Numero procedimento legale (RG)',
    'riferimento_catastale': 'Riferimento catastale (foglio/mappale)',
    'parte_in_causa':        'Parti in procedimento giudiziario',
}


def load_checkpoint_info(checkpoint_dir: Path) -> dict:
    """Legge config.json e finetune_summary.json dal checkpoint."""
    config = json.loads((checkpoint_dir / 'config.json').read_text())
    summary = json.loads((checkpoint_dir / 'finetune_summary.json').read_text())
    return {'config': config, 'summary': summary}


def build_model_card(
    repo_id: str,
    checkpoint_dir: Path,
    description: str,
    github_url: str,
) -> str:
    info = load_checkpoint_info(checkpoint_dir)
    cfg, sm = info['config'], info['summary']

    span_classes = sm.get('span_class_names', [])
    # Skip 'O' nella label table
    label_rows = [
        f'| `{lab}` | {CATEGORY_DESCRIPTIONS.get(lab, "—")} |'
        for lab in span_classes if lab != 'O'
    ]

    best_loss = sm.get('best_metric', 0.0)
    token_acc = 0.0
    if sm.get('epoch_metrics'):
        best_ep = sm.get('best_epoch', 1)
        for ep in sm['epoch_metrics']:
            if ep.get('epoch') == best_ep:
                token_acc = ep.get('validation_token_accuracy', 0.0)
                break

    repo_name = repo_id.split('/')[-1] if '/' in repo_id else repo_id

    return MODEL_CARD_TEMPLATE.format(
        repo_name=repo_name,
        repo_id=repo_id,
        description=description,
        github_url=github_url,
        n_labels=len(span_classes) - 1,  # escludi 'O'
        label_table='\n'.join(label_rows),
        n_train=sm.get('num_train_examples', '?'),
        n_val=sm.get('num_validation_examples', '?'),
        n_test='(held-out, non usato in training)',
        epochs=sm.get('best_epoch', '?'),
        batch_size=sm.get('batch_size', '?'),
        grad_accum=sm.get('grad_accum_steps', '?'),
        lr=sm.get('learning_rate', '?'),
        device=sm.get('device', '?'),
        best_epoch=sm.get('best_epoch', '?'),
        best_loss=best_loss,
        param_dtype=sm.get('serialized_param_dtype', '?'),
        token_acc=token_acc,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint', required=True, type=Path,
                        help='Path alla cartella del checkpoint (es. models/checkpoint_generic_italian_v3)')
    parser.add_argument('--repo-id', required=True,
                        help='Repo su HF: username/nome-modello (es. mario/privacy-filter-it-v3)')
    parser.add_argument('--private', action='store_true',
                        help='Crea il repo come privato (default: pubblico)')
    parser.add_argument('--description', default='',
                        help='Descrizione breve da inserire nel model card')
    parser.add_argument('--github-url', default='https://github.com/capazme/privacy-filter-it',
                        help='Link al repo GitHub da citare nel model card')
    parser.add_argument('--token', default=None,
                        help='HF token (default: usa cache da `huggingface-cli login`)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Genera solo il README.md senza uploadare')
    args = parser.parse_args()

    # ─ Validazioni ─
    ckpt = args.checkpoint.resolve()
    if not ckpt.is_dir():
        sys.exit(f'❌ Checkpoint non trovato: {ckpt}')
    required = ['config.json', 'model.safetensors']
    for f in required:
        if not (ckpt / f).exists():
            sys.exit(f'❌ File mancante nel checkpoint: {f}')

    if '/' not in args.repo_id:
        sys.exit('❌ --repo-id deve avere il formato "username/nome-modello"')

    # ─ Genera model card ─
    print(f'📝 Genero model card per {args.repo_id}...')
    card = build_model_card(
        repo_id=args.repo_id,
        checkpoint_dir=ckpt,
        description=args.description or f'Modello addestrato su dataset sintetico italiano ({ckpt.name}).',
        github_url=args.github_url,
    )
    readme_path = ckpt / 'README.md'
    readme_path.write_text(card)
    print(f'✅ README.md scritto: {readme_path}')
    print(f'   ({len(card):,} caratteri)')

    if args.dry_run:
        print('\n--dry-run attivo: nessun upload eseguito.')
        print(f'Puoi leggere il model card in {readme_path} prima di procedere.')
        return

    # ─ Upload su HF Hub ─
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        sys.exit('❌ huggingface_hub non installato. Esegui: pip install -U huggingface_hub')

    api = HfApi(token=args.token)

    print(f'\n🔧 Creo/verifico repo {args.repo_id} (private={args.private})...')
    create_repo(
        repo_id=args.repo_id,
        private=args.private,
        exist_ok=True,
        token=args.token,
    )
    print('✅ Repo pronto')

    print(f'\n📤 Upload di {ckpt} → {args.repo_id}...')
    print(f'   (model.safetensors è ~{(ckpt / "model.safetensors").stat().st_size / 1e9:.1f} GB, potrebbero volerci minuti)')
    api.upload_folder(
        folder_path=str(ckpt),
        repo_id=args.repo_id,
        repo_type='model',
        commit_message=f'Upload fine-tuned checkpoint: {ckpt.name}',
        ignore_patterns=['.DS_Store', '*.pyc', '__pycache__'],
    )
    print(f'✅ Upload completato!')
    print(f'\n🎉 Modello pubblicato su:')
    print(f'   https://huggingface.co/{args.repo_id}')


if __name__ == '__main__':
    main()

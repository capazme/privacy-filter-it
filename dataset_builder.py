"""
Generatori di dataset sintetici per fine-tuning di `opf` (openai/privacy-filter)
su documenti d'identità italiani + dominio legale.

Uso nel notebook:
    from dataset_builder import (
        LABEL_SPACE,
        gen_step1_examples, gen_step2_examples,
        validate_spans, label_distribution,
        build_all,
    )
"""

import random
import string
from collections import Counter


# ═══════════════════════════════════════════════════════════════════════
# Label space
# ═══════════════════════════════════════════════════════════════════════

LABEL_SPACE = {
    "category_version": "italian_legal_v1",
    "span_class_names": [
        "O",                     # OBBLIGATORIO primo elemento
        # Categorie originali del modello base
        "private_person",
        "private_address",
        "private_email",
        "private_phone",
        "private_url",
        "private_date",
        "account_number",
        "secret",
        # Documenti italiani
        "codice_fiscale",
        "carta_identita",
        "patente",
        "passaporto",
        "partita_iva",
        "iban",
        "tessera_sanitaria",
        # Dominio legale
        "numero_procedimento",
        "riferimento_catastale",
        "parte_in_causa",
    ],
}


# ═══════════════════════════════════════════════════════════════════════
# Constants — dati anagrafici
# ═══════════════════════════════════════════════════════════════════════

NOMI_M = [
    'Marco', 'Luca', 'Andrea', 'Giovanni', 'Paolo', 'Matteo', 'Lorenzo', 'Stefano',
    'Roberto', 'Giuseppe', 'Antonio', 'Francesco', 'Davide', 'Alessandro', 'Riccardo',
]
NOMI_F = [
    'Maria', 'Laura', 'Sara', 'Anna', 'Elena', 'Giulia', 'Francesca', 'Valentina',
    'Chiara', 'Federica', 'Silvia', 'Paola', 'Roberta', 'Alessandra', 'Beatrice',
]
COGNOMI = [
    'Rossi', 'Bianchi', 'Ferrari', 'Esposito', 'Romano', 'Colombo', 'Ricci', 'Marino',
    'Greco', 'Bruno', 'Gallo', 'Conti', 'De Luca', 'Mancini', 'Costa', 'Giordano',
    'Lombardi', 'Barbieri', 'Moretti', 'Fontana', 'Santoro', 'Mariani', 'Russo', 'Ferrara',
]
COMUNI = [
    'Roma', 'Milano', 'Napoli', 'Torino', 'Palermo', 'Genova', 'Bologna', 'Firenze',
    'Bari', 'Catania', 'Venezia', 'Verona', 'Messina', 'Padova', 'Trieste', 'Brescia',
]
COD_COMUNI = {
    'Roma': 'H501', 'Milano': 'F205', 'Napoli': 'F839', 'Torino': 'L219',
    'Palermo': 'G273', 'Genova': 'D969', 'Bologna': 'A944', 'Firenze': 'D612',
    'Bari': 'A662', 'Catania': 'C351', 'Venezia': 'L736', 'Verona': 'L781',
    'Messina': 'F158', 'Padova': 'G224', 'Trieste': 'L424', 'Brescia': 'B157',
}
MESI_CF = 'ABCDEHLMPRST'

# Dominio legale
TRIBUNALI = [
    'Tribunale di Roma', 'Tribunale di Milano', 'Tribunale di Napoli',
    'Tribunale di Torino', 'Tribunale di Bologna', 'Tribunale di Firenze',
    "Corte d'Appello di Roma", "Corte d'Appello di Milano",
    'Tribunale Civile di Bari', 'Corte di Cassazione',
]
NOTAI = ['Notaio dott.', 'Notaio dott.ssa', 'Notaio']
RUOLI = [
    'attore', 'convenuto', 'ricorrente', 'resistente', 'appellante', 'appellato',
    'opponente', 'intimato', 'terzo chiamato', 'interveniente',
]
TIPI_CONTRATTO = [
    'contratto di compravendita', 'contratto di locazione', "contratto d'appalto",
    'contratto di mutuo', "contratto di prestazione d'opera",
    'contratto di mandato', 'atto di donazione', 'scrittura privata',
]
SEZIONI = [
    'Prima Sezione Civile', 'Seconda Sezione Civile', 'Sezione Lavoro',
    'Sezione Fallimentare', 'Terza Sezione Civile', 'Prima Sezione Penale',
]

# ── Frasi negative (nessun PII) ─────────────────────────────────────────
NEGATIVES_STEP1 = [
    'La documentazione deve essere presentata entro i termini previsti dalla legge.',
    'Si richiede di produrre copia conforme del documento originale.',
    'Il modulo è disponibile presso gli uffici competenti.',
    'La domanda deve essere corredata di marca da bollo da 16 euro.',
    "L'ufficio è aperto dal lunedì al venerdì dalle 9:00 alle 13:00.",
    'Il pagamento deve essere effettuato entro 30 giorni dalla notifica.',
    'Per qualsiasi chiarimento è possibile rivolgersi al responsabile del procedimento.',
    'La ricevuta di pagamento deve essere allegata alla domanda.',
    'I dati personali saranno trattati ai sensi del Regolamento UE 2016/679.',
    'La presente comunicazione ha valore meramente informativo.',
    "In caso di mancato riscontro, l'istanza si intenderà respinta.",
    'Il modulo va compilato in ogni sua parte e firmato in originale.',
    'La pratica sarà evasa nei tempi previsti dalla normativa vigente.',
    'È possibile richiedere una copia autenticata presso la cancelleria.',
    'La scadenza per la presentazione è fissata al 31 dicembre.',
    "L'istanza deve contenere l'indicazione del codice univoco.",
    'La presente attestazione è rilasciata in carta libera a uso amministrativo.',
]

NEGATIVES_STEP2 = [
    'Le parti si danno atto di non avere nulla da eccepire in ordine alla regolarità del contratto.',
    "Il presente atto è esente da imposta di registro ai sensi dell'art. 1 della Tariffa allegata al D.P.R. 131/1986.",
    'Il Giudice si riserva di decidere con separato provvedimento.',
    'Le spese processuali seguono la soccombenza e si liquidano in dispositivo.',
    "La sentenza è esecutiva per legge ai sensi dell'art. 282 c.p.c.",
    "Il presente atto sarà trascritto presso l'Agenzia del Territorio competente.",
    'Visti gli atti e udite le parti, il Tribunale così provvede.',
    "L'istanza è respinta per mancanza dei presupposti di legge.",
    "La memoria difensiva è depositata nei termini di cui all'art. 183 c.p.c.",
    'Letti gli atti del procedimento, il Collegio si riserva di pronunciare sentenza.',
    "Il ricorso è dichiarato inammissibile ai sensi dell'art. 360 bis c.p.c.",
    'La presente procedura è regolata dalle norme del codice civile in materia di contratti.',
    'Il deposito telematico è stato effettuato ai sensi della normativa vigente.',
    'La causa è rinviata per la precisazione delle conclusioni.',
    'Il mandato si intende conferito con ogni più ampia facoltà di legge.',
    'Nulla osta al rilascio del provvedimento richiesto.',
]


# ═══════════════════════════════════════════════════════════════════════
# Generatori di valori sintetici
# ═══════════════════════════════════════════════════════════════════════

def rand_nome(genere=None):
    g = genere or random.choice(['M', 'F'])
    nome = random.choice(NOMI_M if g == 'M' else NOMI_F)
    return nome, random.choice(COGNOMI), g


def rand_data(anno_min=1940, anno_max=2000):
    anno = random.randint(anno_min, anno_max)
    mese = random.randint(1, 12)
    giorno = random.randint(1, 28)
    return giorno, mese, anno


def rand_comune():
    return random.choice(COMUNI)


# Codice Fiscale ────────────────────────────────────────────────────────

def _cf_consonanti(s):
    return [c for c in s.upper() if c in 'BCDFGHJKLMNPQRSTVWXYZ']


def _cf_vocali(s):
    return [c for c in s.upper() if c in 'AEIOU']


def _cf_cognome(cog):
    c, v = _cf_consonanti(cog), _cf_vocali(cog)
    p = c + v + ['X', 'X', 'X']
    return ''.join(p[:3]).upper()


def _cf_nome(nom):
    c = _cf_consonanti(nom)
    if len(c) >= 4:
        return (c[0] + c[2] + c[3]).upper()
    v = _cf_vocali(nom)
    p = c + v + ['X', 'X', 'X']
    return ''.join(p[:3]).upper()


def gen_cf(nome=None, cognome=None, genere=None,
           giorno=None, mese=None, anno=None, comune=None):
    if nome is None:
        nome, cognome, genere = rand_nome(genere)
    if giorno is None:
        giorno, mese, anno = rand_data()
    if comune is None:
        comune = rand_comune()
    p_cog = _cf_cognome(cognome)
    p_nom = _cf_nome(nome)
    p_ann = str(anno)[-2:]
    p_mes = MESI_CF[mese - 1]
    p_gg = str(giorno + (40 if genere == 'F' else 0)).zfill(2)
    p_com = COD_COMUNI.get(comune, 'H501')
    base = p_cog + p_nom + p_ann + p_mes + p_gg + p_com
    # Carattere di controllo (semplificato — non implementa la tabella ufficiale)
    conv_p = [1, 0, 5, 7, 9, 13, 15, 17, 19, 21, 1, 0, 5, 7, 9, 13, 15, 17, 19, 21,
              2, 4, 18, 20, 11, 3, 6, 8, 12, 14, 16, 10, 22, 25, 24, 23]
    s = 0
    for i, c in enumerate(base):
        v = int(c) if c.isdigit() else ord(c) - 65
        s += conv_p[v] if i % 2 == 0 else v
    ctrl = chr(65 + (s % 26))
    return base + ctrl, nome, cognome


# Altri documenti ──────────────────────────────────────────────────────

def gen_ci():
    """Carta d'identità nuovo formato: AA0000000."""
    lettere = ''.join(random.choices(string.ascii_uppercase, k=2))
    cifre = ''.join(random.choices(string.digits, k=7))
    return lettere + cifre


def gen_patente():
    """Patente italiana: AA000AA00000A."""
    p = ''.join(random.choices(string.ascii_uppercase, k=2))
    p += ''.join(random.choices(string.digits, k=3))
    p += ''.join(random.choices(string.ascii_uppercase, k=2))
    p += ''.join(random.choices(string.digits, k=5))
    p += random.choice(string.ascii_uppercase)
    return p


def gen_passaporto():
    """Passaporto italiano: AA0000000."""
    return (
        ''.join(random.choices(string.ascii_uppercase, k=2))
        + ''.join(random.choices(string.digits, k=7))
    )


def gen_piva():
    """Partita IVA italiana (11 cifre, ultima = controllo Luhn-like)."""
    cifre = [random.randint(0, 9) for _ in range(10)]
    s = 0
    for i, d in enumerate(cifre):
        if i % 2 == 0:
            s += d
        else:
            dd = d * 2
            s += dd if dd < 10 else dd - 9
    ctrl = (10 - (s % 10)) % 10
    return ''.join(str(d) for d in cifre) + str(ctrl)


def gen_iban():
    """IBAN italiano (27 caratteri)."""
    banca = (
        ''.join(random.choices(string.ascii_uppercase, k=1))
        + ''.join(random.choices(string.digits, k=4))
    )
    filiale = ''.join(random.choices(string.digits, k=5))
    conto = ''.join(random.choices(string.digits + string.ascii_uppercase, k=12))
    ctrl = ''.join(random.choices(string.digits, k=2))
    cin = random.choice(string.ascii_uppercase)
    return f'IT{ctrl}{cin}{banca}{filiale}{conto}'


def gen_ts(cf):
    """Tessera sanitaria: prefisso numerico + CF."""
    prefisso = '80038' + ''.join(random.choices(string.digits, k=7))
    return prefisso + cf


def gen_procedimento():
    anno = random.randint(2015, 2024)
    num = random.randint(100, 9999)
    tipo = random.choice(['RG', 'RGNR', 'RGN', 'R.G.'])
    return f'n. {num}/{anno} {tipo}'


def gen_catastale():
    foglio = random.randint(1, 999)
    mappale = random.randint(1, 9999)
    sub = random.randint(1, 99)
    sezione = random.choice(['A', 'B', 'C', 'D', 'E', '']).strip()
    if sezione:
        return f'foglio {foglio}, mappale {mappale}, sub. {sub}, sez. {sezione}'
    return f'foglio {foglio}, mappale {mappale}, sub. {sub}'


def gen_telefono():
    prefissi = ['02', '06', '011', '081', '091', '049', '051', '055', '010', '090']
    cel_pref = ['320', '328', '333', '334', '338', '339', '347', '348', '349',
                '366', '380', '388']
    if random.random() > 0.4:
        return (
            '+39 '
            + random.choice(cel_pref)
            + ' '
            + ''.join(random.choices(string.digits, k=3))
            + ' '
            + ''.join(random.choices(string.digits, k=4))
        )
    return (
        random.choice(prefissi)
        + ' '
        + ''.join(random.choices(string.digits, k=7))
    )


def gen_email(nome, cognome):
    domini = ['gmail.com', 'libero.it', 'yahoo.it', 'hotmail.it',
              'outlook.it', 'tiscali.it', 'alice.it']
    sep = random.choice(['.', '_', ''])
    n = nome.lower().replace(' ', '')
    c = cognome.lower().replace(' ', '')
    variants = [f'{n}{sep}{c}', f'{c}{sep}{n}', f'{n[0]}{sep}{c}',
                f'{n}{random.randint(1, 99)}']
    return random.choice(variants) + '@' + random.choice(domini)


# ═══════════════════════════════════════════════════════════════════════
# Helpers per costruire esempi
# ═══════════════════════════════════════════════════════════════════════

def make_ex(text, entities):
    """Costruisce un esempio nel formato opf {text, spans:{label:[[start,end]]}}.

    entities: lista di tuple (substring, label).
    Salta silenziosamente substring non trovate.
    """
    spans = {}
    for sub, label in entities:
        idx = text.find(sub)
        if idx == -1:
            continue
        spans.setdefault(label, []).append([idx, idx + len(sub)])
    return {'text': text, 'spans': spans}


# ═══════════════════════════════════════════════════════════════════════
# Step 1 — Documenti d'identità italiani
# ═══════════════════════════════════════════════════════════════════════

def gen_step1_examples(n=300, negative_rate=0.15):
    """Genera n esempi su documenti d'identità italiani in contesti comuni.

    negative_rate: quota di esempi senza PII (0.15 = 15%) per ridurre falsi positivi.
    """
    examples = []

    for _ in range(n):
        # Esempio negativo (nessun PII): scelto con probabilità negative_rate
        if random.random() < negative_rate:
            examples.append(make_ex(random.choice(NEGATIVES_STEP1), []))
            continue

        nome, cognome, genere = rand_nome()
        nome_completo = f'{nome} {cognome}'
        gg, mm, aa = rand_data()
        comune = rand_comune()
        data_nascita = f'{gg:02d}/{mm:02d}/{aa}'
        cf, _, _ = gen_cf(nome, cognome, genere, gg, mm, aa, comune)
        ci = gen_ci()
        pat = gen_patente()
        pas = gen_passaporto()
        piva = gen_piva()
        iban = gen_iban()
        ts = gen_ts(cf)
        tel = gen_telefono()
        email = gen_email(nome, cognome)
        art = 'il sig.' if genere == 'M' else 'la sig.ra'
        art2 = 'nato' if genere == 'M' else 'nata'
        art3 = 'residente'
        via_num = random.choice(['Via Roma', 'Via Garibaldi', 'Corso Italia',
                                  'Viale Europa', 'Via Manzoni', 'Piazza Duomo',
                                  'Via Verdi', 'Corso Vittorio'])
        num_civ = random.randint(1, 200)
        indirizzo = f'{via_num} {num_civ}, {comune}'

        tpl = random.randint(0, 35)

        if tpl == 0:
            t = f'Il sottoscritto {nome_completo}, codice fiscale {cf}, dichiara quanto segue.'
            e = [(nome_completo, 'private_person'), (cf, 'codice_fiscale')]
        elif tpl == 1:
            t = f"Carta d'identità n. {ci} rilasciata al sig. {nome_completo}."
            e = [(ci, 'carta_identita'), (nome_completo, 'private_person')]
        elif tpl == 2:
            t = f'Patente di guida: {pat}, intestata a {nome_completo}, {art2} a {comune}.'
            e = [(pat, 'patente'), (nome_completo, 'private_person'), (comune, 'private_address')]
        elif tpl == 3:
            t = f'Passaporto n. {pas} del sig. {nome_completo}, valido fino al {gg:02d}/{mm:02d}/{aa+10}.'
            e = [(pas, 'passaporto'), (nome_completo, 'private_person')]
        elif tpl == 4:
            t = f'Partita IVA: {piva} — Titolare: {nome_completo}, con sede in {comune}.'
            e = [(piva, 'partita_iva'), (nome_completo, 'private_person'), (comune, 'private_address')]
        elif tpl == 5:
            t = f'Coordinate bancarie: IBAN {iban}, intestato a {nome_completo}.'
            e = [(iban, 'iban'), (nome_completo, 'private_person')]
        elif tpl == 6:
            t = f'Tessera sanitaria n. {ts} — Assistito: {nome_completo}.'
            e = [(ts, 'tessera_sanitaria'), (nome_completo, 'private_person')]
        elif tpl == 7:
            t = f'{art.capitalize()} {nome_completo}, C.F. {cf}, {art2} il {data_nascita} a {comune}, chiede il rilascio del documento.'
            e = [(nome_completo, 'private_person'), (cf, 'codice_fiscale'),
                 (data_nascita, 'private_date'), (comune, 'private_address')]
        elif tpl == 8:
            t = f'Ai sensi del D.Lgs. 196/2003, si comunica che i dati di {nome_completo} (C.F. {cf}) saranno trattati in conformità alla normativa vigente.'
            e = [(nome_completo, 'private_person'), (cf, 'codice_fiscale')]
        elif tpl == 9:
            t = f'Si attesta che {art} {nome_completo}, titolare di patente n. {pat}, è autorizzato alla guida.'
            e = [(nome_completo, 'private_person'), (pat, 'patente')]
        elif tpl == 10:
            t = f'Intestatario: {nome_completo} — IBAN: {iban} — P.IVA: {piva}.'
            e = [(nome_completo, 'private_person'), (iban, 'iban'), (piva, 'partita_iva')]
        elif tpl == 11:
            t = f"Il {art2} come indicato nel documento d'identità n. {ci}, rilasciato a {comune}."
            e = [(ci, 'carta_identita'), (comune, 'private_address')]
        elif tpl == 12:
            t = f'Contatti: {email} oppure {tel}. Riferimento fiscale: {cf}.'
            e = [(email, 'private_email'), (tel, 'private_phone'), (cf, 'codice_fiscale')]
        elif tpl == 13:
            t = f'Il sig. {cognome} {nome}, {art2} a {comune} il {data_nascita}, ha presentato domanda.'
            e = [(f'{cognome} {nome}', 'private_person'), (comune, 'private_address'),
                 (data_nascita, 'private_date')]
        elif tpl == 14:
            t = f'Passaporto: {pas}. Scadenza: {gg:02d}/{mm:02d}/{aa+10}. Titolare: {nome_completo}.'
            e = [(pas, 'passaporto'), (nome_completo, 'private_person')]
        elif tpl == 15:
            t = f'Dati anagrafici — Nome: {nome_completo} | CF: {cf} | Residenza: {indirizzo}.'
            e = [(nome_completo, 'private_person'), (cf, 'codice_fiscale'),
                 (indirizzo, 'private_address')]
        elif tpl == 16:
            t = f'{art.capitalize()} {nome_completo}, {art3} in {indirizzo}, comunica il proprio IBAN: {iban}.'
            e = [(nome_completo, 'private_person'), (indirizzo, 'private_address'), (iban, 'iban')]
        elif tpl == 17:
            t = f'Numero tessera sanitaria: {ts}. Codice fiscale: {cf}. Paziente: {nome_completo}.'
            e = [(ts, 'tessera_sanitaria'), (cf, 'codice_fiscale'),
                 (nome_completo, 'private_person')]
        elif tpl == 18:
            t = f'La società di {nome_completo}, P.IVA {piva}, ha emesso fattura n. {random.randint(1,999)}/{random.randint(2020,2024)}.'
            e = [(nome_completo, 'private_person'), (piva, 'partita_iva')]
        elif tpl == 19:
            t = f'Autocertificazione: io sottoscritt{"a" if genere=="F" else "o"} {nome_completo}, C.F. {cf}, dichiaro di essere {art2} il {data_nascita} a {comune}.'
            e = [(nome_completo, 'private_person'), (cf, 'codice_fiscale'),
                 (data_nascita, 'private_date'), (comune, 'private_address')]
        elif tpl == 20:
            t = f'Riferimento patente: {pat} — scadenza {gg:02d}/{mm:02d}/{aa+10} — titolare {nome_completo}.'
            e = [(pat, 'patente'), (nome_completo, 'private_person')]
        elif tpl == 21:
            t = f"Per informazioni contattare {nome_completo} al numero {tel} o all'indirizzo {email}."
            e = [(nome_completo, 'private_person'), (tel, 'private_phone'),
                 (email, 'private_email')]
        elif tpl == 22:
            t = f'Codice fiscale del beneficiario: {cf}. Importo bonificato su IBAN {iban}.'
            e = [(cf, 'codice_fiscale'), (iban, 'iban')]
        elif tpl == 23:
            t = f"Documento: carta d'identità n. {ci}, codice fiscale {cf}, rilasciata a {nome_completo}."
            e = [(ci, 'carta_identita'), (cf, 'codice_fiscale'),
                 (nome_completo, 'private_person')]
        elif tpl == 24:
            t = f'Visura catastale: foglio 45, particella 123, intestata a {nome_completo}, C.F. {cf}.'
            e = [(nome_completo, 'private_person'), (cf, 'codice_fiscale')]
        elif tpl == 25:
            t = f'Prescrizione medica per {nome_completo} (CF: {cf}) — tessera sanitaria {ts}.'
            e = [(nome_completo, 'private_person'), (cf, 'codice_fiscale'),
                 (ts, 'tessera_sanitaria')]
        elif tpl == 26:
            t = f'Certificato di residenza: il sig. {nome_completo} risulta residente in {indirizzo} dal {data_nascita}.'
            e = [(nome_completo, 'private_person'), (indirizzo, 'private_address'),
                 (data_nascita, 'private_date')]
        elif tpl == 27:
            t = f'Autocertificazione: {nome_completo}, C.F. {cf}, residente in {indirizzo}, tel. {tel}, email {email}.'
            e = [(nome_completo, 'private_person'), (cf, 'codice_fiscale'),
                 (indirizzo, 'private_address'), (tel, 'private_phone'),
                 (email, 'private_email')]
        elif tpl == 28:
            t = f'Busta paga 03/2024 — Dipendente: {nome_completo} — CF: {cf} — IBAN accredito: {iban}.'
            e = [(nome_completo, 'private_person'), (cf, 'codice_fiscale'), (iban, 'iban')]
        elif tpl == 29:
            t = f'Dichiarazione dei redditi 2024 — Contribuente: {nome_completo} (CF {cf}, P.IVA {piva}).'
            e = [(nome_completo, 'private_person'), (cf, 'codice_fiscale'),
                 (piva, 'partita_iva')]
        elif tpl == 30:
            t = f'Fattura n. {random.randint(1,999)}/2024 emessa a {nome_completo}, P.IVA {piva}, per prestazioni professionali.'
            e = [(nome_completo, 'private_person'), (piva, 'partita_iva')]
        elif tpl == 31:
            t = f'Ricetta bianca: paziente {nome_completo}, nat{"a" if genere=="F" else "o"} il {data_nascita}, tessera sanitaria {ts}.'
            e = [(nome_completo, 'private_person'), (data_nascita, 'private_date'),
                 (ts, 'tessera_sanitaria')]
        elif tpl == 32:
            t = f'Bolletta utenze — Intestatario: {nome_completo} — Indirizzo fornitura: {indirizzo} — Scadenza: {gg:02d}/{mm:02d}/2024.'
            e = [(nome_completo, 'private_person'), (indirizzo, 'private_address')]
        elif tpl == 33:
            t = f'Estratto conto del {data_nascita} — Titolare: {nome_completo} — IBAN: {iban} — CF: {cf}.'
            e = [(data_nascita, 'private_date'), (nome_completo, 'private_person'),
                 (iban, 'iban'), (cf, 'codice_fiscale')]
        elif tpl == 34:
            t = f'Atto notorio: il sottoscritto {nome_completo}, C.F. {cf}, dichiara di essere {art2} a {comune} il {data_nascita}.'
            e = [(nome_completo, 'private_person'), (cf, 'codice_fiscale'),
                 (comune, 'private_address'), (data_nascita, 'private_date')]
        elif tpl == 35:
            t = f'Domanda di iscrizione — Cognome: {cognome} — Nome: {nome} — Data di nascita: {data_nascita} — Email: {email} — Tel: {tel}.'
            e = [(cognome, 'private_person'), (nome, 'private_person'),
                 (data_nascita, 'private_date'), (email, 'private_email'),
                 (tel, 'private_phone')]

        examples.append(make_ex(t, e))

    return examples


# ═══════════════════════════════════════════════════════════════════════
# Step 2 — Dominio legale italiano
# ═══════════════════════════════════════════════════════════════════════

def gen_step2_examples(n=260, negative_rate=0.15):
    """Genera n esempi di atti notarili, contratti, sentenze, procure.

    negative_rate: quota di esempi senza PII (0.15 = 15%).
    """
    examples = []

    for _ in range(n):
        if random.random() < negative_rate:
            examples.append(make_ex(random.choice(NEGATIVES_STEP2), []))
            continue

        nome1, cog1, gen1 = rand_nome()
        nome2, cog2, gen2 = rand_nome()
        nc1 = f'{nome1} {cog1}'
        nc2 = f'{nome2} {cog2}'
        cf1, _, _ = gen_cf(nome1, cog1, gen1)
        cf2, _, _ = gen_cf(nome2, cog2, gen2)
        ci1 = gen_ci()
        piva1 = gen_piva()
        iban1 = gen_iban()
        gg1, mm1, aa1 = rand_data()
        gg2, mm2, aa2 = rand_data()
        com1 = rand_comune()
        com2 = rand_comune()
        data1 = f'{gg1:02d}/{mm1:02d}/{aa1}'
        data2 = f'{gg2:02d}/{mm2:02d}/{aa2}'
        proc = gen_procedimento()
        cat = gen_catastale()
        trib = random.choice(TRIBUNALI)
        sez = random.choice(SEZIONI)
        notaio_nome, notaio_cog, _ = rand_nome()
        notaio = f'{random.choice(NOTAI)} {notaio_nome} {notaio_cog}'
        ruolo1 = random.choice(RUOLI)
        ruolo2 = random.choice([r for r in RUOLI if r != ruolo1])
        tipo_contr = random.choice(TIPI_CONTRATTO)
        importo = f'euro {random.randint(1000, 500000):,}'.replace(',', '.')
        art1 = 'il sig.' if gen1 == 'M' else 'la sig.ra'
        art2 = 'il sig.' if gen2 == 'M' else 'la sig.ra'
        via = random.choice(['Via Roma', 'Via Garibaldi', 'Corso Italia',
                              'Viale Europa', 'Via Manzoni'])
        num_civ = random.randint(1, 200)
        indirizzo1 = f'{via} {num_civ}, {com1}'

        tpl = random.randint(0, 28)

        if tpl == 0:
            t = (f'Avanti a me, {notaio}, sono comparsi: {art1} {nc1}, nato il {data1} a {com1}, '
                 f'codice fiscale {cf1}, e {art2} {nc2}, nato il {data2} a {com2}, '
                 f'codice fiscale {cf2}, i quali convengono quanto segue.')
            e = [(nc1, 'private_person'), (data1, 'private_date'), (com1, 'private_address'), (cf1, 'codice_fiscale'),
                 (nc2, 'private_person'), (data2, 'private_date'), (com2, 'private_address'), (cf2, 'codice_fiscale')]
        elif tpl == 1:
            t = (f'Il {trib}, {sez}, nel procedimento {proc}, '
                 f'tra {nc1} ({ruolo1}) e {nc2} ({ruolo2}), ha emesso la seguente sentenza.')
            e = [(proc, 'numero_procedimento'), (nc1, 'parte_in_causa'), (nc2, 'parte_in_causa')]
        elif tpl == 2:
            t = (f'Con {tipo_contr} del {data1}, {art1} {nc1}, C.F. {cf1}, residente in {indirizzo1}, '
                 f"cede a {art2} {nc2} l'immobile sito in {com2}, {cat}.")
            e = [(data1, 'private_date'), (nc1, 'private_person'), (cf1, 'codice_fiscale'),
                 (indirizzo1, 'private_address'), (nc2, 'private_person'), (cat, 'riferimento_catastale')]
        elif tpl == 3:
            t = (f'Il prezzo della compravendita è stabilito in {importo}, '
                 f"da versarsi tramite bonifico bancario sull'IBAN {iban1} intestato a {nc1}.")
            e = [(iban1, 'iban'), (nc1, 'private_person')]
        elif tpl == 4:
            t = (f'Con procura speciale, {art1} {nc1}, nato il {data1} a {com1}, C.F. {cf1}, '
                 f'delega {art2} {nc2} a rappresentarlo in ogni sede giudiziaria.')
            e = [(nc1, 'private_person'), (data1, 'private_date'), (com1, 'private_address'),
                 (cf1, 'codice_fiscale'), (nc2, 'private_person')]
        elif tpl == 5:
            t = (f"L'immobile identificato catastalmente come {cat}, sito nel Comune di {com1}, "
                 f'è di proprietà di {nc1} per la quota di 1/1.')
            e = [(cat, 'riferimento_catastale'), (com1, 'private_address'), (nc1, 'private_person')]
        elif tpl == 6:
            t = (f'Nella causa {proc} pendente avanti al {trib}, '
                 f'il {ruolo1} {nc1} ha depositato atto di citazione in data {data1}.')
            e = [(proc, 'numero_procedimento'), (nc1, 'parte_in_causa'), (data1, 'private_date')]
        elif tpl == 7:
            t = (f'{art1.capitalize()} {nc1}, titolare di P.IVA {piva1}, con sede legale in {indirizzo1}, '
                 f'ha stipulato {tipo_contr} con {nc2}.')
            e = [(nc1, 'private_person'), (piva1, 'partita_iva'),
                 (indirizzo1, 'private_address'), (nc2, 'private_person')]
        elif tpl == 8:
            t = (f'Il locatore {nc1} (C.F. {cf1}) concede in locazione al conduttore {nc2} (C.F. {cf2}) '
                 f"l'immobile sito in {indirizzo1}, al canone mensile di {importo}.")
            e = [(nc1, 'private_person'), (cf1, 'codice_fiscale'),
                 (nc2, 'private_person'), (cf2, 'codice_fiscale'),
                 (indirizzo1, 'private_address')]
        elif tpl == 9:
            t = (f'Rilevato che con ordinanza del {data1} il {trib} ha disposto la notifica '
                 f'del ricorso {proc} nei confronti di {nc1}.')
            e = [(data1, 'private_date'), (proc, 'numero_procedimento'), (nc1, 'parte_in_causa')]
        elif tpl == 10:
            t = (f"Si dichiara che {art1} {nc1}, identificato mediante carta d'identità n. {ci1}, "
                 f"C.F. {cf1}, è il legittimo proprietario dell'immobile identificato come {cat}.")
            e = [(nc1, 'private_person'), (ci1, 'carta_identita'),
                 (cf1, 'codice_fiscale'), (cat, 'riferimento_catastale')]
        elif tpl == 11:
            t = (f'Omissis — Il Giudice, letti gli atti del procedimento {proc}, '
                 f'udite le parti {nc1} e {nc2}, così decide: ...')
            e = [(proc, 'numero_procedimento'), (nc1, 'parte_in_causa'), (nc2, 'parte_in_causa')]
        elif tpl == 12:
            t = (f'Il mutuatario {nc1}, C.F. {cf1}, residente in {com1}, '
                 f"si obbliga al rimborso di {importo} sull'IBAN {iban1}.")
            e = [(nc1, 'private_person'), (cf1, 'codice_fiscale'),
                 (com1, 'private_address'), (iban1, 'iban')]
        elif tpl == 13:
            t = (f"Il difensore dell'{ruolo1} {nc1} avv. {nome2} {cog2}, "
                 f'con studio in {com2}, deposita la seguente memoria difensiva nel proc. {proc}.')
            e = [(nc1, 'parte_in_causa'), (f'{nome2} {cog2}', 'private_person'),
                 (com2, 'private_address'), (proc, 'numero_procedimento')]
        elif tpl == 14:
            t = (f'Atto di donazione: il donante {nc1} (nato il {data1}, C.F. {cf1}) '
                 f'trasferisce a titolo gratuito al donatario {nc2} il bene censito come {cat}.')
            e = [(nc1, 'private_person'), (data1, 'private_date'), (cf1, 'codice_fiscale'),
                 (nc2, 'private_person'), (cat, 'riferimento_catastale')]
        elif tpl == 15:
            t = (f'Con decreto del {data1}, il {trib} ha omologato il piano di rientro '
                 f'del debitore {nc1}, P.IVA {piva1}, nel proc. {proc}.')
            e = [(data1, 'private_date'), (nc1, 'private_person'),
                 (piva1, 'partita_iva'), (proc, 'numero_procedimento')]
        elif tpl == 16:
            t = (f"Verbale di udienza del {data1} — Presente l'{ruolo1} {nc1} "
                 f"e il {ruolo2} {nc2}. Il giudice rinvia all'udienza del {data2}.")
            e = [(data1, 'private_date'), (nc1, 'parte_in_causa'),
                 (nc2, 'parte_in_causa'), (data2, 'private_date')]
        elif tpl == 17:
            t = (f'Si costituisce in giudizio {art1} {nc1}, C.F. {cf1}, residente in {com1}, '
                 f'come da procura in calce al presente atto nel proc. n. {proc}.')
            e = [(nc1, 'private_person'), (cf1, 'codice_fiscale'),
                 (com1, 'private_address'), (proc, 'numero_procedimento')]
        elif tpl == 18:
            t = (f'Il bene immobile sito in {com1}, {cat}, è gravato da ipoteca '
                 f'a favore di {nc1} per un importo di {importo}.')
            e = [(com1, 'private_address'), (cat, 'riferimento_catastale'),
                 (nc1, 'private_person')]
        elif tpl == 19:
            t = (f'Sentenza n. {random.randint(100,9999)}/{random.randint(2018,2024)} — Nel procedimento di divorzio '
                 f'tra {nc1} e {nc2}, il {trib} dichiara sciolto il matrimonio.')
            e = [(nc1, 'parte_in_causa'), (nc2, 'parte_in_causa')]
        elif tpl == 20:
            t = (f'Decreto ingiuntivo n. {random.randint(100,9999)}/{random.randint(2020,2024)}: '
                 f'si ingiunge a {nc1}, C.F. {cf1}, il pagamento di {importo} in favore di {nc2}.')
            e = [(nc1, 'parte_in_causa'), (cf1, 'codice_fiscale'), (nc2, 'parte_in_causa')]
        elif tpl == 21:
            t = (f'Atto di pignoramento: si procede al pignoramento dei beni di {nc1}, C.F. {cf1}, '
                 f'residente in {indirizzo1}, su istanza di {nc2} nel proc. {proc}.')
            e = [(nc1, 'parte_in_causa'), (cf1, 'codice_fiscale'),
                 (indirizzo1, 'private_address'), (nc2, 'parte_in_causa'),
                 (proc, 'numero_procedimento')]
        elif tpl == 22:
            t = (f'Testamento pubblico — Io sottoscritto {nc1}, nato il {data1} a {com1}, C.F. {cf1}, '
                 f'nomino mio erede universale {nc2}.')
            e = [(nc1, 'private_person'), (data1, 'private_date'),
                 (com1, 'private_address'), (cf1, 'codice_fiscale'),
                 (nc2, 'private_person')]
        elif tpl == 23:
            t = (f'Verbale di conciliazione del {data1} — Le parti {nc1} e {nc2}, nel procedimento {proc}, '
                 f'dichiarano di conciliare la controversia alle seguenti condizioni.')
            e = [(data1, 'private_date'), (nc1, 'parte_in_causa'),
                 (nc2, 'parte_in_causa'), (proc, 'numero_procedimento')]
        elif tpl == 24:
            t = (f'Fideiussione bancaria n. {random.randint(1000,9999)}: il fideiussore {nc1} '
                 f'(C.F. {cf1}) garantisce il pagamento di {importo} in favore del creditore {nc2}.')
            e = [(nc1, 'private_person'), (cf1, 'codice_fiscale'), (nc2, 'private_person')]
        elif tpl == 25:
            t = (f'Atto di precetto: si intima a {nc1}, residente in {indirizzo1}, '
                 f'il pagamento entro 10 giorni della somma di {importo} come da titolo esecutivo nel proc. {proc}.')
            e = [(nc1, 'parte_in_causa'), (indirizzo1, 'private_address'),
                 (proc, 'numero_procedimento')]
        elif tpl == 26:
            t = (f'Nomina di CTU: il {trib} nomina consulente tecnico il dott. {notaio_nome} {notaio_cog} '
                 f"per l'espletamento delle operazioni peritali nel procedimento {proc}.")
            e = [(f'{notaio_nome} {notaio_cog}', 'private_person'),
                 (proc, 'numero_procedimento')]
        elif tpl == 27:
            t = (f'Verbale di assemblea della società rappresentata da {nc1}, P.IVA {piva1}, '
                 f'tenutasi il {data1} presso la sede legale in {indirizzo1}, presente il socio {nc2}.')
            e = [(nc1, 'private_person'), (piva1, 'partita_iva'),
                 (data1, 'private_date'), (indirizzo1, 'private_address'),
                 (nc2, 'private_person')]
        elif tpl == 28:
            t = (f"Ricorso in appello: l'{ruolo1} {nc1}, C.F. {cf1}, impugna la sentenza n. "
                 f'{random.randint(100,9999)}/{random.randint(2015,2023)} del {trib} nel proc. {proc}.')
            e = [(nc1, 'parte_in_causa'), (cf1, 'codice_fiscale'),
                 (proc, 'numero_procedimento')]

        examples.append(make_ex(t, e))

    return examples


# ═══════════════════════════════════════════════════════════════════════
# Validazione e statistiche
# ═══════════════════════════════════════════════════════════════════════

def validate_spans(data, name='dataset', verbose=False):
    """Ritorna il numero di errori negli offset. Stampa i problemi trovati."""
    errors = 0
    for i, ex in enumerate(data):
        text = ex.get('text', '')
        spans = ex.get('spans', {})
        if not isinstance(spans, dict):
            print(f'  ❌ [{name}][{i}] spans non è un dict: {type(spans).__name__}')
            errors += 1
            continue
        for label, offsets in spans.items():
            if not isinstance(offsets, list):
                print(f'  ❌ [{name}][{i}][{label}] offsets non è una lista')
                errors += 1
                continue
            for pair in offsets:
                if not (isinstance(pair, (list, tuple)) and len(pair) == 2):
                    print(f'  ❌ [{name}][{i}][{label}] span non è [start,end]: {pair}')
                    errors += 1
                    continue
                s, e = pair
                if e > len(text) or s >= e or s < 0 or text[s:e] == '':
                    print(f'  ❌ [{name}][{i}][{label}] offset errato [{s},{e}] '
                          f'in {text[:60]!r}')
                    errors += 1
                elif verbose:
                    print(f'  [{name}][{i}][{label}] {text[s:e]!r} ({s}:{e})')
    return errors


def label_distribution(data):
    """Conta gli span per label nel dataset. Ritorna un Counter."""
    counts = Counter()
    for ex in data:
        for label, offsets in ex.get('spans', {}).items():
            counts[label] += len(offsets)
    return counts


# ═══════════════════════════════════════════════════════════════════════
# Convenience: build all datasets in one call
# ═══════════════════════════════════════════════════════════════════════

def build_complete_dataset(
    n_step1=(1500, 250, 250),
    n_step2=(1000, 200, 200),
    seed=42,
    negative_rate=0.15,
):
    """Genera dataset completo con split train/val/test per entrambi gli step.

    n_step1, n_step2: tuple (train, val, test) con le dimensioni.
    seed: riproducibilità.
    negative_rate: quota di esempi negativi (nessun PII).

    Ritorna dict annidato:
        {
          'label_space': LABEL_SPACE,
          'step1': {'train': [...], 'val': [...], 'test': [...]},
          'step2': {'train': [...], 'val': [...], 'test': [...]},
        }

    Strategia split: genera un pool totale, shuffle deterministico, poi divide.
    Questo garantisce distribuzione IID tra i tre split.
    """
    n1_tr, n1_va, n1_te = n_step1
    n2_tr, n2_va, n2_te = n_step2

    # Step 1
    random.seed(seed)
    total1 = n1_tr + n1_va + n1_te
    pool1 = gen_step1_examples(total1, negative_rate=negative_rate)
    random.shuffle(pool1)
    s1 = {
        'train': pool1[:n1_tr],
        'val':   pool1[n1_tr:n1_tr + n1_va],
        'test':  pool1[n1_tr + n1_va:],
    }

    # Step 2 (seed offset per indipendenza)
    random.seed(seed + 1000)
    total2 = n2_tr + n2_va + n2_te
    pool2 = gen_step2_examples(total2, negative_rate=negative_rate)
    random.shuffle(pool2)
    s2 = {
        'train': pool2[:n2_tr],
        'val':   pool2[n2_tr:n2_tr + n2_va],
        'test':  pool2[n2_tr + n2_va:],
    }

    return {'label_space': LABEL_SPACE, 'step1': s1, 'step2': s2}


def write_jsonl(data, path):
    """Scrive una lista di esempi in un file .jsonl (una riga per esempio)."""
    import json
    import os
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for ex in data:
            f.write(json.dumps(ex, ensure_ascii=False) + '\n')
    return os.path.getsize(path)


def write_splits_to_disk(bundle, base_dir='datasets'):
    """Scrive i 6 file .jsonl da un bundle di build_complete_dataset.

    Produce:
        {base_dir}/step1_train.jsonl, step1_val.jsonl, step1_test.jsonl
        {base_dir}/step2_train.jsonl, step2_val.jsonl, step2_test.jsonl
    """
    import os
    paths = {}
    for step_name, splits in (('step1', bundle['step1']), ('step2', bundle['step2'])):
        for split_name, data in splits.items():
            path = os.path.join(base_dir, f'{step_name}_{split_name}.jsonl')
            size = write_jsonl(data, path)
            paths[f'{step_name}_{split_name}'] = (path, len(data), size)
    return paths

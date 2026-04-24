"""
Stima memoria necessaria per fine-tuning di openai/privacy-filter su M-series.

Modello: MoE (Mixture of Experts)
- 8 layer transformer
- 128 expert per layer, 4 attivi per token
- hidden_size=640, intermediate_size=640
- vocab=200064, num_labels=33
- param_dtype=bfloat16 (checkpoint), ma training in fp32 (forzato da opf/_train/runner.py:679)
"""

# ── Config modello (da ~/.opf/privacy_filter/config.json) ──
HIDDEN      = 640
INTERMEDIATE = 640
N_LAYERS    = 8
N_EXPERTS   = 128
EXPERTS_PER_TOKEN = 4
VOCAB       = 200064
N_LABELS    = 33
N_KV_HEADS  = 2
HEAD_DIM    = 64
KV_HIDDEN   = N_KV_HEADS * HEAD_DIM  # 128

FILE_SIZE_GB = 2.8  # safetensors su disco


def count_params():
    embed = VOCAB * HIDDEN                          # 128M
    attn_per_layer = (HIDDEN * HIDDEN               # Q
                      + HIDDEN * KV_HIDDEN          # K
                      + HIDDEN * KV_HIDDEN          # V
                      + HIDDEN * HIDDEN)            # O
    moe_per_layer = N_EXPERTS * 2 * HIDDEN * INTERMEDIATE   # mlp1 + mlp2
    router_per_layer = HIDDEN * N_EXPERTS           # routing
    norms_per_layer = 4 * HIDDEN                    # layer norms
    per_layer = attn_per_layer + moe_per_layer + router_per_layer + norms_per_layer
    head = N_LABELS * HIDDEN
    total = embed + N_LAYERS * per_layer + head
    return {
        'embed': embed,
        'attn_per_layer': attn_per_layer,
        'moe_per_layer': moe_per_layer,
        'per_layer_total': per_layer,
        'total_params': total,
        'total_B': total / 1e9,
    }


def activation_memory_gb(batch, seq_len):
    """Stima memoria activation per forward+backward.

    L'MoE fa gather dei pesi expert → tensore denso (batch*seq*experts_per_token, H, I) × bf16.
    Più: activation standard transformer (attention scores, MLP hidden, residual).
    """
    # MoE gather (il collo di bottiglia su MPS senza Triton)
    # Per ogni token attivo, expert_per_token copie dei pesi hidden×intermediate (bf16)
    moe_gather_per_layer_gb = (
        batch * seq_len * EXPERTS_PER_TOKEN * HIDDEN * INTERMEDIATE * 2
    ) / 1e9

    # Activation standard (approssimazione semplificata)
    # attention: batch×seq×seq (scores) + batch×seq×hidden (QKV proj)
    attn_act_gb = (batch * seq_len * seq_len * 4 +         # fp32 attention scores
                   batch * seq_len * HIDDEN * 4 * 4) / 1e9   # QKV+O projections
    # MLP hidden: batch × seq × intermediate × 4 bytes (fp32)
    mlp_act_gb = batch * seq_len * INTERMEDIATE * 4 / 1e9

    per_layer = moe_gather_per_layer_gb + attn_act_gb + mlp_act_gb
    # Non tutti i layer memorizzano activation contemporaneamente, ma per gradient checkpointing
    # servono all'incirca tutti durante il backward. Sommo su N_LAYERS come upper bound.
    return per_layer * N_LAYERS


def training_memory_estimate(batch, seq_len, model_dtype='fp32'):
    """Memoria totale (GB) per uno step di training.

    opf forza fp32 durante training (opf/_train/runner.py:679), anche con --output-param-dtype bf16.
    """
    params = count_params()
    n = params['total_params']

    bytes_per_param = 4 if model_dtype == 'fp32' else 2

    # Stato statico (persiste tra step)
    weights_gb   = n * bytes_per_param / 1e9
    grads_gb     = n * bytes_per_param / 1e9               # gradienti stesso dtype
    adam_gb      = n * 8 / 1e9                              # Adam: momentum(fp32) + variance(fp32)

    # Dinamico (per step)
    act_gb       = activation_memory_gb(batch, seq_len)
    workspace_gb = 0.5                                      # overhead MPS/PyTorch

    total = weights_gb + grads_gb + adam_gb + act_gb + workspace_gb

    return {
        'weights_gb': weights_gb,
        'grads_gb': grads_gb,
        'adam_gb': adam_gb,
        'activations_gb': act_gb,
        'workspace_gb': workspace_gb,
        'total_gb': total,
    }


def print_report():
    params = count_params()
    print('═' * 70)
    print('MODELLO openai/privacy-filter')
    print('═' * 70)
    print(f'  Parametri totali:  {params["total_B"]:.2f} B ({params["total_params"]/1e6:.0f} M)')
    print(f'  File safetensors:  {FILE_SIZE_GB} GB (bf16 su disco)')
    print(f'  Breakdown:')
    print(f'    embedding:         {params["embed"]/1e6:>6.1f} M')
    print(f'    MoE per layer:     {params["moe_per_layer"]/1e6:>6.1f} M  ({N_EXPERTS} experts × 2 matrici × {HIDDEN}×{INTERMEDIATE})')
    print(f'    Attn per layer:    {params["attn_per_layer"]/1e6:>6.1f} M')
    print(f'    Totale × {N_LAYERS} layer: {(N_LAYERS*params["per_layer_total"])/1e6:>6.1f} M')
    print()
    print(f'  ⚠️  opf forza fp32 durante training (linea 679 runner.py)')
    print(f'      Anche con --output-param-dtype bf16, i pesi in RAM sono fp32')
    print()

    print('═' * 70)
    print('STIMA MEMORIA PER CONFIGURAZIONE')
    print('═' * 70)
    print(f'{"batch":>6} {"seq":>5} {"weights":>9} {"grads":>8} {"adam":>7} '
          f'{"activ":>7} {"TOT":>7} {"M4 16GB":>10} {"M4 Pro 24GB":>12} {"M4 Max 48GB":>12}')
    print('─' * 100)

    limits = {'M4 16GB': 11, 'M4 Pro 24GB': 17, 'M4 Max 48GB': 35}  # MPS limits con watermark default
    configs = [
        (4, 512), (4, 256), (4, 128),
        (2, 512), (2, 256), (2, 128),
        (1, 512), (1, 256), (1, 128), (1, 64),
    ]

    for batch, seq in configs:
        r = training_memory_estimate(batch, seq, model_dtype='fp32')
        marks = []
        for mac, lim in limits.items():
            marks.append('✅' if r['total_gb'] <= lim else '❌')
        print(f'{batch:>6} {seq:>5} {r["weights_gb"]:>8.1f}G {r["grads_gb"]:>7.1f}G '
              f'{r["adam_gb"]:>6.1f}G {r["activations_gb"]:>6.1f}G {r["total_gb"]:>6.1f}G '
              f'{marks[0]:>10} {marks[1]:>12} {marks[2]:>12}')

    print()
    print('═' * 70)
    print('RACCOMANDAZIONI PER M4 16 GB')
    print('═' * 70)
    print('  ❌ batch=4 (default): ~25-40 GB richiesti → OOM come hai visto')
    print('  ⚠️  batch=1 seq=128: ~24 GB → ancora sopra il limite MPS (~11 GB)')
    print('  → Il modello richiede PIÙ RAM di quella che hai')
    print()
    print('  OPZIONI:')
    print('   1) Training su CPU (lento ma funziona): --device cpu')
    print('      Tempo stimato: ore per 1500 esempi. Ma non va OOM.')
    print()
    print('   2) Riduci BATCH_SIZE a 1 + aumenta gradient accumulation:')
    print('      --batch-size 1 --grad-accum-steps 4')
    print('      Potrebbe comunque non bastare: Adam+gradient occupano ~17 GB')
    print()
    print('   3) Alza il watermark MPS e accetta swap SSD (molto lento):')
    print('      PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 opf train ...')
    print()
    print('   4) Usa Google Colab (T4 16GB VRAM dedicata) per il training,')
    print('      poi scarichi il checkpoint finetuned sul tuo M4 per inferenza.')


if __name__ == '__main__':
    print_report()

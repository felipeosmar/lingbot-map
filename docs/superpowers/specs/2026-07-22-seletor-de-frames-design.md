# Design — Seletor de frames (pré-processador com detector de cena)

**Data:** 2026-07-22
**Status:** aprovado (aguardando revisão do spec)
**Contexto:** LingBot-Map, reconstrução 3D streaming de vídeo monocular.

## 1. Problema

A extração de frames atual do projeto (`demo_render/batch_demo.py` →
`load_images_from_video`) faz apenas **decimação temporal uniforme**
(`--fps` / `--target_frames` / `--stride`). Isso ignora a qualidade e o conteúdo
de cada frame, o que causou dois problemas observados na reconstrução de um vídeo
handheld de uma sala:

- **Borrão no meio da sala (modo streaming):** em panorâmicas rápidas a ~5 fps, os
  frames têm muito *motion blur* e pouca sobreposição → a atenção cross-frame do
  modelo não encontra correspondência → pose/profundidade degeneram no trecho.
- **Necessidade de caber no limite do streaming:** o modo `streaming` (correção de
  deriva global) só suporta ≤ 1024 frames (tabela do 3D RoPE, `max_frame_num`).

Queremos um **pré-processador standalone** que selecione melhor os frames *antes* de
entregá-los ao modelo, entregando um conjunto **enxuto, nítido e bem distribuído**,
respeitando um **orçamento de frames**.

## 2. Objetivo

Selecionar frames com base em **qualidade + conteúdo**, respeitando um **orçamento**:

- descartar frames borrados (motion blur) e mal-expostos;
- descartar redundância (câmera quase parada) e evitar saltos grandes;
- priorizar cobertura de conteúdo novo;
- garantir que o total final fique próximo de um alvo (ex.: ~900 para caber no streaming).

**Não-objetivos (YAGNI):** detecção de corte de cena (o vídeo é tomada contínua);
seleção baseada em modelo/embedding/GPU; qualquer alteração no modelo ou no solver de
trajetória.

## 3. Arquitetura

**Pré-processador standalone**, sem tocar no código do modelo.

- **Módulos:**
  - `preprocess/frame_metrics.py` — funções puras de métrica (sem estado, testáveis diretamente).
  - `preprocess/frame_selector.py` — portão de qualidade, seleção, orçamento, escrita e CLI `main()`.
- **Dependências:** `opencv-python`, `numpy`, `tqdm` (já são dependências do repo).
  **Sem GPU, sem PyTorch, sem o pacote do modelo.**
- **Fluxo de uso:**
  ```bash
  python preprocess/frame_selector.py \
    --video_path VID.mp4 --output_dir frames_sel/ --budget 900
  # em seguida, o pipeline existente consome a pasta:
  python demo_render/batch_demo.py --image_folder frames_sel/ --mode streaming ...
  ```
- **Entrada:** um vídeo (`--video_path`) ou uma pasta de imagens (`--image_folder`).
- **Saída:** pasta com os frames selecionados + manifesto CSV + relatório TXT
  (ver seção 6).

O acoplamento é só o contrato de pasta de frames que o `batch_demo` já suporta
(`list_image_paths` lê `.jpg,.jpeg,.png` em ordem de nome).

## 4. Interface (CLI)

| Flag | Default | Descrição |
|---|---|---|
| `--video_path` / `--image_folder` | — | Fonte (um dos dois, obrigatório). |
| `--output_dir` | — | Pasta de saída (obrigatório). |
| `--budget` | `900` | Alvo de nº de frames selecionados. |
| `--sample_fps` | `0` (=todos) | Analisar 1 a cada N para acelerar (0 = todos os frames da fonte). |
| `--analysis_width` | `320` | Largura do grayscale usado nas métricas (velocidade). |
| `--motion_threshold` | `auto` | Limiar de movimento acumulado; `auto` = busca para atingir o orçamento. |
| `--max_gap` | `45` | Máx. de frames consecutivos sem seleção antes de forçar um keep. |
| `--min_entropy` | `auto` | Piso de entropia para o portão de exposição. |
| `--novelty_floor` | `0.02` | Distância mínima de histograma vs último selecionado. |
| `--contact_sheet` | `false` | Gera mosaico PNG dos frames escolhidos. |

## 5. Algoritmo

Todas as métricas são calculadas sobre **grayscale reduzido** (`--analysis_width`).

### Passo 1 — métricas por frame + portão de qualidade
Varre a fonte uma vez, computando por frame:
- **Nitidez** `sharpness = var(Laplacian(gray))`.
- **Exposição** `brightness = mean(gray)`, `clipped = frac(pixels < 8 ou > 247)`,
  `entropy = entropia do histograma`.

**Portão de qualidade (descarte imediato):** frames severamente mal-expostos
(muito escuros/estourados) ou quase sem informação (`entropy < min_entropy`).
O **blur não usa limiar absoluto** — é tratado por comparação **local** no Passo 2
(a variância do Laplaciano depende do conteúdo, então comparação relativa é mais robusta).

### Passo 2 — seleção sequencial por movimento
Sobre os frames que passaram no portão, em ordem temporal:
- **Fluxo** `flow[i]` = magnitude média do fluxo óptico Farneback entre frames
  consecutivos sobreviventes `i-1 → i`. **Calculado uma única vez** e mantido em cache.
- Mantém `last_selected` e um acumulador `motion_accum` (soma de `flow`).
- Quando `motion_accum ≥ motion_threshold`, fecha um **segmento** e seleciona o
  frame **mais nítido** do segmento (anti-blur); zera o acumulador a partir dele.
- **Novidade:** se o histograma do candidato for quase idêntico ao `last_selected`
  (distância `< novelty_floor`), pula (evita duplicar ao varrer o mesmo canto).
- **Gap máximo:** se `max_gap` frames passarem sem seleção (câmera parada), força a
  seleção do mais nítido do intervalo (`reason = max_gap`).
- O primeiro frame válido é sempre selecionado (`reason = forced_first`).

### Orçamento
`motion_threshold = auto`: como `flow` está em cache, re-integrar o Passo 2 com outro
limiar é barato. Faz **busca binária** no `motion_threshold` até o total cair em
`[0.85·budget, budget]`. Se a fonte tiver ≤ `budget` frames válidos, mantém todos.

## 6. Saídas (em `--output_dir`)

- `frame_000000.png`, `frame_000001.png`, … — frames selecionados, **renomeados
  sequencialmente** em ordem temporal, na **resolução original** (o `batch_demo`
  faz o resize).
- `selection_manifest.csv` — uma linha por frame selecionado:
  `out_name, src_frame_idx, timestamp_s, sharpness, motion_accum, brightness, entropy, novelty, reason`.
- `selection_report.txt` — resumo: total da fonte → sobreviventes do portão →
  selecionados; `motion_threshold` final; contagem de descartes por
  blur/exposição/redundância; fps efetivo médio.
- `contact_sheet.png` (se `--contact_sheet`) — mosaico dos frames escolhidos.

## 7. Tratamento de erros / casos de borda

- **Vídeo/pasta ilegível** → erro claro, sem stack trace cru.
- **Portão reprova tudo** → relaxa os limiares de exposição automaticamente e avisa;
  nunca retorna vazio.
- **`budget` ≥ frames válidos** → mantém todos (não força descarte).
- **`budget` pequeno demais vs `max_gap`** → prioriza o orçamento e **registra no
  relatório** o que foi cortado (proibido truncar silenciosamente).

## 8. Componentes (unidades isoláveis)

- **`preprocess/frame_metrics.py`** (funções puras): `sharpness(gray)`,
  `exposure(gray) → (brightness, clipped, entropy)`, `flow_magnitude(gray_a, gray_b)`,
  `histogram(gray)`, `histogram_distance(h_a, h_b)`. Sem estado, testáveis diretamente.
- **`preprocess/frame_selector.py`**:
  - `quality_gate(frames_metrics) → índices sobreviventes`.
  - `select_by_motion(survivors, flow_cache, threshold, max_gap, novelty) → índices selecionados`.
  - `fit_budget(...) → threshold` (busca binária que chama `select_by_motion`).
  - `main()` — orquestra: decodifica, computa métricas (via `frame_metrics`), roda
    gate + seleção + orçamento, escreve saídas.

Cada unidade tem propósito único e interface clara; o fluxo de dados é
`fonte → métricas → gate → seleção → orçamento → escrita`.

## 9. Testes (`tests/test_frame_selector.py`, pytest)

**Unitários das métricas:**
- imagem sintética nítida vs. borrada (Gaussian blur) → `sharpness` cai;
- imagem deslocada por translação conhecida → `flow_magnitude` > 0 e cresce com o deslocamento;
- imagens escura/estourada → `exposure` sinaliza ruim;
- histogramas idênticos → `histogram_distance ≈ 0`; muito diferentes → distância alta.

**Integração (vídeo sintético pequeno, dezenas de frames):**
- alguns frames deliberadamente borrados → são descartados;
- trecho estático → gera poucas seleções;
- contagem final respeita `budget`;
- ordem temporal preservada nos nomes de saída.

**Orçamento:** a busca no `motion_threshold` aterrissa em `[0.85·budget, budget]`.

## 10. Impacto esperado

Entregar ao modelo um conjunto nítido e bem espaçado deve reduzir o borrão por
motion blur (Passo 1 + "mais nítido do segmento") e permitir usar `streaming`
(deriva global, sem paredes duplas) dentro do orçamento de 1024 frames — atacando as
duas falhas observadas. Não corrige a ausência de bundle adjustment global do modelo,
mas melhora a qualidade da entrada, que é a variável sob nosso controle.

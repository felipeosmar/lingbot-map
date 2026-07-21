# Relatório de Testes — LingBot-Map

Registro dos testes de deploy, profiling e exploração do **LingBot-Map** (Geometric
Context Transformer para reconstrução 3D em streaming), executados em **2026-07-21**.

---

## 1. Ambiente de execução

| Item | Valor |
|---|---|
| Máquina | **ctrob-workstation** (SENAI Automação) |
| GPU | **NVIDIA RTX PRO 6000 Blackwell Max-Q** — 97,9 GB VRAM |
| CPU | 32 cores |
| RAM | 125 GB |
| Acesso | SSH via Tailscale (`100.121.227.88`) |
| PyTorch | 2.8.0 + CUDA 12.8 |
| Atenção | FlashInfer 0.6.15.post1 (paged KV cache) |
| Ambiente Python | `uv venv` (Python 3.10) em `/home/ctrob/lingbot-env` |
| Checkpoint | `lingbot-map.pt` (balanced, 4,63 GB) + `skyseg.onnx` (176 MB) |

---

## 2. Orquestração do deploy (Frota Claude / "Central")

O deploy foi executado como **job distribuído na Frota Claude** (fila NATS JetStream),
não manualmente:

- Repositório publicado como bare em `/home/ctrob/lingbot-map.git`; briefing em `MISSION.md`.
- **Roteamento por capability:** worker transiente iniciado na ctrob com `--capabilities gpu`
  (o GB10 anuncia apenas `linux`), garantindo que o job rodasse na GPU correta.
- O worker clonou o repo, criou o venv, instalou dependências, baixou o checkpoint e
  lançou o viser como **serviço systemd `--user` persistente** (sobrevive ao fim do job).
- Acompanhamento via Loki (`{app="fleet"}`): `job.claimed → started → completed`.

---

## 3. Consumo de recursos (processo do viser, cena oxford)

| Recurso | Consumo do LingBot | Total | % |
|---|---|---|---|
| **CPU** | ~67% de 1 core (147 threads) | 32 cores | ~2% |
| **RAM** | 5,96 GB (RSS) / 5,8 GB (PSS) | 125 GB | ~4,5% |
| **VRAM** | 17,6 GB (17.978 MiB) | 97,9 GB | ~18% |
| **GPU compute** | 0% util · 24,7 W · 36 °C (idle após inferência) | — | — |
| **Armazenamento** | **~12,3 GB** | — | — |

**Detalhe do armazenamento:**
- `lingbot-env` (venv PyTorch + deps): 7,4 GB
- `lingbot-models` (checkpoint + sky-seg): 4,5 GB
- bare repo: 350 MB · cache HuggingFace: 90 MB

> **Nota:** a GPU fica ociosa (0%) enquanto se navega na cena — a renderização 3D é
> **client-side** (WebGL no navegador da TV). A GPU só trabalha na inferência inicial.
> O ~67% de CPU constante vem do busy-loop do viser (`while True: sleep(0.001)`).

---

## 4. Profiling de velocidade (FPS) — 518×378, flashinfer/bf16

Medido com `gct_profile.py` (streaming de 1000 frames):

| Métrica | Valor |
|---|---|
| **Throughput global** | **13,3 FPS** (75 ms/frame) |
| Início da sequência (frame ~100) | 13,7 FPS |
| Meio (frame ~500) | 13,0 FPS |
| Fim (frame ~900) | 12,5 FPS |
| Phase 1 (8 scale frames) | 171 ms |

O FPS decai suavemente ao longo da sequência conforme o KV cache cresce.

---

## 5. Profiling de VRAM vs comprimento da sequência

Medido com `scripts/benchmark_gct_memory.py` (518×378, flashinfer):

| Frames | VRAM (peak alloc) | Tempo |
|---|---|---|
| 64 | 17,81 GiB | 3,7 s |
| 128 | 17,81 GiB | 9,0 s |
| 256 | 17,90 GiB | 20,0 s |
| 512 | 18,08 GiB | 42,8 s |
| 1.024 | 18,36 GiB | 90,0 s |
| 2.048 | 18,91 GiB | 188,8 s |
| 4.096 | **20,00 GiB** | 628,0 s |

**Insight principal 🎯** — a VRAM é **quase plana** com o tamanho da sequência: de 64 → 4.096
frames (64× mais dados), o footprint sobe apenas **+2,2 GiB**. É o **paged KV cache attention**
em ação — a maior parte dos ~17,8 GiB é o modelo base; cada duplicação de frames adiciona só
~0,5 GiB. Na prática: o gargalo para sequências longas é **tempo, não memória**, o que valida
o claim do paper de sequências >10.000 frames.

---

## 6. Cenas reconstruídas e testadas

As 4 cenas de exemplo foram reconstruídas e exibidas, cada uma num viser dedicado:

| Cena | Porta | Sky mask | Frames | Conteúdo |
|---|---|---|---|---|
| **oxford** | 8080 | sim | 320 | pátio universitário |
| **courthouse** | 8081 | sim | 286 | prédio + estacionamento |
| **university** | 8082 | sim | — | passarela nevada (inverno) |
| **loop** | 8083 | não | — | corredor interno (loop-closure) |

Cada viser roda como serviço systemd `--user` (`lingbot-<cena>.service`), consumindo ~15 GiB
de VRAM. Com as 4 simultâneas: ~61 GB / 97,9 GB usados.

---

## 7. Pipeline de exibição na TV2 (Pesquisa 2)

O visualizador foi exibido numa **Android TV** (`4K SMART TV PH`, tag `pesquisa2`):

- **webview_server** (FastAPI) fala WebSocket com o app `com.felipe.webviewtv` na TV.
- Trocar de cena = `POST /api/webviews/{tv_id}/load_url` com `http://100.121.227.88:<porta>`.
- **Gotcha:** o app só aceita comandos com a Activity em **foreground** (registra o handler em
  `onStart`, remove em `onStop`). Quando a TV está no launcher/screensaver, retorna
  `webview_not_ready` — resolvido trazendo o app ao foreground via ADB
  (`monkey -p com.felipe.webviewtv -c android.intent.category.LAUNCHER 1`).

---

## 8. Resumo

- ✅ Deploy do LingBot-Map via Frota Claude, com roteamento por capability para a GPU correta.
- ✅ Viser persistente exibindo reconstrução 3D em tempo real na TV2.
- ✅ Profiling: **13,3 FPS** @ 518×378 e footprint de VRAM quase constante (paged KV cache).
- ✅ 4 cenas de exemplo reconstruídas e alternáveis na TV.

---

*Testes conduzidos com Claude Code (Opus 4.8).*

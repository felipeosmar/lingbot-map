# Relatório de Testes — LingBot-Map no GB10

Registro dos testes de deploy e caracterização do **LingBot-Map** no **NVIDIA GB10
(Grace Blackwell)**, executados em **2026-07-21** via **Fleet Claude**. Documento
complementar ao [RELATORIO-TESTES.md](RELATORIO-TESTES.md) (que cobre a ctrob-workstation
com RTX PRO 6000).

---

## 1. Ambiente de execução

| Item | Valor |
|---|---|
| Máquina | **GB10** (host `GB10`) |
| Chip | **NVIDIA GB10 Grace Blackwell** — **arquitetura aarch64 (ARM)** |
| Memória | **121 GiB unificada** (LPDDR5X, compartilhada CPU+GPU) |
| CPU | 20 cores (Grace) |
| CUDA | 13.0 (driver 580.159.03) |
| Acesso | SSH via Tailscale (`100.85.216.71`) |
| PyTorch | **2.11.0 + cu130** (índice `download.pytorch.org/whl/cu130`, wheels aarch64) |
| Atenção | FlashInfer 0.6.15 (aarch64) |
| Ambiente Python | **`python -m venv` + pip** (não há `uv`/conda) |
| Checkpoint | `lingbot-map.pt` (balanced, 4,63 GB) |

**Diferenças-chave vs ctrob-workstation (RTX PRO 6000):**
- Arquitetura **ARM (aarch64)** vs x86_64 → PyTorch cu130 (não cu128), wheels ARM.
- **Memória unificada** (CPU+GPU no mesmo pool de 121 GiB) vs VRAM dedicada (97,9 GB GDDR7).
- `venv+pip` (uv ausente); systemd `--user` indisponível (viser via `setsid`).

---

## 2. Orquestração do deploy (Fleet Claude no GB10)

Deploy executado como **job da Fleet Claude**, roteado ao GB10 pela capability **`linux`**
(o worker oficial do GB10 já roda com `--capabilities linux --concurrency 2`):

- Repositório publicado como bare em `/home/ctrob/lingbot-map.git` (no próprio GB10);
  briefing em **`MISSION-GB10.md`** (adaptado para ARM: venv+pip, torch cu130, `setsid`).
- O worker clonou o repo, criou o venv, instalou torch aarch64 + lingbot + flashinfer,
  baixou o checkpoint e lançou o viser (cena oxford, `--mask_sky`, porta **8090** —
  a 8080 está ocupada pelo Open WebUI do GB10).
- Acompanhamento via Loki: `job.claimed → started → session.start → tool.* `. **Deploy
  concluído com sucesso** — `<title>Viser</title>` respondendo em `http://100.85.216.71:8090`.

**Observação:** o `claude` (Claude Code CLI) já estava no PATH do worker do GB10, então
não houve o problema de PATH que ocorreu na ctrob.

---

## 3. Consumo de recursos (processo do viser, cena oxford) — MEDIDO

| Recurso | Consumo no GB10 | Total | Observação |
|---|---|---|---|
| **CPU** | ~14,7% de 1 core (81 threads) | 20 cores | busy-loop do viser |
| **RAM/RSS** | **7,05 GB** (RSS do processo) | 121 GiB | inclui pesos + buffers |
| **Memória unificada** | 32 GiB usados no host (com viser) | 121 GiB | CPU+GPU compartilham |
| **Armazenamento** | venv 5,4 GB + modelo 4,4 GB ≈ **~10 GB** | — | |

> **Nota sobre "VRAM":** o GB10 usa **memória unificada** — não há VRAM dedicada, e o
> `nvidia-smi` não reporta `memory.used/free` por processo. A "VRAM" do modelo faz parte
> do pool de RAM de 121 GiB. O RSS de 7 GB do processo é a melhor proxy medível.

---

## 4. Desempenho (FPS) — ESTIMATIVA

⚠️ **Metodologia (transparência):** os scripts de profiling sintético usados na ctrob
(`gct_profile.py` e `scripts/benchmark_gct_memory.py`) **não foram reproduzíveis no GB10**:
- `gct_profile.py` força `torch.compile(mode="reduce-overhead")` + CUDA graphs, que
  **travam em aarch64/CUDA 13** (o flag `--compile` tem `default=True` sem `--no-compile`).
- O `benchmark_gct_memory.py` e execuções manuais de `demo.py` via SSH ficaram presas após
  a fase `Input:` (a inferência não completou em runs manuais; o SSH ao GB10 também esteve
  instável). A **única** execução de inferência bem-sucedida foi a do próprio job da Fleet.

**FPS estimado pelo deploy real (job da Fleet):** o pipeline completo da cena oxford
(320 frames, `--mask_sky`) — do checkpoint carregado até o viser no ar — levou **~137 s**.
Descontando carga de modelo, leitura de imagens, download/execução do sky-seg, a inferência
pura fica na ordem de **~3–4 FPS**.

### Comparação de desempenho

| Máquina | GPU / memória | FPS (518×378) | Método |
|---|---|---|---|
| **ctrob-workstation** | RTX PRO 6000 (GDDR7 dedicada) | **13,3 FPS** | medido (`gct_profile`) |
| **GB10** | Grace Blackwell (LPDDR5X unificada) | **~3–4 FPS** (est.) | estimado (deploy real) |

O GB10 é **~3–4× mais lento** neste workload de inferência sequencial. A causa provável é a
**largura de banda de memória**: a inferência streaming é memory-bound, e a LPDDR5X unificada
do GB10 tem banda muito menor que a GDDR7 dedicada da RTX PRO 6000. Em contrapartida, o GB10
oferece **121 GiB de memória unificada**, permitindo modelos/sequências muito maiores sem
esbarrar num limite de VRAM.

---

## 5. Resumo

- ✅ Deploy do LingBot-Map no GB10 (aarch64) **via Fleet Claude**, roteado por capability `linux`.
- ✅ Adaptação ARM bem-sucedida: torch 2.11 cu130, flashinfer aarch64, venv+pip.
- ✅ Viser persistente reconstruindo a cena oxford em `http://100.85.216.71:8090`.
- ⚠️ Profiling sintético não reproduzível em aarch64 (compile/CUDA-graph e runs manuais travam);
  FPS estimado (~3–4) pelo tempo de inferência do deploy real.
- 📊 **Conclusão comparativa:** GB10 ~3–4× mais lento que a RTX PRO 6000 (memory-bound), mas
  com 121 GiB de memória unificada para escala. VRAM dedicada da ctrob vence em throughput.

---

*Testes conduzidos com Claude Code (Opus 4.8). Números marcados como MEDIDO vs ESTIMATIVA.*

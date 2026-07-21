# MISSÃO (GB10 / aarch64): Subir o visualizador do LingBot-Map (cena oxford)

Você roda como job da Frota Claude no **GB10** (NVIDIA GB10 Grace Blackwell, aarch64,
memória unificada ~121 GB, CUDA 13.0). Objetivo: reconstruir `example/oxford` com o
LingBot-Map e deixar o **viser no ar de forma PERSISTENTE** (sobrevive ao fim do job),
acessível pela rede para exibição numa TV.

## Regras invioláveis (frota)
1. **NADA grande no diretório de trabalho (cwd = clone do repo)** — tudo lá é commitado no
   branch de resultado. Portanto: venv em **`/home/ctrob/lingbot-env`**, checkpoint em
   **`/home/ctrob/lingbot-models`**, logs em **`/home/ctrob/lingbot-viser.log`** (todos FORA do cwd).
2. Não commite venv, modelos, `__pycache__`, `example/*_sky_masks/`.
3. Não use `sudo`. Rode como usuário `ctrob`.

## Passos

### 1. Ambiente Python (aarch64 — usar venv+pip; NÃO há uv/conda aqui)
```bash
cd "$PWD"
python3 -m venv /home/ctrob/lingbot-env
source /home/ctrob/lingbot-env/bin/activate
pip install --upgrade pip
```

### 2. PyTorch aarch64 + CUDA 13.0 (índice cu130 tem wheels ARM)
```bash
pip install torch==2.11.0 torchvision --index-url https://download.pytorch.org/whl/cu130
```

### 3. Pacote + visualização + FlashInfer (aarch64 disponível)
```bash
pip install -e ".[vis]"     # traz a constraint numpy<2 do lingbot
pip install flashinfer-python || echo "flashinfer opcional falhou; usar --use_sdpa"
```

### 4. Baixar checkpoint balanced (FORA do cwd)
```bash
mkdir -p /home/ctrob/lingbot-models
hf download robbyant/lingbot-map lingbot-map.pt --local-dir /home/ctrob/lingbot-models \
  || huggingface-cli download robbyant/lingbot-map lingbot-map.pt --local-dir /home/ctrob/lingbot-models
test -f /home/ctrob/lingbot-models/lingbot-map.pt || { echo "FALHA: checkpoint"; exit 1; }
```

### 5. Sanidade CUDA
```bash
python -c "import torch; assert torch.cuda.is_available(); print('GPU OK:', torch.cuda.get_device_name(0))"
```

### 6. Lançar o viser DESTACADO (sobrevive ao fim do job)
Não rode em foreground nem com `nohup &` simples. Use `systemd-run --user`; se falhar, `setsid`.
```bash
FLINFER=""; python -c "import flashinfer" 2>/dev/null || FLINFER="--use_sdpa"
systemd-run --user --unit=lingbot-viser --collect \
  --working-directory="$PWD" \
  --setenv=PATH=/home/ctrob/lingbot-env/bin:/usr/bin:/bin \
  -p StandardOutput=append:/home/ctrob/lingbot-viser.log \
  -p StandardError=append:/home/ctrob/lingbot-viser.log \
  /home/ctrob/lingbot-env/bin/python demo.py \
    --model_path /home/ctrob/lingbot-models/lingbot-map.pt \
    --image_folder example/oxford --mask_sky --port 8080 $FLINFER \
  || setsid bash -c "cd $PWD && source /home/ctrob/lingbot-env/bin/activate && python demo.py --model_path /home/ctrob/lingbot-models/lingbot-map.pt --image_folder example/oxford --mask_sky --port 8080 $FLINFER >> /home/ctrob/lingbot-viser.log 2>&1" < /dev/null &
```

### 7. Aguardar o viser subir (inferência roda ANTES do viser abrir; poll ~25 min)
```bash
for i in $(seq 1 150); do
  grep -q "3D viewer at" /home/ctrob/lingbot-viser.log 2>/dev/null && { echo VISER_UP; break; }
  curl -sf -m2 http://localhost:8080 >/dev/null 2>&1 && { echo PORT_OK; break; }
  sleep 10
done
```

### 8. Escrever SUMARIO.md (no cwd — pequeno)
Campos: `STATUS` (UP/FAILED), `URL_LOCAL` http://localhost:8080,
`URL_REDE` http://100.85.216.71:8080 (Tailscale do GB10, usada pela TV),
`SYSTEMD_UNIT` lingbot-viser.service, `CENA` example/oxford (--mask_sky),
`BACKEND` (flashinfer ou sdpa), `LOG_TAIL` (20 linhas de /home/ctrob/lingbot-viser.log).
Se FALHOU, inclua o erro. NÃO reinicie serviços de GPU do host (vLLM/DiffusionGemma).

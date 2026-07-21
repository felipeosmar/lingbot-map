# MISSÃO: Subir o visualizador do LingBot-Map (cena oxford) na ctrob-workstation

Você está rodando como job da Frota Claude na **ctrob-workstation** (GPU RTX PRO 6000,
~97 GB VRAM livres). Seu objetivo é reconstruir a cena `example/oxford` com o
LingBot-Map e deixar o **visualizador viser no ar de forma PERSISTENTE**, acessível
pela rede, para ser exibido numa TV. Ao terminar, o viser deve continuar rodando
mesmo depois que este job encerrar.

## Regras invioláveis (frota)

1. **NADA grande pode nascer no diretório de trabalho (cwd = clone do repo).** Tudo que
   ficar no cwd é commitado no branch de resultado. Portanto:
   - o ambiente Python vai em **`/home/ctrob/lingbot-env`** (fora do cwd);
   - o checkpoint vai em **`/home/ctrob/lingbot-models`** (fora do cwd);
   - logs vão em **`/home/ctrob/lingbot-viser.log`** (fora do cwd).
2. Não commite venv, modelos, `__pycache__`, nem `example/*_sky_masks/`. Se algo grande
   cair no cwd, mova para fora antes de terminar.
3. Não use `sudo`. Tudo roda como usuário `ctrob`.

## Passos

### 1. Ambiente Python com uv (conda não existe nesta máquina)
```bash
cd "$PWD"   # cwd = clone do repo lingbot-map
uv venv /home/ctrob/lingbot-env --python 3.10
source /home/ctrob/lingbot-env/bin/activate
```

### 2. PyTorch CUDA 12.8 + o pacote + visualização + FlashInfer
```bash
uv pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install -e ".[vis]"
uv pip install --index-url https://pypi.org/simple flashinfer-python || echo "flashinfer opcional falhou; seguirá com --use_sdpa"
```

### 3. Baixar o checkpoint balanced (FORA do cwd)
```bash
mkdir -p /home/ctrob/lingbot-models
hf download robbyant/lingbot-map lingbot-map.pt --local-dir /home/ctrob/lingbot-models \
  || huggingface-cli download robbyant/lingbot-map lingbot-map.pt --local-dir /home/ctrob/lingbot-models
test -f /home/ctrob/lingbot-models/lingbot-map.pt || { echo "FALHA: checkpoint não baixado"; exit 1; }
```

### 4. Sanidade da GPU
```bash
python -c "import torch; assert torch.cuda.is_available(), 'CUDA indisponível'; print('GPU OK:', torch.cuda.get_device_name(0))"
```

### 5. Lançar o viser como serviço systemd --user TRANSIENTE (sobrevive ao fim do job)
NÃO rode `demo.py` em foreground (ele bloqueia) nem com `nohup &` simples (o job pode
matar a árvore de processos). Use `systemd-run --user`, que desacopla o processo:
```bash
FLINFER=""; python -c "import flashinfer" 2>/dev/null || FLINFER="--use_sdpa"
systemd-run --user --unit=lingbot-viser --collect \
  --working-directory="$PWD" \
  --setenv=PATH=/home/ctrob/lingbot-env/bin:/usr/bin:/bin \
  -p StandardOutput=append:/home/ctrob/lingbot-viser.log \
  -p StandardError=append:/home/ctrob/lingbot-viser.log \
  /home/ctrob/lingbot-env/bin/python demo.py \
    --model_path /home/ctrob/lingbot-models/lingbot-map.pt \
    --image_folder example/oxford --mask_sky --port 8080 $FLINFER
```
Se `systemd-run --user` falhar (sem instância user), caia para:
`setsid bash -c '... demo.py ... >> /home/ctrob/lingbot-viser.log 2>&1' < /dev/null &`

### 6. Aguardar o viser subir (a inferência roda ANTES de o viser abrir)
O demo processa os 320 frames e só então imprime `3D viewer at http://localhost:8080`.
Faça poll de até ~25 min:
```bash
for i in $(seq 1 150); do
  if grep -q "3D viewer at" /home/ctrob/lingbot-viser.log 2>/dev/null; then echo "VISER UP"; break; fi
  if curl -sf -m 2 http://localhost:8080 >/dev/null 2>&1; then echo "PORT 8080 OK"; break; fi
  sleep 10
done
curl -sf -m 3 http://localhost:8080 >/dev/null 2>&1 && echo "confirmado: porta 8080 respondendo"
```

### 7. Escrever SUMARIO.md (no cwd — é pequeno, pode commitar)
Grave um `SUMARIO.md` com EXATAMENTE estes campos preenchidos:
- `STATUS:` UP ou FAILED
- `URL_LOCAL:` http://localhost:8080
- `URL_REDE:` http://100.121.227.88:8080   (Tailscale da ctrob — usada pela TV)
- `SYSTEMD_UNIT:` lingbot-viser.service (user)  — parar com `systemctl --user stop lingbot-viser`
- `CENA:` example/oxford (--mask_sky)
- `LOG_TAIL:` as últimas 20 linhas de /home/ctrob/lingbot-viser.log
- `VRAM:` saída de `nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader`

Se STATUS=FAILED, inclua o erro relevante do log. NÃO reinicie serviços de GPU
(llama/vllm/ollama/comfyui) — eles devem permanecer desligados.

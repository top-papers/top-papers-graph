<!-- SPDX-FileCopyrightText: 2026 top-papers-graph contributors -->
<!-- SPDX-License-Identifier: GPL-3.0-or-later -->

# Kaggle API: FP16 primary и NF4 sensitivity на T4x2

Этот каталог запускает base и tuned arms в двух отдельных private Kaggle script kernels. Каждый
kernel требует ровно две NVIDIA T4, использует обе через `device_map=balanced`, выполняет один arm и
сохраняет mode-bound проверяемый state. Запрос `NvidiaTeslaT4` в API не гарантирует фактически
выданную машину, поэтому runner проверяет `nvidia-smi`, число GPU в Torch и `T4` в обоих именах.

Поддерживаются два не взаимозаменяемых режима:

- `fp16-primary`: confirmatory endpoint, unquantized FP16 base/compute и native FP32 PEFT LoRA;
- `nf4-sensitivity`: FP16 compute + NF4 base и тот же FP32 LoRA, только automatic diagnostics.

Зафиксированы exact `N=150` и два эксперта, каждый оценивает все FP16 primary pairs. NF4 outputs не
передаются экспертам и не используются для semantic claims. Оба режима требуют corrected immutable
benchmark `B`, adapter/training release `R`, lineage attestation `M`, отдельные IDs/output directories,
plan и fresh strict `prepare`. Старые blocked plan, prepare bundle и predictions переносить нельзя.

## Что фиксируется

FP16 primary config обязан иметь в обеих arms точные unquantized kwargs: FP16, balanced placement,
SDPA, low-memory loading и `trust_remote_code=false`. Tuned arm обязан явно задавать
`adapter_kwargs.autocast_adapter_dtype=true`, а experiment -- `precision_mode=fp16-primary`, exact
`power.n_items=150`, `require_exact_n_items=true` и two-reviewer overlap. Loader проверяет фактические
FP16 base parameters, FP32 LoRA parameters, отсутствие quantization metadata/packed parameters,
missing/unexpected PEFT checkpoint keys, ровно один active adapter и exact CUDA devices `{0,1}` без
CPU/disk/meta placement. Strict prepare допускает только plain LoRA без `modules_to_save`, saved bias
или trainable-token state.

`make_nf4_config.py` принимает только этот final FP16 capacity config и сохраняет B/R/M, benchmark,
processor, generation, review, statistics, training protocol и FP32 adapter contract. Меняются только
identity/output fields и одинаковые model kwargs с exact NF4 config. IDs обязаны содержать `nf4` и
`sensitivity`; output directory должен быть новым safe path под `runs/`. Generator записывает
`precision_mode=nf4-sensitivity`; inference использует `result_scope=automatic_sensitivity_only`.

Kaggle base image содержит Torch, torchvision и CUDA. Runner требует Torch >=2.3 и не переустанавливает
его. `requirements-kaggle.txt` является общим FP16 runtime без bitsandbytes;
`requirements-kaggle-nf4.txt` добавляет exact `bitsandbytes==0.48.1`. Затем выполняются `pip check`,
`pip freeze` и mode-specific import/version checks. Runtime, mode и transitive environment сравниваются
между arms. Metadata использует `docker_image_pinning_type=original`.

Runtime фиксирует `peft==0.19.1`: более ранний `0.19.0` нельзя использовать для этого workflow из-за
ошибки совместимости quantized adapter со старыми допустимыми версиями Torch.

Payload builder повторно валидирует config, prepare manifest и audit, требует
`publication_ready=true`, `exploratory=false`, совпадение output directory и неизменный source
fingerprint. Он копирует точный `code_provenance` inventory и только prepare allowlist: manifest,
concrete artifact files, training files и перечисленные audit JSON изображения. Predictions, другие
run extras и credential-like paths (`kaggle.json`, `.env*`, credentials/tokens/private keys)
исключены; source допускается только из явного reviewed path allowlist. Результат
находится в `payload/top-papers-graph/`; рядом создаются
private `dataset-metadata.json` и полный SHA256 inventory. Rendered kernel фиксирует SHA256 самого
`payload_manifest.json`, а runner выбирает mounted input только по одновременному совпадению filename,
dataset slug и этого SHA256. Поэтому новая Kaggle dataset version с тем же slug не может незаметно
подменить уже rendered protocol. Перед render локальные config и runner template обязаны byte-for-byte
совпадать с их entries в payload inventory; config нельзя заменить после создания input dataset.
`.git`, произвольные файлы репозитория и credentials не копируются. Symlink и overwrite запрещены.
Controlled `--dir-mode zip` превращает `payload/`, а позднее `predictions/` и `runtime/`, в Kaggle ZIP.
Runner потоково распаковывает их во временный каталог, отклоняя ambiguity с unpacked layout,
необъявленные/дублирующиеся/опасные entries, symlink attributes, size/hash mismatch и zip bombs.

## 1. Локальное окружение

Команды выполняются в Windows PowerShell 5.1 из корня `top-papers-graph`:

```powershell
python -m venv .venv-kaggle
.\.venv-kaggle\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r experiments/vlm_ab_evaluation/kaggle/requirements-local.txt
python -m pip install -e ".[vlm_ab,dev]"
```

Скачайте API token в Kaggle: **Settings -> API -> Create New Token**. Сохраните полученный JSON
ровно как:

```text
%USERPROFILE%\.kaggle\kaggle.json
```

Это стандартный `%USERPROFILE%\.kaggle\kaggle.json`. Не помещайте его в repo, `kaggle/build`,
payload или командную строку. Оркестратор только читает этот стандартный файл и никогда его не
копирует. Проверка клиента и auth:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File `
  experiments/vlm_ab_evaluation/kaggle/kaggle_api.ps1 -Action CheckAuth
```

Требуется ровно `kaggle==2.2.3`; другая версия отклоняется. Auth check выполняет реальный API request,
получает effective username после environment overrides и использует тот же `python -m kaggle`, что и
остальные CLI actions.

## 2. Config, plan и prepare до просмотра outputs

`$CapacityConfig` -- финальный strict cap150 FP16 config с B/R/M и двумя pseudonymous expert IDs.
Benchmark section этого config обязан pin-ить опубликованный Phase 0 `assembly_manifest.json` по пути и
SHA256; strict `prepare` проверяет и сохраняет его до запуска Kaggle. Сначала commit, clean plan и strict
prepare для primary:

```powershell
$Mode = "fp16-primary"
$CapacityConfig = "experiments/vlm_ab_evaluation/configs/qwen3vl-cap150-capacity-v1.yaml"
$CapacityRun = "runs/vlm_ab/qwen3vl-cap150-capacity-v1"
$RunConfig = $CapacityConfig
$PreparedRun = $CapacityRun

git status --short
python experiments/vlm_ab_evaluation/run_pipeline.py --config $RunConfig plan
python experiments/vlm_ab_evaluation/run_pipeline.py --config $RunConfig prepare
```

До просмотра primary outputs создайте и preregister отдельный NF4 sensitivity run:

```powershell
$Kaggle = "experiments/vlm_ab_evaluation/kaggle"
$CapacityConfig = "experiments/vlm_ab_evaluation/configs/qwen3vl-cap150-capacity-v1.yaml"
# Non-capacity alternative only:
# $CorrectedStrict = "experiments/vlm_ab_evaluation/configs/qwen3vl_corrected_strict.yaml"
$Nf4Config = "experiments/vlm_ab_evaluation/configs/qwen3vl-nf4-sensitivity-t4x2.yaml"
$Nf4Run = "runs/vlm_ab/qwen3vl-nf4-sensitivity-t4x2-v1"

python "$Kaggle/make_nf4_config.py" `
  --input $CapacityConfig `
  --output $Nf4Config `
  --experiment-id "qwen3vl-nf4-sensitivity-t4x2-v1" `
  --public-id "study-2026-nf4-sensitivity-t4x2-v1" `
  --output-dir $Nf4Run

git add $Nf4Config
git commit -m "Add NF4 sensitivity protocol"
git status --short
git rev-parse HEAD

# Continue only if git status --short printed no lines.
python experiments/vlm_ab_evaluation/run_pipeline.py --config $Nf4Config plan
python experiments/vlm_ab_evaluation/run_pipeline.py --config $Nf4Config prepare
```

Не используйте `--exploratory`. Оба режима должны находиться в reviewed commit; `git status --short`
пуст до соответствующего `plan` и `prepare`. После prepare нельзя менять VLM/Kaggle source или
config. Любое изменение требует новых IDs/output directory и повторных commit/plan/prepare.

## 3. Private input dataset

Ниже показан primary. Задайте имена один раз; slugs уникальны в Kaggle account:

```powershell
$Api = "experiments/vlm_ab_evaluation/kaggle/kaggle_api.ps1"
$Mode = "fp16-primary"
$RunConfig = $CapacityConfig
$PreparedRun = $CapacityRun
$InputSlug = "vlm-ab-fp16-primary-input-v1"
$BaseStateSlug = "vlm-ab-fp16-primary-base-state-v1"
$FinalStateSlug = "vlm-ab-fp16-primary-final-state-v1"
$BaseKernel = "vlm-ab-fp16-primary-base-v1"
$TunedKernel = "vlm-ab-fp16-primary-tuned-v1"
$InputStaging = "experiments/vlm_ab_evaluation/kaggle/staging/$Mode/$InputSlug-v1"
$CommonArgs = @(
  "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $Api,
  "-Mode", $Mode, "-Config", $RunConfig, "-PreparedRun", $PreparedRun,
  "-InputSlug", $InputSlug, "-BaseStateSlug", $BaseStateSlug,
  "-FinalStateSlug", $FinalStateSlug, "-BaseKernelSlug", $BaseKernel,
  "-TunedKernelSlug", $TunedKernel, "-StagingDir", $InputStaging
)

& powershell @CommonArgs -Action CreateInputDataset
```

Builder и API отказываются перезаписывать staging. Для новой dataset version задайте новый
`$InputStaging`, заново создайте `$CommonArgs` и используйте
`& powershell @CommonArgs -Action VersionInputDataset`. Тот же `$InputStaging` обязателен при
последующем render. Version допустима только до plan/prepare
для соответствующего нового protocol либо при byte-identical validated payload; нельзя незаметно
заменять input уже начатого эксперимента.

## 4. Base kernel

```powershell
& powershell @CommonArgs -Action RenderKernel -Arm base

& powershell @CommonArgs -Action PushBase

& powershell @CommonArgs -Action Status -Arm base
```

`RenderKernel` создает только ignored `kaggle/build/<mode>/base/runner.py`,
`kernel-metadata.json` и `render-lock.json`. Lock связывает mode/arm/slugs, exact input dataset
version, SHA256 payload manifest, config/template bytes, exact-version SDK bridge, independent state
validator и reconstructed runner/metadata. Runner
включает независимый `execution_lock_sha256`, который затем обязан присутствовать в state.
`PushBase` перед upload полностью реконструирует expected runner из verified payload и workflow
inputs; простое редактирование runner и его lock не проходит. До remote call атомарно создаётся
write-once push-attempt lock, блокирующий concurrent/ambiguous repeated push. После принятого push
атомарно создаётся `push-receipt.json` с точным Kaggle kernel version. Локальный SDK bridge перед status/download
запрашивает именно этот `version_label` и сверяет remote source SHA256, privacy, GPU metadata и
version-pinned dataset sources. Metadata
фиксирует private script kernel, internet, GPU, `machine_shape=NvidiaTeslaT4`, original-image pinning
и version-pinned private input dataset. Push использует `kaggle kernels push -p ... --accelerator
NvidiaTeslaT4 --timeout 43200`, затем принимает initial status `new`, `queued`, `running` или
`complete`.

После push можно проверять status; download сам polling-ом ждёт `complete` до 12 часов:

```powershell
& powershell @CommonArgs -Action DownloadBase

& powershell @CommonArgs -Action CreateBaseStateDataset
```

Второе действие принимает только `complete=true`, `arm=base` и sidecar с `error_rows=0`,
`successful_rows=completed_rows=expected_rows` и правильным output SHA256. Оно создает private state
dataset и ждет status `ready`. Complete state обязан иметь точный mode/arm-specific inventory с
правильными SHA256; JSONL фактически перечитывается с проверкой exact row schema, row count, unique
sample/condition, arm/status/input fingerprints и raw/parsed consistency. Independent Python validator
заново строит expected condition rows из immutable payload, проверяет sidecar configurations/runtime и
пересчитывает automatic metrics; сам validator byte-bound к payload. Arm summary сверяется с sidecar и
inference summary. Final validation дополнительно требует byte-for-byte совпадение унаследованных
base prediction, sidecar и arm summary с exact parent state, зафиксированным tuned render lock.
Отсутствующие и
дополнительные даже allowlisted файлы запрещены. Download запрашивает kernel version из push receipt,
проверяет state во временном каталоге и только затем публикует fixed local path. Если kernel
упал, `finally` всегда экспортирует traceback в
`runtime/error.log`, доступные partial artifacts и `complete=false`, даже при ошибке до определения
run output. Такой output служит только диагностикой. Автоматический partial base resume не
поддерживается: исправьте причину и запустите новую base kernel version с исходным input dataset.

## 5. Tuned kernel и final output

```powershell
& powershell @CommonArgs -Action RenderKernel -Arm tuned -StateSlug $BaseStateSlug

& powershell @CommonArgs -Action PushTuned
```

`PushTuned` сначала требует `ready` у base-state dataset, проверяет точные input/base-state sources в
rendered metadata и скачивает receipted complete base kernel output в новый timestamped каталог. Его
manifest обязан совпасть с локальным state, из которого создан dataset. Tuned runner на Kaggle
ищет input и base state по точным SHA256 manifests, затем требует `complete=true`, `arm=base`, точный
dataset slug, exact complete inventory, base sidecar, каждый file SHA256, config/run/protocol/input
fingerprints и mode-specific result scope. После этого он overlays только `predictions/*` и
`inference_summary.json` в точный run output. Runtime logs state dataset проверяются, но в run tree
не overlays. Tuned `render-lock.json` также фиксирует exact base-state dataset version, SHA256 локально
проверенного base state manifest и новый execution lock; `PushTuned` реконструирует runner перед
upload. Tuned model запускается отдельным process/kernel после освобождения base kernel.

```powershell
& powershell @CommonArgs -Action Status -Arm tuned

& powershell @CommonArgs -Action DownloadFinal
```

Проверьте `downloads/fp16-primary/final/kaggle_export/state_manifest.json`: `complete=true`,
`workflow_mode=fp16-primary`, `arm=tuned`,
`base_state_complete=true`, hashes совпадают, а `parent_state_manifest_sha256` равен использованному
base state и `execution_lock_sha256` равен tuned render lock. `DownloadFinal` автоматически отклоняет
missing/extra state files, несогласованные
sidecars/inference summary, неправильный result scope, fingerprints или hashes до принятия output.
После этого только FP16 predictions можно переносить в blind/review pipeline.

## 6. NF4 automatic sensitivity

Повторите sections 3-5 с `-Mode nf4-sensitivity`, `$RunConfig=$Nf4Config`,
`$PreparedRun=$Nf4Run` и отдельными `InputSlug`, `BaseStateSlug`, `FinalStateSlug`, `BaseKernelSlug` и
`TunedKernelSlug`, каждый из которых содержит `nf4-sensitivity`. Generated/state/download paths
разделяются по mode. После задания этих переменных заново создайте `$CommonArgs`. После complete tuned
state вычисляйте только заранее заданные automatic
generation, parse/schema и runtime diagnostics из `inference_summary.json`. Команды `blind`,
`aggregate` и `run` для NF4 отклоняются кодом. Не создавайте expert packages из NF4 outputs и не
интерпретируйте schema validity как semantic quality.

## Secrets и диагностика

Kaggle API не умеет безопасно прикреплять notebook secrets к kernel version. Public Hugging Face
repositories скачиваются анонимно при `enable_internet=true`. Если Hub требует token из-за rate
limit, один раз создайте Kaggle Secret в web UI и предоставьте его kernel через поддерживаемый Kaggle
Secrets mechanism; этот workflow намеренно не автоматизирует secret и не помещает token в metadata,
dataset или source. Не делайте repository private без отдельного reviewed secret-loading change.

`& powershell @CommonArgs -Action Logs -Arm base|tuned` скачивает output receipted kernel version в
новый timestamped mode-specific каталог. При
ошибке сначала читайте `runtime/error.log`, `runtime/install.log`, `runtime/pip-check.log`,
`runtime/pip-freeze.txt`, `runtime/infer-<arm>.log`, `runtime/environment.json` и
`state_manifest.json`. Dataset create/version/state upload считаются успешными только после polling
до exact expected version со статусом `ready`; scientific `DownloadBase`/`DownloadFinal` требуют
complete receipted kernel и совпадающий execution lock, а `Logs` отдельно
пытается получить диагностический output failed run. Generated `build/`,
`staging/` и `downloads/` игнорируются Git, но source scripts никогда не игнорируются.

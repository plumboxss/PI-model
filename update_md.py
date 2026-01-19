# -*- coding: utf-8 -*-
from pathlib import Path

files = {}

files["README.md"] = """# PI-model 由щ뱶誘?
## 媛쒖슂
李⑤웾 ?쒖뒪?쒖뀡 ?쒕??덉씠?섏뿉???좏샇??湲곕컲 蹂댁긽???숈뒿?섎뒗 VAE ?꾨줈?앺듃?낅땲?? ?꾩옱??**Posterior Collapse ?꾪솕**? **z ?ъ슜 媛뺤젣**瑜??꾪빐 ?곗씠?걔룸え?맞룻븰??援ъ“媛 ?ш쾶 媛쒖꽑?섏뿀?듬땲??

## ?꾩옱 ?듭떖 援ъ“
- **?곗씠???앹꽦**: PD+shaping ?쒖뼱, 踰뷀봽 ?믪씠/???쒕뜡?? `rms_acceleration` ?ы븿
- **?좏샇 ?ㅻ씪??*: 200紐?媛???좎?, **A/B ?곸땐 洹몃９(?뚯닔/?묒닔 遺꾪룷)**
- **?붿퐫??*: `r = <phi(s,a), psi(z)>` dot?몆roduct 援ъ“, `feature_dim=16`
- **?몄퐫??*: TrajectoryEncoder??**Attention Pooling**
- **?낅젰 泥섎━**: `x_com` ?쒓굅, obs/act **Z-score ?뺢퇋??*, **T ??T/5 ?ㅼ슫?섑뵆**
- **?숈뒿**: `latent_dim=4`, `context_size=30`, `reward_scaling=1000`
- **KL ?ㅼ?以?*: cosine annealing, cycles=4
- **?듯떚留덉씠?**: decoder lr = encoder lr * 0.2
- **ROC ?쒓컖??*: eval留덈떎 `roc_epoch_{epoch}.png` ???
## 鍮좊Ⅸ ?ㅽ뻾
### 1) ?곗씠???앹꽦
```bash
python scripts/generate_data.py --num-episodes 500 --dataset-id A --dataset-name raw_trajectories_A --visualize
```

### 2) ?좏샇???곗씠?곗뀑
```bash
python scripts/build_preference_dataset.py \
  --input_path artifacts/A/datasets/raw_trajectories_A.pkl \
  --output_path datasets/preference_dataset_A.pkl \
  --num_pairs 20000 \
  --visualize
```

### 3) ?숈뒿
```bash
python scripts/train_model.py \
  --dataset_path datasets/preference_dataset_A.pkl \
  --logging.output_dir "logs" \
  --comment "pretrain_suspension_model_A" \
  --seed 42
```

## 二쇱슂 湲곕낯媛?- `trajectory_encoder_type=mlp`
- `latent_dim=4`
- `context_size=30`
- `annealer_type=cosine`, `annealer_cycles=4`
- `reward_scaling=1000`
- `free_bits=0.0`

## 異쒕젰 寃쎈줈
- ?곗씠?곗뀑: `artifacts/{id}/datasets/*.pkl`
- ?좏샇?꾩뀑: `datasets/preference_dataset_*.pkl`
- ?숈뒿 濡쒓렇/紐⑤뜽: `logs/{env}/{model_type}/{comment}/s{seed}/`
- ROC 而ㅻ툕: `logs/.../roc_epoch_{epoch}.png`

## 李멸퀬 臾몄꽌
- `PROJECT_VALIDATION_REPORT.md`: 寃利??댁뒋 ?뺣━
- `ARCHITECTURAL_AUDIT.md`: 援ъ“ 蹂寃?湲곕줉
- `VISUALIZATION_GUIDE.md`: ?쒓컖??媛?대뱶
"""

files["FILE_REFERENCE.md"] = """# ?꾨줈?앺듃 ?ㅻ챸: ?뚯씪/?붾젆?좊━ ?붿빟

## 猷⑦듃
- `README.md`: ?ㅽ뻾/援ъ꽦 ?붿빟
- `requirements.txt`, `environment.yml`: ?섍꼍 援ъ꽦
- `PROJECT_VALIDATION_REPORT.md`: 寃利?湲곕줉
- `ARCHITECTURAL_AUDIT.md`: 援ъ“ 蹂寃?湲곕줉

## scripts/
- `generate_data.py`: ?쒕??덉씠???곗씠???앹꽦 (PD+shaping, bump ?ㅼ뼇??
- `build_preference_dataset.py`: ?좏샇???곗씠?곗뀑 ?앹꽦 (A/B ?곸땐 ?ㅻ씪??
- `train_model.py`: VAE ?숈뒿 + ROC ???- `run_interactive_adaptation.py`: ?곸쓳 猷⑦봽
- `evaluate_adaptation.py`: ?곸쓳 ?깅뒫 ?됯?

## src/
- `simulation/`: 臾쇰━/?섍꼍/踰뷀봽/?쒖뼱湲?- `models/`: VAE + encoder/decoder
- `data/`: loader (x_com ?쒓굅, obs/act ?뺢퇋?? ?ㅼ슫?섑뵆)
- `utils/`: ?쒓컖?? ?숈뒿 ?좏떥
"""

files["CODE_UNDERSTANDING_GUIDE.md"] = """# ?꾨줈?앺듃 ?ㅻ챸: 肄붾뱶 ?댄빐 媛?대뱶

## 鍮좊Ⅸ ?댄빐 ?쒖꽌
1. `README.md` ?꾩껜 援ъ“
2. ?곗씠???앹꽦: `scripts/generate_data.py` ??`src/utils/simulation_utils.py`
3. ?좏샇???앹꽦: `scripts/build_preference_dataset.py`
4. 紐⑤뜽: `src/models/vae.py`, `src/models/trajectory_encoder.py`, `src/models/reward_decoder.py`
5. ?숈뒿: `scripts/train_model.py`

## ?듭떖 ?먮쫫
`generate_data.py` ??`build_preference_dataset.py` ??`train_model.py`

## ?꾩옱 援ъ“ ?뱀쭠
- z??`set_encoder`?먯꽌 異붾줎
- 蹂댁긽? **dot?몆roduct ?붿퐫??*濡?怨꾩궛
- encoder ?留곸? **attention pooling**
- obs/act ?뺢퇋??+ ?쒓컙 ?ㅼ슫?섑뵆
"""

files["VISUALIZATION_GUIDE.md"] = """# ?꾨줈?앺듃 ?ㅻ챸: ?쒓컖??媛?대뱶

## ?앹꽦?섎뒗 ?쒓컖??- ?곗씠?? `trajectory_samples.png`, `feature_distributions.png`
- ?좏샇?? `preference_distribution.png`
- ?숈뒿: `training_curves.png`, `roc_epoch_*.png`
- ?곸쓳/?됯?: `z_evolution.png`, `reward_distribution.png`

## ?ъ슜踰?- `generate_data.py --visualize`
- `build_preference_dataset.py --visualize`
- `train_model.py` (ROC??eval留덈떎 ?먮룞 ???

## ?댁꽍 ?ъ씤??- ROC AUC媛 0.5 洹쇱쿂硫?遺꾨퀎??遺議?- KL??0?쇰줈 ?섎졃?섎㈃ posterior collapse 媛?μ꽦 ?믪쓬
"""

files["VISUALIZATION_SUMMARY.md"] = """# ?꾨줈?앺듃 ?ㅻ챸: ?쒓컖???붿빟

- ?곗씠???덉쭏: `feature_distributions.png`
- ?좏샇??洹좏삎: `preference_distribution.png`
- ?숈뒿 ?덉젙?? `training_curves.png`
- 遺꾨퀎?? `roc_epoch_*.png`
"""

files["PARAMETER_REVIEW.md"] = """# ?곌뎄怨쇱젙 ?뺣━: ?뚮씪誘명꽣 寃곗젙 ?꾪솴

## ?꾩옱 ?듭떖 ?ㅼ젙
- `latent_dim=4`
- `context_size=30`
- `trajectory_encoder_type=mlp`
- `annealer_type=cosine`, `annealer_cycles=4`
- `reward_scaling=1000`
- `free_bits=0.0`
- decoder lr = encoder lr * 0.2

## 蹂寃??댁쑀 ?붿빟
- Posterior collapse ?꾪솕
- z ?ъ슜 媛뺤젣
- ?쒗??湲몄씠 異뺤냼(T/5)濡??덉젙???뺣낫
"""

files["PROJECT_VALIDATION_REPORT.md"] = """# ?곌뎄怨쇱젙 ?뺣━: ?꾨줈?앺듃 寃利?湲곕줉

## 理쒓렐 ?뺤씤 ?ы빆
- `x_com` ?쒓굅 諛?obs/act ?뺢퇋???곸슜
- ?ㅼ슫?섑뵆濡?T=1000 ??200
- dot?몆roduct ?붿퐫???곸슜
- ROC 而ㅻ툕 ????뺤씤

## ?⑥? 由ъ뒪??- z 寃쎈줈 gradient ?쏀솕 ?щ? 紐⑤땲?곕쭅 ?꾩슂
- 洹몃９ A/B 遺꾪룷 ?④낵??寃利??꾩슂
"""

files["ARCHITECTURAL_AUDIT.md"] = """# ?곌뎄怨쇱젙 ?뺣━: ?꾪궎?띿쿂 蹂寃?湲곕줉

- ?붿퐫?? residual ??dot?몆roduct 援ъ“
- ?몄퐫?? mean/max pooling ??attention pooling
- ?곗씠?곗뀑: K-Means ?쒓굅, 媛???좎? 媛以묒튂 ?ㅻ씪?대줈 ?꾪솚
- ?낅젰: obs/act ?뺢퇋??+ ?ㅼ슫?섑뵆
"""

files["REFACTORING_SUMMARY.md"] = """# ?곌뎄怨쇱젙 ?뺣━: 由ы뙥?좊쭅 ?붿빟

- ?쒕??덉씠???ㅼ뼇???뺣? (PD+shaping + bump ?ㅼ뼇??
- ?좏샇 ?ㅻ씪??援ъ“ 蹂寃?(A/B ?곸땐)
- 蹂댁긽 ?붿퐫???ъ꽕怨?(dot?몆roduct)
- ?숈뒿 ?덉젙??(reward scaling, KL schedule, ROC)
"""

files["REFACTORING_EVALUATION.md"] = """# ?곌뎄怨쇱젙 ?뺣━: 由ы뙥?좊쭅 寃利?
- ROC AUC 濡쒓렇/?대?吏 ????뺤씤
- KL collapse ?щ? 異붿쟻 ?꾩슂
- z 誘쇨컧???뚯뒪???덉젙
"""

files["FINAL_EVALUATION.md"] = """# ?곌뎄怨쇱젙 ?뺣━: ?꾩옱 ?곹깭 ?됯?

- 援ъ“???꾨갑 遺뺢눼 ?꾪솕 ?μ튂 ?곸슜 ?꾨즺
- ?곗씠??紐⑤뜽/?숈뒿 ?ㅼ젙???꾩옱 紐⑺몴???뺥빀??- ?ㅼ젣 ?숈뒿 寃곌낵濡?KL 諛?AUC ?뺤씤 ?꾩슂
"""

root = Path('.')
for path, content in files.items():
    Path(path).write_text(content.strip() + "\n", encoding="utf-8")
print(f"Updated {len(files)} markdown files.")

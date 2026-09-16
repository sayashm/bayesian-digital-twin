# متن اسلایدها — پرزنتیشن دفاع (۷ سپتامبر ۲۰۲۶، ساعت ۱۱)

زمان‌بندی: ۱۵ دقیقه ارائه + ۲۰ دقیقه پرسش‌و‌پاسخ

برای هر بخش: متن پیشنهادی اسلاید، تعداد اسلاید، توضیح چیدمان، و این‌که کجا شکل/فرمول لازمه.

---

## بخش ۱ — Introduction (۲ اسلاید)

منبع: `thesis/chapt1.tex` (Problem Context, Limitation of Classical Approaches, Research Objective) + `thesis/abstract_english.tex`

### اسلاید ۱

**Title:** The Problem: From Reactive to Predictive Maintenance

**Body (bullets):**
- Industrial systems degrade with use — e.g. turbofan engines wear a little more every flight cycle
- Two extremes both fail: run-to-failure (unsafe, costly) vs. fixed-calendar replacement (wastes usable life)
- **Predictive maintenance:** use sensor data to estimate the *Remaining Useful Life* (RUL) — the number of operating cycles left before failure — and schedule maintenance on actual condition
- Most existing RUL methods (regression, deep learning) output a single number, with no confidence attached — not enough to safely support a maintenance decision

### اسلاید ۲

**Title:** Research Objective

**Body (bullets):**
- Goal: build a **probabilistic digital twin** of engine degradation, using a **particle filter** as the inference engine (Bayesian Sequential Monte Carlo)
- Hidden state: a scalar **Health Index** $h_t \in [0,1]$ (1 = healthy, 0 = failed), tracked from noisy sensor readings
- At every cycle: a full posterior *distribution* over $h_t$ — not a point value
- RUL is extracted as a **distribution** too: a median + 90% credible interval, not a single number
- Evaluated on NASA C-MAPSS (FD001, FD003), on both point accuracy AND whether the credible intervals are *calibrated*

**چیدمان پیشنهادی:**
- اسلاید ۱: عنوان بالا، ۴ بولت با فونت متوسط، شاید یه آیکون ساده برای «هواپیما/موتور توربوفن» کنار متن اول. اگه وقت داشتی یه نمودار خیلی ساده افقی با سه جعبه (Run-to-failure → Fixed calendar → Predictive maintenance) خوب میشه ولی اجباری نیست — بولت‌ها به‌تنهایی هم کافیه.
- اسلاید ۲: همینجا بهتره یه دیاگرام ساده بذاری چون این مفهوم مرکزی کل پایان‌نامه‌ست: چهار جعبه با فلش
  `Noisy sensor data → Particle Filter → Health Index posterior (h_t) → RUL distribution`
  این دیاگرام رو خودت راحت با shape های ساده‌ی پاورپوینت می‌تونی بسازی؛ فایل آماده‌ای برای این نداریم (تو `Fig/` فقط نمودار نتایجه، نه دیاگرام مفهومی).
- فرمول لازم نیست چیز پیچیده‌ای باشه — فقط همون $h_t \in [0,1]$ رو به‌صورت متن/نماد کنار بولت مربوطه بنویس، کافیه.

---
## بخش ۲ — The Data (۱ اسلاید، قبل از Methodology)

منبع: `thesis/chapt4.tex` (بخش Dataset: NASA C-MAPSS، Table 4.1 — اعداد دقیقاً از همینجا کپی شده، تغییر نده) + `data/cmapss/readme.txt` (توضیح رسمی NASA/PHM08 در مورد truncation)

### اسلاید ۳

**Title:** The Data: NASA C-MAPSS Turbofan Dataset

**Body (bullets):**
- Simulated turbofan run-to-failure data, PHM08 challenge. Two sub-datasets used: **FD001** (1 fault mode: HPC) and **FD003** (1 fault mode + a second, HPC + Fan)
- Each row = one engine-cycle: engine ID, cycle number, 3 operating settings, 21 sensor readings
- **Train set:** every engine run to failure → true RUL known exactly at every cycle
- **Test set:** each sequence cut off at some point *before* failure — true RUL given separately, one value per engine, only used for evaluation

**Table (زیرمجموعه‌ی Table 4.1 پایان‌نامه — اعداد رو عیناً از پایان‌نامه کپی کن، تغییرشون نده):**

| | FD001 | FD003 |
|---|---|---|
| Fault mode(s) | HPC | HPC + Fan |
| Train / Test engines | 100 / 100 | 100 / 100 |
| Train rows | 20,631 | 24,720 |
| Test rows | 13,096 | 16,596 |
| Test true RUL (min / median / max) | 7 / 86 / 145 | 6 / 77 / 145 |

**چیدمان پیشنهادی:**
- بالا ۴ بولت کوتاه، پایین جدول. اگه جا کم اومد، بولت دوم (فرمت ردیف‌ها) رو می‌تونی حذف کنی چون تو نوت میگیش شفاهی، لازم نیست رو اسلاید باشه.
- ردیف آخر جدول (Test true RUL) رو یه‌جوری هایلایت کن (بولد یا رنگ) — همین ردیف مستقیماً نشون میده تست truncated هست چون RUL هیچ‌جا صفر نیست و پخش شده بین ۶ تا ۱۴۵.
- شکلی لازم نیست بسازی؛ دو تا شکل مربوط به انتخاب سنسور (`fig_sensor_composite_ranking.png`, `fig_selected_sensor_trajectories.png`) تو `Fig/` هست ولی اونا مال بخش Methodology (ساخت Health Index) هستن، نه این اسلاید — نگهشون دار برای بعد.

---
## بخش ۳ — Data Analysis: Sensor Selection (۱ اسلاید)

منبع: `thesis/chapt4.tex` (زیربخش "Sensor selection" — همون بخشی که مستقیم زیر Dataset اومده، Table 4.2)، شکل آماده: `Fig/fig_sensor_composite_ranking.png`

این دقیقاً همون بخشیه که برازنده‌ی رشته‌ی خودته — نشون می‌ده چطور از ۲۱ سنسور خام، با چند معیار آماری، ۱۰ تا انتخاب شدن (نه صرفاً حدسی).

### اسلاید ۴

**Title:** Data Analysis: Selecting Informative Sensors

**Body (bullets):**
- Not every sensor carries a degradation signal — start from 21 raw sensors
- Funnel: **21** raw → **7** constant across the whole fleet, excluded → **14** informative → drop composite score < 0.40 (**2**) → drop redundant/high-VIF (**2**) → **10 selected sensors**
- Composite score = average of 5 statistical criteria (each normalised to [0,1]):
  - Mann–Kendall test — fraction of engines with a significant monotonic trend
  - Monotonicity — consistency of direction within one engine's life
  - Trendability — consistency of direction *across* engines
  - Mutual information with RUL — catches nonlinear relevance
  - Variance Inflation Factor (inverted) — flags redundancy with other sensors
- Computed across all four C-MAPSS sub-datasets (even though only FD001/FD003 are used later) — so selection isn't overfit to just these two

**Figure:** `Fig/fig_sensor_composite_ranking.png` — بار چارت ۵ معیار برای ۱۴ سنسور اطلاعاتی. آماده‌ست، فقط از پوشه‌ی `thesis/Fig/` کپیش کن تو اسلاید.

⚠️ **نکته‌ی مهم درباره‌ی این شکل:** این نمودار فقط برای FD001 رسم شده (خود کپشنش تو پایان‌نامه هم می‌گه "(FD001)")؛ فایل جدایی برای FD003 وجود نداره. ولی این به این معنی نیست که انتخاب سنسورها فقط رو FD001 انجام شده — طبق متن، composite score هر سنسور جدا جدا برای هر ۴ زیردیتاست (FD001-FD004) حساب و بعد میانگین‌گیری شده، و همون ۱۰ سنسور نهایی برای FD001 و FD003 هر دو یکسان استفاده می‌شه (Table 4.2 هم یه لیست واحده، نه جدا جدا). پس رو اسلاید بهتره زیر شکل یه زیرنویس کوچیک بذاری: *"Shown for FD001; the composite score itself is averaged across all four sub-datasets, and the same 10 sensors are used for FD003."* — این‌جوری اگه پرسیدن «این فقط FD001 نیست؟» جواب از قبل رو اسلایدت هست.

**چیدمان پیشنهادی:**
- بالا فانِل ۵ مرحله‌ای (۲۱→۷→۱۴→...→۱۰) رو به‌جای بولت ساده، به‌شکل یه ردیف افقی با فلش بذار — چون این خودش تصویریه و فهمش سریع‌تره.
- زیرش شکل `fig_sensor_composite_ranking.png` رو بذار (کل عرض اسلاید، چون شکل نسبتاً پر از جزئیاته).
- ۵ معیار آماری رو اگه جا نشد می‌تونی به‌صورت یه لیست کوچیک کنار شکل بذاری یا فقط تو نوت بگی؛ رو اسلاید فقط اسم‌هاشون کافیه.

---
## بخش ۴ — Methodology: State Space Model (۲ اسلاید)

منبع: `thesis/chapt3.tex` §3.1 (State Space Model for Turbofan Degradation → Health Index Construction → Degradation Dynamics → Observation Model → Initial Distribution)

⚠️ **نکته‌ی مهم:** رو این اسلایدها دقیقاً از نماد خود پایان‌نامه استفاده کن ($x_t \in [0,100]$ برای Health Index، $h_t=x_t/100\in[0,1]$ معادلش). این همون نمادیه که قبلاً یه‌بار (تو راند بازبینی examiner) عوض شد و باعث دردسر شد چون با نسخه‌ی فرستاده‌شده به استاد فرق داشت — پس چیزی رو از خودت تغییر نده، عیناً کپی کن.

**تصمیم گرفتیم ۲ اسلاید باشه** (نه ۳): Initial Distribution اسلاید جدا نداره دیگه — فقط یه جمله‌ی شفاهی آخر اسلاید ۶ میشه، بدون فرمول رو اسلاید. این‌جوری اسلایدها شلوغ/شلخته به نظر نمی‌رسن ولی محتوا حذف نشده، فقط از رو اسلاید برداشته شده.

### اسلاید ۵ — Health Index

**Title:** State Space Model: The Health Index

**Body (bullets):**
- Hidden state: **Health Index** $x_t \in [0,100]$ — $x_t{=}100$ healthy, $x_t{=}0$ failed (equivalently $h_t = x_t/100 \in [0,1]$)
- Built from the 10 selected sensors: each min–max normalised to $[0,100]$ using the training-fleet range, **oriented** so all sensors decrease as the engine degrades, then averaged with uniform weights
- Two constructions compared (weighted average, PCA); weighted average kept — fits better once paired with the chosen dynamics

**Formula:**
$$s_{i,t}^{\text{norm}} = \frac{s_{i,t}-\min_i}{\max_i-\min_i}\times100, \qquad Y_t^{\text{weighted}} = \sum_{i=1}^m w_i\, s_{i,t}^{\text{norm,oriented}}, \qquad w_i = 1/m$$

**Figure (پیشنهادی):** `Fig/fig_selected_sensor_trajectories.png` — نمودار trajectory های سنسورهای انتخاب‌شده، خوب نشون می‌ده قبل از ترکیب‌شدن تو Health Index، سنسورهای خام چه شکلی‌ان.

---

### اسلاید ۶ — Degradation Dynamics + Observation Model

**Title:** Degradation Dynamics & Observation Model

**Body (bullets — Dynamics):**
- Transition: $x_t = f_\theta(x_{t-1}) + v_t,\quad v_t\sim\mathcal{N}(0,\sigma_v^2)$
- **Exponential** shape chosen: $f_\theta(x) = 100-(100-x)\exp(k)$
- Why: matches the C-MAPSS simulator's *own documented* exponential damage-growth model, and is the standard wear-out shape in reliability literature — not picked from the $R^2$ table (that comparison was inconclusive, see notes)
- $k$ fitted by log-linear regression on training data; $\sigma_v$ left **free**, refined later via PMMH

**Body (bullets — Observation):**
- $y_t = x_t + w_t,\quad w_t\sim p_v(\cdot)$
- Noise density fitted on training residuals: **Gaussian** ($\sigma_w\approx9.26$) kept as default for its closed-form likelihood, even though a skew-normal fits mildly better by AIC — reported as a robustness check, not adopted

**چیدمان:** فرمول‌ها رو دو ستون کنار هم (چپ Dynamics، راست Observation) بذار، شکل لازم نیست.

**نکته:** چیزی رو این اسلاید در مورد Initial Distribution نمی‌بینی — عمداً — فقط شفاهی، آخر همین اسلاید، یه جمله بگو: مدل با $h_0\sim\mathcal{N}(1,\sigma_0^2)$ با $\sigma_0=0.05$ شروع می‌شه (یه پخش کوچیک، نه یه نقطه‌ی ثابت)، و این‌که همه‌ی پارامترها (وزن‌ها، $k$، نویز مشاهده) یه‌بار از دیتای train فیت و ثابت شدن؛ فقط $\sigma_v$ (نویز فرآیند) آزاده و تو بخش بعد (PMMH) دقیق میشه. متن کامل این جمله تو `speaker_notes.md` هست.

---
## بخش ۵ — Particle Filter Algorithm (۱ اسلاید)

منبع: `thesis/chapt3.tex` §3.2 (Bootstrap Particle Filter Algorithm — Sequential Importance Resampling، Filtering Recursion، Effective Sample Size، Algorithm Summary)

با همون منطق بخش قبل (نگه‌داشتن هرچی می‌شه شفاهی، رو اسلاید فقط چیزی که واقعاً لازمه)، این بخش رو یه اسلاید در نظر گرفتم.

### اسلاید ۸

**Title:** Bootstrap Particle Filter

**Body (bullets):**
- **Bootstrap filter:** propose from the transition model itself → no custom proposal needed, weight reduces to just the observation likelihood
- Per cycle, per particle: **predict** (draw new state) → **weight** (multiply in log-space — over up to 350 cycles, linear-space likelihoods underflow to exactly zero)
- Without resampling, effective sample size collapses to ≈1 within 5–10 cycles (weight degeneracy)
- **Adaptive resampling:** resample only when $\widehat{\mathrm{ESS}}_t < \tau N$, $\tau=0.5$ — balances particle diversity against computational cost
- Each of the 100 engines filtered **independently** (shared fitted dynamics, separate posterior per engine)

**Formula:**
$$w_t^{(i)} \propto w_{t-1}^{(i)}\, p\big(y_t \mid x_t^{(i)}\big), \qquad \widehat{\mathrm{ESS}}_t = \left(\sum_{i=1}^N \big(w_t^{(i)}\big)^2\right)^{-1}$$

**چیدمان پیشنهادی:**
- یه دیاگرام حلقه‌ای ۴ مرحله‌ای بذار: **Predict → Weight (log-space) → Check ESS → Resample if low** (با یه فلش برگشت به Predict) — این خودش جای خیلی از توضیح شفاهی رو می‌گیره و بصریه.
- فرمول‌ها رو زیر دیاگرام یا کنارش، کوچیک.
- شکل آماده‌ای برای این بخش نداریم؛ دیاگرام رو خودت با shape ساده بساز.

---
## بخش ۶ — RUL Extraction from the Particle Posterior (۲ اسلاید)

منبع: `thesis/chapt3.tex` §3.3 (RUL Extraction → Forward Simulation → **Calibrating the Failure Threshold** → Censoring)

⚠️ اسلاید ۱۰ (Calibrating the Failure Threshold) مهم‌ترین اسلاید کل متودولوژی‌ست — خود پایان‌نامه می‌گه این "the single change that makes the extracted RUL both accurate and calibrated". یعنی همین‌جا دلیل اصلی دستاورد بزرگ نتایج (coverage از تقریباً صفر به ۰.۷۵/۰.۹۹) شکل می‌گیره. حتماً وقت کافی بهش بده.

### اسلاید ۹ — Forward Simulation

**Title:** RUL Extraction: Forward Simulation

**Body (bullets):**
- After the last observed cycle, propagate **every particle forward** under the same transition model, drawing new process noise at each step
- Each particle stops when it crosses **its own** failure threshold → one RUL sample per particle
- Collect all $N$ samples → empirical **RUL posterior**: report median + 90% credible interval, not a single number
- No new sensor data here — this is pure model-driven extrapolation past the last observation

**Formula:**
$$R^{(i)} = \min\{\, r\ge1 : x_{T+r}^{(i)} \le x_{\text{fail}}^{(i)} \,\}$$

**چیدمان:** فرمول کوتاهه، شکل لازم نیست — می‌تونی یه mini-sketch کوچیک کنار فرمول بذاری (چندتا خط زیگزاگ که به یه خط افقی «threshold» می‌رسن) ولی اجباری نیست.

---

### اسلاید ۱۰ — Calibrating the Failure Threshold (کلیدی‌ترین اسلاید)

**Title:** Calibrating the Failure Threshold

**Body (bullets):**
- Naive choice ($x_{\text{fail}}=0$) is wrong on **3 counts**, all measured from training data:
  1. Real engines don't fail at HI $=0$ — training fleet fails around HI $\approx 0.256$
  2. Must calibrate in the filter's **posterior space**, not raw observation space — the posterior runs systematically above the observed HI; comparing against an observation-space number double-counts that gap
  3. The threshold itself is **uncertain** — drawn independently **per particle**, not one fixed number
- This turns real engine-to-engine variability in failure point into honest posterior width

**Formula:**
$$x_{\text{fail}}^{(i)} \overset{\text{iid}}{\sim} \mathcal{N}(\mu_{\text{fail}}, \sigma_{\text{fail}}^2)$$
هر دو پارامتر فقط از دیتای train فیت می‌شن.

**این جمله رو حتماً بگو (شفاهی، یا حتی رو اسلاید اگه جا داشت):** *"This is the single change that makes the extracted RUL both accurate and calibrated."* — این جمله‌ی خودِ پایان‌نامه‌ست و دقیقاً پلی به بخش نتایجه.

**چیدمان:** این اسلاید مهم‌ترینه، پس اگه لازمه یکم فونت رو کوچیک‌تر کن ولی چیزی حذف نکن؛ ۳ دلیل رو می‌تونی به‌شکل ۳ باکس عمودی کوچیک بذاری (نه بولت ساده) تا بصری‌تر باشه و وزنش حس بشه.

---
## بخش ۷ — Parameter Estimation via PMMH (۲ اسلاید)

منبع: `thesis/chapt3.tex` §3.4 (Why Point Calibration Is Not Enough → Pseudo-Marginal MH → Likelihood Estimator → Prior/Proposal → Fleet-Wide Results)

این بخش از نظر ریاضی سنگین‌ترین بخش پایان‌نامه‌ست (pseudo-marginal correctness, log-space random walk, Jacobian). رو اسلاید فقط ایده‌ی اصلی + نتیجه‌ی نهایی رو گذاشتم؛ جزئیات فنی (چرا Jacobian لازمه، چرا log-space) رفت تو Q&A چون احتمال زیاد به‌عنوان سوال شفاهی پرسیده میشه، نه چیزی که رو اسلاید لازم باشه.

### اسلاید ۱۱ — PMMH: The Idea

**Title:** Why Point Calibration Isn't Enough → PMMH

**Body (bullets):**
- $\sigma_v$ (process noise) was the **one** parameter with no closed-form estimator — hand-set, never fitted from data
- Idea: treat $\sigma_v$ as a random variable with a posterior $p(\sigma_v\mid y_{1:T})$ — but this needs an intractable marginal likelihood (same integral that motivated the particle filter itself, one level up)
- **Pseudo-Marginal MH:** substitute a noisy but **unbiased** estimate of that likelihood inside an ordinary Metropolis–Hastings step → still converges to the *exact* correct posterior
- The bootstrap filter **already computes this estimate for free**, as a byproduct of ordinary filtering — no new machinery needed

**Formula:**
$$\alpha(\sigma_v\to\sigma_v') = \min\left(1,\ \frac{\widehat{Z}(\sigma_v')\,p(\sigma_v')\,q(\sigma_v\mid\sigma_v')}{\widehat{Z}(\sigma_v)\,p(\sigma_v)\,q(\sigma_v'\mid\sigma_v)}\right)$$

**چیدمان:** فرمول یه‌خطیه، جای زیادی نمی‌گیره. شکل لازم نیست.

---

### اسلاید ۱۲ — Fleet-Wide PMMH: Results

**Title:** Fleet-Wide PMMH: Results

**Body (bullets):**
- Run **per engine**, jointly estimating $(k, \sigma_v)$: $N=3200$ particles, $1000$ iterations
- Convergence rule: acceptance rate $\ge 0.10$; chains that don't converge fall back to the regression-fitted rate — every engine still contributes an estimate
- Converged: **31/100** (FD001), only **8/100** (FD003 — its second fault mode makes the likelihood surface noisier) — reported honestly as a finding, not hidden
- Fleet result: $\sigma_v \approx 0.66$ on both datasets (vs. hand-set default $0.5$) — **confirms the hand-set value understated the true process noise**
- Growth rate: $k\approx0.0039$ (FD001) vs. $\approx0.0018$ (FD003)

**چیدمان:** بولت چهارم ($\sigma_v\approx0.66$) رو هایلایت کن — این عددیه که بعداً تو Results دوباره میاد و به calibration بهتر ربط داره.

---
## بخش ۸ — Similarity-Based Reference Library (۲ اسلاید)

منبع: `thesis/chapt3.tex` §3.5 (Per-Engine Reference Library → Similarity Matching via MMD → Streaming Deployment → Simplifications relative to Cai et al. 2020)

این دقیقاً همون بخشیه که خودت گفتی بهترین/مهم‌ترین کارتونه — و دلیلش هم خوبه: این بخش مستقیماً همون ایده‌ی «digital twin» رو که تو اسلاید ۲ گفتی (یه مدل که یه asset مشخص رو mirror می‌کنه، نه میانگین کل fleet) عملی می‌کنه. تو نوت اسلاید ۱۳ این ارتباط رو صریح گفتم — خوبه که سر جلسه هم همین پل رو بزنی، چون نشون می‌ده کار بدون برنامه پیش نرفته، دقیقاً منطق «digital twin» رو دنبال کرده.

### اسلاید ۱۳ — Per-Engine Library + Similarity Matching

**Title:** Similarity-Based Reference Library

**Body (bullets):**
- The pooled fleet-wide rate is a **population average**, not any real engine's own trajectory — biased toward the low end (pulled by fast-degrading short-lived engines *and* slow-degrading long-lived engines at once)
- **Fix:** fit **one growth rate per training engine** instead of one pooled rate → a reference library of 100 individual rates $\{k_u\}$
- On FD001: per-engine mean $k=0.003727$ vs. the pooled fit's $k=0.002762$ — confirms the pooling bias directly
- At test time: match the test engine's observed Health Index so far against the library via **Maximum Mean Discrepancy**, average the $k{=}5$ nearest matches → a **personalized** growth rate for that specific engine

**Formula:**
$$\widehat{\mathrm{MMD}}(X,Y) = \sqrt{\tfrac{1}{t^2}\!\sum \kappa(x_i,x_j) + \tfrac{1}{t^2}\!\sum \kappa(y_i,y_j) - \tfrac{2}{t^2}\!\sum \kappa(x_i,y_j)}\ ,\quad \kappa(a,b)=\exp\!\big(\!-\tfrac{(a-b)^2}{2\ell^2}\big)$$

**چیدمان:** شکلی لازم نیست. اگه جا داشتی، یه جمله‌ی کوچیک زیر اسلاید بذار که این بخش رو به تعریف digital twin تو اسلاید ۲ وصل کنه — همون‌جوری که تو گفتی، این نشون می‌ده کار مطابق ماهیت مسئله (per-asset، نه fleet-average) پیش رفته.

---

### اسلاید ۱۴ — Streaming Deployment

**Title:** Streaming Deployment: One-Shot vs. Periodic

**Body (bullets):**
- A real digital twin doesn't have the full trajectory upfront — Health Index arrives **one cycle at a time**
- **One-shot:** match once after a 20-cycle warm-up, hold the matched rate fixed for the rest of the engine's life (cheapest; closest to Cai et al. 2020)
- **Periodic:** re-match every 20 cycles as more data arrives — both the matched rate and matched engines can change
- Before the first match: both variants fall back to the pooled fleet dynamics (nothing more specific known yet)
- Which variant wins is decided **empirically** in Results, on the full test fleet — not assumed in advance

**چیدمان:** فرمول/شکل لازم نیست؛ می‌تونی دو حالت رو به‌شکل دو تایم‌لاین کوچیک افقی کنار هم بذاری (one-shot: یه نقطه‌ی match ثابت؛ periodic: چندتا نقطه‌ی match تکرارشونده) — اختیاریه.

---
---
## بخش ۹ — Results (نتایج)

منبع: `thesis/chapt4.tex` §4.4 (Results: Health State Estimation), §4.5 (Results: RUL Prediction), §4.6 (Results: Comparison with Baseline), §4.7 (Results: Similarity-Based Reference Library)

**مشورت صادقانه در مورد تعداد اسلاید:** این فصل، نتیجهٔ نهایی و محوری کل تز هست — همون‌جایی که همهٔ methodology (state-space model، PF، PMMH، library) نتیجه‌شون رو نشون می‌دن. بنابراین به نظرم ارزش ۴ اسلاید رو داره، نه ۲-۳ تا:
1. Health Tracking (اعتبارسنجی مقدماتی — فیلتر درست کار می‌کنه؟)
2. RUL Prediction: Fixed vs. PMMH (ablation اصلی)
3. Similarity Library: نتیجهٔ نهایی (بهترین config)
4. Comparison with Baseline — این اسلاید رو خود تز صراحتاً «the comparison central to this thesis» می‌نامه، پس به نظرم سزاواره یه اسلاید مستقل و برجسته باشه، نه ادغام‌شده با بقیه.

**نگرانی زمان‌بندی:** تا الان (بخش ۱-۸) ۱۴ اسلاید داریم؛ این ۴ تا می‌شه ۱۸ تا، و هنوز Limitations/Conclusions مونده. برای ۱۵ دقیقه یکم فشرده می‌شه. پیشنهادم اینه: این ۴ تا رو نگه داریم چون محتوای محوریه، ولی جدول‌محور و کم‌کلمه باشن (تو الان تند و مطمئن ارائه می‌دی، پس فشردگی مشکلی نیست) و توضیح کلامی رو بار اصلی بذاریم، نه متن روی اسلاید. اگه دیدی زمان کم میاری، اولین کاندیدای حذف/ادغام، اسلاید ۱۵ (Health Tracking) هست — می‌تونی خیلی خلاصه‌ش کنی یا حتی شفاهی رد بشی و بری سراغ RUL، چون ESS/HI-RMSE بیشتر برای Q&A مفیدن تا ارائه.
از این ۶ شکل موجود (`fig_*_bestworst_*`) فقط پیشنهاد می‌کنم **یکی** استفاده بشه (اسلاید ۱۷، library) تا ریتم رو کند نکنه — بقیه رو برای Q&A نگه دار (اگه پرسیدن "می‌تونی نشونم بدی روی یه موتور خاص؟" آماده باشی نشونشون بدی).

---

### اسلاید ۱۵ — Health State Tracking

**Title:** Does the Filter Actually Track Health Well?

**Body (bullets):**
- Pooled Health-Index RMSE across all $13{,}096$ FD001 test cycles: **$0.082$** under PMMH vs. $0.092$ under fixed parameters — the filtered estimate stays within roughly a tenth of a unit of the observed Health Index throughout
- Particle cloud stays diverse, not degenerate: mean $\widehat{\mathrm{ESS}}_t/N = 0.79$ (SD $0.02$) fleet-wide — adaptive resampling ($\tau=0.5$) keeps at least $250/500$ particles supporting the posterior at all times
- Even the **worst-tracked engine** (FD001 engine 1 — only 31 of its 143 cycles observed) has its health tracked well; its large error shows up only later, in the RUL projection — not here
- One systematic weakness: trajectory RMSE correlates with observation length ($r=-0.58$) — shortest-observed quartile averages $111$ cycles RMSE vs. $61$ for the longest → a degradation-*model* limitation, not a filter-health problem

**چیدمان:** شکلی لازم نیست (اختیاری: `fig_pmmh_bestworst_fd001.png` اگه وقت داری). این اسلاید صرفاً "prerequisite trust" رو می‌سازه قبل از رفتن سراغ RUL.

---

### اسلاید ۱۶ — RUL Prediction: Fixed vs. PMMH

**Title:** RUL Accuracy: Does Estimating the Parameters Help?

**Table:**

| Dataset | Configuration | RMSE (cycles) | MAPE | Coverage |
|---|---|---|---|---|
| FD001 | Fixed parameters | 98.3 | 1.35 | 0.49 |
| FD001 | **PMMH (primary)** | **52.9** | **0.77** | **0.68** |
| FD003 | Fixed parameters | 160.4 | 2.45 | 1.00 |
| FD003 | **PMMH (primary)** | **104.6** | **1.72** | 1.00 |

**Body (bullets):**
- PMMH roughly **halves** the error on both datasets (FD001: RMSE $52.9$ vs $98.3$; FD003: $104.6$ vs $160.4$)
- Calibration diverges by dataset — **FD001**: coverage rises $0.49\to0.68$ (still below nominal $0.90$, mildly overconfident, but no longer the $0.000$ of the earlier, uncalibrated threshold) — **FD003**: coverage stays at $1.00$ on *both* configs — over-conservative under the noisier two-fault-mode signal (opposite failure mode)
- Residual weakness: short-observed engines still over-predicted (engine 1: median $226$ vs. true $112$ — better than fixed parameters' $336$, but not solved) → motivates the next step

**چیدمان:** جدول وسط، بولت‌ها زیرش. شکلی لازم نیست مگر بخوای تضاد fixed-vs-PMMH رو تصویری نشون بدی (`fig_fixed_bestworst_fd001.png` در کنار `fig_pmmh_bestworst_fd001.png`) — ولی برای حفظ ریتم پیشنهاد نمی‌کنم اینجا.

---

### اسلاید ۱۷ — Similarity Library: Final Results

**Title:** The Reference Library: Closing the Gap Further

**Table:**

| Dataset | Method | RMSE (cycles) | MAPE | Coverage |
|---|---|---|---|---|
| FD001 | Fixed parameters | 98.3 | 1.35 | 0.49 |
| FD001 | PMMH | 52.9 | 0.77 | 0.68 |
| FD001 | Library, periodic | 47.3 | 0.75 | 0.70 |
| FD001 | **Library, one-shot** | **44.3** | **0.63** | **0.75** |
| FD003 | Fixed parameters | 160.4 | 2.45 | 1.00 |
| FD003 | PMMH | 104.6 | 1.72 | 1.00 |
| FD003 | Library, periodic | 107.5 | 1.55 | 0.99 |
| FD003 | **Library, one-shot** | **100.1** | **1.54** | 0.99 |

**Body (bullets):**
- Monotonic improvement on **both** datasets, fixed → PMMH → one-shot library: FD001 RMSE $98.3\to52.9\to44.3$, coverage $0.49\to0.68\to\mathbf{0.75}$ — the *closest to nominal $0.90$ of any configuration, at the lowest RMSE*
- **One-shot beats periodic** on both fleets (FD001: $44.3$ vs $47.3$, ahead on 59/100 engines; FD003: $100.1$ vs $107.5$) → adopted as the thesis's final configuration
- The library improves point accuracy *and* calibration together — a per-engine matched rate re-centers the RUL trajectory without needing to widen the interval to compensate

**Figure:** `fig_library_bestworst_fd001.png` (or fd003) — best-tracked engine's RUL median follows the truth almost exactly (final-cycle error 2 cycles); worst-tracked (shortest observation window) still over-predicts — the honest remaining limit.

**چیدمان:** جدول بالا (کوچیک، فونت کوچیک اوکیه چون فقط برای reference چشمی)، بولت‌ها وسط، شکل پایین یا کنار.

---

### اسلاید ۱۸ — Comparison with Baseline (نتیجهٔ محوری تز)

**Title:** The Central Trade-Off: Point Accuracy vs. Calibrated Uncertainty

**Table:**

| Dataset | Model | RMSE (cycles) | MAPE | Coverage |
|---|---|---|---|---|
| FD001 | Linear regression baseline | **32.0** | **0.53** | — |
| FD001 | Library, one-shot (this thesis) | 44.3 | 0.63 | **0.75** |
| FD003 | Linear regression baseline | **58.2** | **1.07** | — |
| FD003 | Library, one-shot (this thesis) | 100.1 | 1.54 | **0.99** |

**Body (bullets):**
- A simple OLS regression on the same 10 sensors is the **sharper point predictor** on both datasets — reported plainly, not downplayed
- But the baseline produces a single number with **no expression of confidence** — asking "does its 90% interval cover the truth?" isn't a poor result, it's not even a question that can be posed
- The particle filter concedes a real point-accuracy margin **in exchange for** a calibrated credible interval a bare point estimate structurally cannot provide
- For a maintenance decision — where acting too late costs far more than acting too early — that calibrated uncertainty is the quantity that actually matters

**چیدمان:** این اسلاید رو به‌عنوان نقطهٔ اوج بخش نتایج معرفی کن — شاید حتی با تاکید کلامی که «این خلاصهٔ کل استدلال تزه». شکلی لازم نیست؛ جدول کوچیک + ۴ بولت کافیه.
---
## بخش ۱۰ — Limitations (محدودیت‌ها)

منبع: `thesis/chapt5.tex` §5.1 (What the Particle Filter Delivers — بخشی که تناقض ظاهری short-lived/shortage-of-evidence رو حل می‌کنه)، §5.3 (Sources of the Residual Miscalibration)، §5.4 (Other Limitations)

**مشورت:** این بخش رو ۲ اسلاید کافی می‌دونم — نه بیشتر. محتواش صرفاً «خودانتقادی صادقانه»‌ست، نه نتیجهٔ جدید، پس نباید طولانی بشه. با توجه به اینکه الان ۱۸ اسلاید داریم و این می‌شه ۲۰ تا، فکر می‌کنم بیشتر از این جا نداریم. یه نکتهٔ مثبت: این دقیقاً همون تمی هست که قبلاً decision‌گیری‌هات (مثل انتخاب Gaussian به‌جای skew-normal، یا r_max) رو خوب توجیه کرده — نشون دادن محدودیت‌ها به‌جای مخفی‌کردنشون، خودش یه امتیازه، نه ضعف. تو Q&A هم همینو تکرار کن.

---

### اسلاید ۱۹ — Residual Miscalibration: Why FD001 and FD003 Differ

**Title:** What's Left of the Calibration Gap

**Body (bullets):**
- Two fixes already address most of the interval problem: PMMH estimates $\sigma_v$ from data (fixes the fixed-parameters overconfidence), the library corrects the *mean* rate (fixes the systematic over-prediction)
- **FD001 (coverage 0.75, still below nominal 0.90):** a single matched rate per engine still can't capture an engine whose true rate has no close analogue in the training library — residual over-prediction on short-lived engines pushes the interval slightly too high and too narrow
- **FD003 (coverage 0.99, above nominal):** $\sigma_v\approx0.66$ on *both* datasets — so it's not a bigger noise estimate. Two compounding causes instead: (1) FD003's longer horizon means the forward-simulated Monte-Carlo spread accumulates over more cycles at the same per-cycle noise, (2) matching under two coexisting fault modes is noisier — a mode-2 engine matched to mode-1 analogues (or vice versa) gives a more variable top-$k$ rate
- Underlying both: **PMMH convergence was engine-dependent** — the MH chain accepted/converged on only **31/100 FD001** and **8/100 FD003** engines (regression fallback used otherwise) → the fleet-wide $\sigma_v$ each dataset uses rests on an uneven, and on FD003 small, set of converged posteriors
- Both point to the same next step: better per-engine PMMH convergence + a genuinely per-engine (not fleet-wide) $\sigma_v$

**چیدمان:** شکلی لازم نیست. ۴-۵ بولت با تاکید بصری روی FD001 vs FD003 (شاید دو رنگ یا دو ستون کوچیک).

---

### اسلاید ۲۰ — Other Limitations

**Title:** Other Limitations, Stated Plainly

**Body (bullets):**
- **Similarity matching simplifies Cai et al. (2020)** in three specific ways (already flagged in the Library section): scalar Health Index instead of the full sensor feature matrix; top-$k$-nearest instead of a formal statistical accept/reject test; a matched rate held fixed after matching rather than tracked jointly with the state online — none shown to be harmless in general, just tractable and good enough to deliver a real fleet-wide improvement
- **PMMH mixing is engine-dependent** (same 31/100, 8/100 numbers as previous slide) — a single global random-walk step size isn't equally well matched to every engine's likelihood surface
- **No operating-condition-aware normalisation** → the pipeline does **not** extend to FD002/FD004 (multiple operating conditions) as-is — a real scope limitation on how broadly these conclusions generalise beyond the single-operating-condition case examined here

**چیدمان:** ۳ بولت، هر کدوم می‌تونه bold روی عبارت کلیدی (Similarity matching / PMMH mixing / Operating conditions) داشته باشه تا سریع اسکن بشه. شکلی لازم نیست.

# Acoustics of pianos: physical modeling, simulations and experiments
*Antoine Chaigne, Juliette Chabassier, Nicolas Burban — SMAC 2013, KTH Stockholm*
*HAL Id: hal-00873639*

---

## Hlavní oblasti

Článek popisuje komplexní fyzikální model grand piana (Steinway D), jehož komponenty jsou:

1. **Nelineární model struny** — Timoshenkova stiff-string rovnice s geometrickou nelinearitou (velká výchylka)
2. **Přenos sil na ozvučnici** — vazba přes bridge, transversální i longitudinální složka
3. **Ozvučnice** — ortotropní Reissner-Mindlin deska s žebry a bridge jako lokální nehomogenitou
4. **3D akustické pole** — FEM s Perfectly Matched Layers (volné pole / anechoic)
5. **Fantomové parciály** — kvadratická a kubická nelinearita struny generuje kombinační frekvence
6. **Zig-zag okrajová podmínka** — vysvětlení otáčení roviny polarizace struny v čase

---

## Principy paperu

### 2.1 Model struny — Timoshenko + geometrická nelinearita

Paper používá soustavu tří vázaných PDE (rovnice 3):

- `u_s` — příčný transversální posun
- `v_s` — longitudinální posun (klíčové: tato složka je v lineárním modelu nulová)
- `φ_s` — úhel průřezu (Timoshenkův smyk)

Zdroj nelinearity: člen `(EA − T₀) · ∂u/∂x / sqrt(...)` v longitudinální rovnici. Tato nelinearita **propojuje** transverzální a longitudinální pohyb struny.

**Důsledky geometrické nelinearity:**
- Longitudinální vibrace jsou buzeny transverzálním pohybem → přítomny i při malé výchylce
- Fantomové parciály na frekvencích `n·f_T ± m·f_L` (sums/differences)
- Density fantomů roste s amplitudou (forte >> piano)
- Precurzory (krátkodobé vysokofrekvenční jevy při útoku) jsou způsobeny disperzí strunové tuhosti

### 2.2 Numerické metody

- Diskrétní energetický formalismus zajišťující dlouhodobou stabilitu
- Časový krok Δt = 10⁻⁶ s (1 MHz vzorkování pro FEM)
- Dvě různá θ-schémata pro lineární a nelineární části (CFL podmínka)
- Ozvučnice: modální dekompozice (2400 módů 0–10 kHz), semianalytické časové řešení
- Úvaha: modální tlumení je diagonální (nezávislé módové ztráty)

### 2.3 Ozvučnice

- Ortotropní deska (jiná tuhost v ose vláken a kolmo), proměnná tloušťka
- Žebra a bridge modelovány jako lokální nehomogenity v tloušťce a elasticitě
- Výsledek: spektrum vibrací ozvučnice obsahuje jak módy ozvučnice (dominantní pod 800 Hz), tak longitudinální složky struny a fantomové parciály

### 2.4 Strunové polarizace a zig-zag podmínka (sekce 4)

Reálné piano: u každé struny se v průběhu tónu mění rovina polarizace (vertikální → eliptická → horizontální). Horizontální složka vidí vyšší admitanci bridge → vertikální složka zaniká rychleji → double-decay efekt.

Příčina: **zig-zag end condition** — dvě jehly v bridge svírají s rovinou různý úhel α. Pro α ∈ 20°–60° dochází k mikro-klouzání struny na jehle, které postupně přenáší energii do horizontální polarizace.

Matematický model: diskrétní massa-spring soustava (2 hmotnosti, 3 pružiny), prediktorkorektor metoda pro okrajové podmínky.

### 2.5 Vliv amplitudy hammeru na spektrum

- Piano (V_H = 0.5 m/s): šířka spektra ~5 kHz, fantomové parciály slabé
- Forte (V_H = 3.0 m/s): šířka ~7 kHz, fantomové parciály výrazně přítomny
- Nonlinearita hammeru (plst) rozšiřuje spektrum samostatně, bez příspěvku strunové nelinearity

---

## Srovnání se současnou implementací

| Princip z paperu | Stav v synth | Poznámka |
|---|---|---|
| **Inharmonicita struny** `f_k = k·f₀·√(1+B·k²)` | **Implementováno** | `note_params.h`: `f_hz` přímo z params.json (extrahováno z nahrávek). `B` uložen. Vzorec hardcoded v komentáři `resonator_voice.h` |
| **Bi-exponenciální útlum** `A₀(a₁·e^{-t/τ₁} + (1-a₁)·e^{-t/τ₂})` | **Implementováno** | `PartialParams`: `tau1`, `tau2`, `a1`. Per-sample multiply: `env1_[k] *= d1_[k]`. Ekvivalentní formě z paperu |
| **Beating strun** (inter-string detuning) | **Implementováno** (frekvenčně) | `beat_hz` / `beat_depth` na parciál. Detuning `±beat_hz/2` per string (STRING_SIGNS). Chybí coupling efekt (polarizace) |
| **Nelineární model struny** (geometrická nelinearita) | **Chybí** | Synth je čistě aditivní (lineární součet parciálů). Žádná PDE simulace, žádné longitudinální vlny |
| **Fantomové parciály** | **Chybí** | Parciály jsou extrahovány z reálných nahrávek — fantomové frekvence tedy *mohou* být obsaženy v params.json pokud byly detekovány při analýze, ale nejsou generovány fyzikálně. Žádné `n·f_T ± m·f_L` výpočty |
| **Precurzory** (attack dispersion transients) | **Chybí** | Pouze 3 ms lineární onset ramp — čistě kosmetický. Fyzikálně správné precurzory by vyžadovaly nelineární strunový model |
| **Ozvučnice** — modální model | **Chybí** | Náhrada: 8-band peaking biquad EQ derivovaný ze spektrální křivky nahrávky (`biquad_eq.cpp`). Neobsahuje módy ozvučnice jako samostatné entity |
| **Módová struktura ozvučnice** (pod 800 Hz) | **Částečně** | EQ křivka je log-spaced 64-bodová — může zachytit globální výkonové rozložení, ale ne individuální módové vrcholy |
| **3D akustické pole** (PML FEM) | **Není v scope** | Synth generuje stereo signál; prostorové šíření zvuku je mimo scope (DAW/room IR) |
| **Strunová polarizace** (zig-zag, double-decay) | **Chybí** | Beat simulation řeší výchylku frekvence, ale ne přenos energie mezi polarizacemi. Double-decay efekt (rychlejší útlum vertikální složky) není modelován |
| **Hammer — nelineární jarní model** (Stulov) | **Částečně** | Velocity je mapován na amplitudu přes `vel_gamma` (power law `(v/127)^γ`). Chybí dynamická spektrální změna dle síly úderu — forte by mělo generovat výrazně více vysokých frekvencí |
| **Coupling bridge** (transversální + longitudinální) | **Chybí** | Vše přenášeno jako jednostranný aditivní signál do stereo pair |
| **Timoshenko string model** (shear + rotational inertia) | **Chybí** | Inharmonicita B extrahována empiricky, fyzikální původ (EI, ρ, A, κAG) není explicitně v synth |
| **Velocity-dependent spectral bandwidth** | **Chybí** | `harmonic_brightness` parametr škáluje `1 + hb·log₂(k)`, ale není funkcí velocity. V paperu forte = širší spektrum = jiná dynamika než piano amplifikovaná |
| **Šum (noise)** | **Implementováno** | `NoiseParams`: `attack_tau_s`, `floor_rms`, `centroid_hz`. 1-pole LP filtr, nezávislé L/R kanály. Reprezentuje hammerstrike noise |
| **Stereo model** | **Implementováno** | Multi-string panning (`STRING_SIGNS`, `computeStringAngles`), M/S width, Schroeder all-pass decorrelation |
| **Velocity layer interpolace** | **Implementováno** | `interpolateNoteLayers` v `voice_manager.cpp`, 8 vrstev, blendování params |

---

## Doporučení

### Priorita 1 — Spektrální dynamika velocity

Paper explicitně ukazuje (Figures 6–7), že forte vs. piano není jen škálování amplitudy — forte má **výrazně širší spektrum** a silnější fantomové parciály. Současný synth aplikuje `(v/127)^γ` uniformně na všechny parciály.

**Doporučení:** Přidat velocity-dependent spectral tilt. Příklad: pro každý parciál `k` aplikovat `gain_k = (vel/127)^(γ + δ·log₂(k))` kde `δ` je extrahovatelný z analýzy nahrávek.

### Priorita 2 — Double-decay a polarizační efekt

Paper identifikuje dvojí útlum (fast/slow) jako důsledek dvou polarizací s různou admitancí bridge. Bi-exponenciální model (`tau1`, `tau2`, `a1`) již tuto strukturu zachycuje — ale parametry nejsou motivovány fyzikálně. `tau2` by ideálně odpovídal horizontální polarizaci (pomalejší útlum).

**Doporučení:** Při extrakci parametrů rozlišit první (rychlý) a druhý (pomalý) segment decay jako fyzikální entity. Ověřit, zda `tau1` koresponduje s dobou dominance vertikální složky.

### Priorita 3 — Fantomové parciály jako aproximace

Implementace plných nelineárních PDE je pro real-time synth nereálná. Avšak fantomové parciály lze aproximovat jako **deterministické aditivní oscilátory**:
- Frekvence: `f_phantom = n·f_T ± m·f_L` kde `f_L ≈ 2640 Hz` (F3) — viz Fig. 5
- Amplituda: škáluje se s `(vel/127)^2` (kvadratická nonlinearita) nebo `^3` (kubická)
- Tyto parciály by rozšířily spektrum forte tónů v rozsahu 1–5 kHz

### Priorita 4 — Precurzory

Pro útok struny jsou precurzory způsobeny disperzní rychlostí vysokofrekvenčních složek (kratší vlnové délky dorazí na bridge dříve). Lze aproximovat krátkodobým additivním shlukem na `f_T·k` pro k > 30, s razantním útlumem (τ < 5 ms). Obsaženy v nahrávce, tedy teoreticky v params.json jako parciály s vysokým `a1` a nízkým `tau1`.

### Priorita 5 — Modální charakter ozvučnice

8-band peaking EQ je příliš hrubý pro zachycení rezonančních vrcholů ozvučnice (Q >> 1.4). Pro frekvence pod 1 kHz existuje hustý soubor módů ozvučnice, viditelných v reálném signálu jako ostré vrcholy. Vhodná náhrada: **IIR resonator bank** na nízkých frekvencích (100–1000 Hz) místo EQ, poháněný longitudinálním buzením.

---

## Shrnutí

Chaigne et al. SMAC 2013 prezentuje úplný fyzikální model grand piana zahrnující nelineární struny (Timoshenko PDE + geometrická nelinearita), modální ozvučnici, a 3D akustické pole. Klíčovým fyzikálním poznatkem je, že **spektrální bohatost piana pochází z nelineárního propojení transverzální a longitudinální vibrace** — fantomové parciály, precurzory, a velocity-dependent šíře spektra jsou přímými důsledky.

Současná C++ implementace (`resonator_voice.cpp`) je **aditivní lineární syntéza** s parametry extrahovanými z reálných nahrávek. Správně implementuje inharmonicitu, bi-exponenciální útlum a inter-string beating. Tyto tři prvky jsou fyzikálně motivované a zachycují hlavní perceptuální charakter piana.

Zásadní chybějící prvky oproti paperu:
1. Nelineární strunová dynamika → žádné skutečné fantomové parciály, žádné precurzory
2. Velocity-dependent spektrální šíře (forte ≠ piano × konstanta)
3. Polarizační double-decay s fyzikálně motivovanými časovými konstantami
4. Modální charakter ozvučnice (diskrétní rezonance, ne EQ)

Implementace je vhodná pro real-time synth, ale bez velocity-dependent spektrálního rozšíření bude forte tón slyšet jako zesílené piano, ne jako fyzikálně odlišný dynamický stav.

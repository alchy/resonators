# Modeling and simulation of a grand piano
**Chabassier, Joly, Chaigne — Journal of the Acoustical Society of America, 134, p. 648, 2013**
HAL: hal-00873089

---

## Hlavní oblasti

Paper je globální fyzikálně přesný model celého grand piana řešený v časové doméně. Zahrnuje pět propojených subsystémů:

1. **Struny** — nelineární tuhý Timoshenko beam (geometrická nelinearita + tuhostní disperze)
2. **Kladívko** — nelineární disipativní kontaktní síla (Hertzův zákon s hysterezí)
3. **Zvuková deska** — 2D ortotropní Reissner-Mindlin deska s žebry a mostem jako heterogenity
4. **Vazba strun-zvuková deska na mostku** — přenos jak příčných, tak longitudinálních vln
5. **Akustické pole** — 3D linearizované Eulerovy rovnice v neohraničeném prostoru, ořezané PML

Numerická formulace je energeticky konzervativní (klesající energie pro disipativní systém), používá higher-order finite elements v prostoru a speciální schémata v čase.

---

## Principy paperu

### 1. Strunový model — Timoshenko beam s geometrickou nelinearitou

Struny jsou popsány třemi prostorovými proměnnými: příčné posunutí `u_s`, longitudinální posunutí `v_s`, smykový úhel `φ_s`. Inharmonicita vychází z Timoshenkova modelu (přesnějšího než Euler-Bernoulli):

```
f^trans_ℓ = ℓ·f0 · sqrt(1 + ε·ℓ²) + O(ℓ⁵)
ε = (π²/2L²) · (EI/T0) · (1 - T0/EA)
```

Tento koeficient ε se od standardního Fletcherova `B` mírně liší. Timoshenkův model zajišťuje asymptotické omezení rychlosti příčné vlny pro vysoké frekvence — fyzikálně správnější i numericky stabilnější než Euler-Bernoulli.

Geometrická nelinearita způsobuje:
- **Precursor**: longitudinální vlna (rychlost ~2914 m/s) dorazí k mostku ~14× dříve než příčná (~209 m/s)
- **Phantom partials**: kvadratické a kubické kombinace frekvencí příčných parciálů (sums/differences)
- **Amplitudová závislost frekvence**: fundamentála klesá s amplitudou (nonlinear softening)

Tlumení strun: frekvenčně závislé, modelováno jako součet konstantního členu `2ρA·Ru` a kvadratického členu `2T0·ηu·f²`. Koeficienty určeny experimentálně.

### 2. Model kladívka

Kontaktní síla kladívko–struna:
```
F^H_i(t) = K^H_i · Φ(u_s - ξ)  +  R^H_i · d/dt Φ(u_s - ξ)
kde Φ(d) = (ξ - d)^p_+   (p ∈ [1.5, 3.5])
```

Disipativní člen `R^H_i` modeluje hysterezní chování plsti. Kontaktní síla je rozložena po krátké délce struny přes spreading funkci `δ_H(x)`. Kladívko může na danou notu zasahovat 1, 2 nebo 3 struny s mírně odlišnými napětími (detunování strun je tak fyzikálně zabudováno přímo do modelu).

### 3. Model zvukové desky

2D Reissner-Mindlin deska: tři proměnné `u_p`, `θ_x`, `θ_y`. Materiál je ortotropní (smrk pro desku a žebra, buk pro mostek). Parametry jsou prostorově proměnné — mostek a žebra jsou modelovány jako lokální heterogenity s proměnnou tloušťkou (6–95 mm). Tlumení je řešeno modálně (diagonální matice tlumení), 2 400 módů pokrývá audio rozsah do 10 kHz.

### 4. Vazba strun-zvuková deska

Klíčový fyzikální detail: struny svírají s horizontální rovinou malý úhel `α` (výška mostku + prohnutí desky). Tím se longitudinální pohyb struny transformuje na příčnou sílu na zvukové desce:

```
F_b(t) = cos(α)·[příčná složka]  +  sin(α)·[longitudinální složka]
```

Z nelineárního rozvoje síly vychází, že kvadratické a kubické členy generují phantom partials a longitudinální složky viditelné na spektru zvukové desky, ale **nikoli** na samotné struně.

### 5. Akustické pole a záření

Linearizované Eulerovy rovnice v 3D neohraničeném prostoru, ořezané Perfectly Matched Layers (PML). Lem piana je tuhý reflektor. Viskotermální ztráty ve vzduchu jsou zanedbány. Výpočet 1 sekundy zvuku (do 10 kHz) vyžadoval 24 hodin na 300-jádrovém clusteru.

### 6. Numerická schémata

- Struny: higher-order FEM v prostoru + hybridní θ-schéma v čase (θ=1/4 pro longitudinální/smyk, θ=1/12 pro příčné — minimální disperze). Nelineární část: energy-preserving gradient scheme (implicitní v čase).
- Zvuková deska: semi-analytická metoda v čase po modální dekompozici.
- Akustika: explicitní leapfrog v čase + higher-order FEM v prostoru.
- Vazby: Schur komplement zajišťuje oddělené updatování subsystémů při zachování energetické konzistence.

---

## Srovnání se současnou implementací

| Princip (paper) | Stav v synth | Poznámka |
|---|---|---|
| **Inharmonicita — Timoshenko model** | Implementována (zjednodušeně) | `f_k = k·f0·sqrt(1 + B·k²)` v `note_params.h` — de facto Fletcherův vzorec. Koeficient `B` extrahován z měření. Timoshenko vs. Euler-Bernoulli rozdíl zanedbatelný v audio rozsahu pro většinu not. |
| **Frekvence parciálů uložené přímo** | Plně implementováno | `PartialParams.f_hz` je přímá frekvence z analýzy, ne vypočtená z B. Přesná shoda s realitou. |
| **Bi-exponenciální útlum parciálů** | Plně implementováno | `env = A0·(a1·e^(-t/τ1) + (1-a1)·e^(-t/τ2))`. Paper jen naznačuje frekvenčně závislé tlumení (konstantní + kvadratický člen) — synth jde dále s per-partiálním bi-expem. |
| **Detunování strun (inter-string beating)** | Implementováno | `beat_hz` per partial, `STRING_SIGNS[n_strings]` pro ±beat/2 nebo -beat/2, 0, +beat/2. Paper má detunování jako fyzikální parametr (`T0,i` odlišné pro každou strunu) — synth to approximuje jako additivní frekvenční offset. Ekvivalentní pro beating, ale nezahrnuje amplitudovou modulaci (Weinreich coupling). |
| **Weinreichovo coupling strun (sdružené struny)** | Chybi | Paper cituje Weinreicha (1977): sdružené struny přes zvukovou desku vytvářejí double-decay envelope, asymetrické doublets. Synth nemá vazbu strun přes desku — každá struna vibruje nezávisle se stejnou obálkou. |
| **Model kladívka (Hertzův zákon + hystereze)** | Chybi | Synth předpokládá hotové parciály s danými amplitudami z params.json. Neexistuje žádný model kontaktní síly — vstup kladívka je parametrizován přes `vel_gain = (vel/127)^vel_gamma`. Tvar ataku (precursor, hammer pulse shape) není modelován. |
| **Precursor (longitudinální vlna)** | Chybi | Výsledek geometrické nelinearity a vysoké rychlosti longitudinálních vln. Synth neimplementuje longitudinální pohyb. Zvukový projev precursoru (krátký click před hlavním tónem) chybí. |
| **Phantom partials** | Chybi | Vznikají z nelineárních kombinací příčných frekvencí. Synth používá čistě lineární additivní syntézu — žádné kombinační tóny. Perceptivně důležité zejm. v basu při forte/fortissimo. |
| **Amplitudová závislost frekvence** | Chybi | Paper: fundamentála klesá s klesající amplitudou (nonlinear softening). Synth: frekvence parciálů jsou konstantní — extrahované ze steady-state části záznamu. |
| **Zvuková deska — modální odezva** | Nepřímo aproximováno | Paper: 2 400 módů desky excitovaných silou na mostku generují hustý spektrální obsah v transientu. Synth: spektrální EQ (8-band peaking biquad z 64-point křivky) aproximuje průměrnou přenosovou funkci desky, ale bez modální struktury a bez časové proměnnosti (neviditelné krátké modální transienty). |
| **Vazba strun-deska-vzduch (záření)** | Chybi | Synth neobsahuje žádný model záření. Výstupní signál je přímo součet parciálů — ekvivalent "vnímání struny bez desky a vzduchu". Prostorová informace (zvuk ve vzduchu) je plně nahrazena stereo post-processingem. |
| **Soundboard modes in transient** | Chybi | Paper: soundboard módy jsou viditelné a slyšitelné v transientu zejm. pro vyšší noty (velký rozestup parciálů strun). Synth: EQ je statická křivka — nezachycuje časově klesající modální transienty. |
| **Frekvenčně závislé tlumení strun** | Implementováno (jinak) | Paper: `α(f) = 2ρA·Ru + 2T0·ηu·f²`. Synth: per-parciální `τ1, τ2` — fakticky implicitně kóduje frekvenční závislost tlumení, přesnější než papírový empirický model. |
| **Počet strun per nota (1/2/3)** | Plně implementováno | `n_strings` 1 pro bas, 2 nebo 3 pro vyšší noty. Shodně s paperem. |
| **Detunování strun — fyzikální původ** | Zjednodušeně | Paper: odlišná klidová napětí `T0,i` každé struny. Synth: `beat_hz` extrahovaný z analýzy. Výsledný efekt beating je ekvivalentní, ale model neobsahuje fyzikální příčinu (výrobní tolerance napnutí). |
| **Stereo panning strun** | Implementováno (nad rámec paperu) | MIDI-závislý úhel středu + `pan_spread`. Paper stereo rozmístění neřeší (mono fyzikální model). |
| **Šum (noise) při ataku** | Implementováno (nad rámec paperu) | 1-pólový LP filtrovaný šum s decay envelope. Paper šum neobsahuje — modeluje deterministické fyzikální signály. |
| **Spektrální EQ** | Implementováno (nad rámec paperu) | 8-band RBJ biquad z 64-point křivky. Approximuje průměrnou přenosovou funkci desky. Plynulý fade pod `eq_freq_min` chrání před kontaminací pokojovou akustikou. |
| **Onset ramp (anti-click)** | Implementováno | 3ms lineární ramp. Artefakt synth architektury, paper nemá analogii. |
| **Schroeder all-pass stereo decorrelation** | Implementováno (nad rámec paperu) | Per-kanálový all-pass s MIDI-závislou sílou. Paper stereo post-processing neobsahuje. |
| **M/S stereo width** | Implementováno (nad rámec paperu) | `width_factor × stereo_boost`. Paper nemá analogii. |
| **Velocity model** | Zjednodušeně | Synth: `(vel/127)^vel_gamma`. Paper: prvotní rychlost kladívka `v^H_0` — fyzikálně přímý parametr. |
| **Sustain pedal** | Implementováno (voice management) | `VoiceManager` drží aktivity strun po uvolnění klávesy. Paper pedál nemoděluje (izoluje jednotlivé noty). |
| **Energetická konzistentnost numeriky** | Nerelevantní | Paper řeší numerickou stabilitu pro PDE. Synth je sample-domain additive synthesis — stabilita je triviální. |

---

## Doporučení

### Kritická chybějící fyzika (high-impact)

**1. Weinreichovo coupled-string double decay**
Toto je pravděpodobně nejdůležitější chybějící prvek pro autentický zvuk. Synth modeluje N nezávislých strun se stejnou obálkou — realita je asymetrická dvojitá obálka (rychlý + pomalý decay, různá časová konstanta pro "in-phase" a "out-of-phase" módy). Projevuje se jako charakteristický "zpěv" piana. Implementace: dva beat-coupled oscilátory se separátními amplitudami pro symetrický/antisymetrický mód, s různými `τ1/τ2`.

**2. Phantom partials (zjednodušená verze)**
Přidání vybraných kombinačních tónů (f_i + f_j, 2f_i) jako slabých extra parciálů do params.json z analýzy spektrogramu — nevyžaduje nelineární solver, pouze identifikaci phantomů z reálných nahrávek.

**3. Modální struktura transientu zvukové desky**
Spectral EQ je statická křivka — nezachycuje rychle doznívající soundboard modes v prvních 50–100 ms. Aproximace: časově proměnná EQ s rychlým decay na vyšších frekvencích, nebo separátní přidání několika krátkých "soundboard mode" oscilátorů s rychlým τ (~0.05–0.2 s).

### Střední priorita

**4. Amplitudová závislost frekvence**
Pro forte/fortissimo hru se fundamentála measurably posouvá dolů. Implementace v synth: `f_k(t) = f_k · (1 - δ·env(t))` kde `δ` malý parametr (~10-50 cents range, registrově závislý).

**5. Precursor (longitudinální click)**
Krátký (~0.5–2 ms) broadband burst před nástupem tónu, výrazný zejm. v basu při forte. Lze aproximovat velmi krátkým broadband noise burst s rychlým decay, oddělený od hlavního noise modelu.

### Nízká priorita pro produkci

**6. Model kladívka**
Aktuální parametrizace přes `vel_gamma` je pragmaticky dostatečná pro produkci. Full Hertzův model by byl nutný pro fyzikální predikci, nikoli pro synthesis z existující sample banky.

**7. Záření v 3D**
Plně nahrazeno stereo post-processingem — pro headphone/speaker output nepotřebné.

---

## Shrnutí

Paper Chabassier et al. (2013) je full PDE model piana v časové doméně — fyzikálně nejkomplexnější dostupný model. Tento synth a paper sdílí inharmonicitu (koeficient B), frekvenčně závislý útlum (bi-exp místo papírového quadratic law) a základní detunování strun. Synth jde nad rámec paperu v oblasti stereo post-processingu, EQ aproximace přenosové funkce desky a explicitního noise modelu.

**Tři klíčové fenomény z paperu zcela chybějí v synth:**
- Weinreichovo coupled-string coupling (double decay) — největší vliv na "přirozený" zvuk
- Phantom partials — sluchově znatelné v basu při forte
- Modální transienty zvukové desky — výrazné v transientu zejm. treble

Synth nepotřebuje replikovat celý PDE aparát — cílem je fenoménologická věrnost. Doporučeným prvním krokem je implementace Weinreichova double-decay modelu, který lze odhadnout přímo z existujících nahrávek v params.json (fitting dvou oddělených obálek pro symetrický a antisymetrický mód každého parciálu).

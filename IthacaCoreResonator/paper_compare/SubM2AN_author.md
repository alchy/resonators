# Time Domain Simulation of a Piano. Part 2: Numerical Aspects
*Chabassier, Duruflé, Joly — ESAIM: M2AN, 2016, 50(1), pp. 93–133 (hal-01085477)*

---

## Hlavní oblasti

Článek je druhou částí ze dvou papers věnovaných numerické simulaci klavíru. Zatímco první část (CCJ14) konstruuje fyzikální PDE model, tato část se zabývá:

1. **Prostorovou diskretizací** — variační formulace, Galerkin FEM pro struny (1D), modální spektrální aproximace pro desku (soundboard), hexahedrální FEM pro akustiku (3D)
2. **Časovou diskretizací** — implicitní θ-schemata pro nelineární struny, semi-analytické řešení pro soundboard, explicitní leapfrog pro akustiku
3. **Stabilitou přes energetické metody** — zachování diskrétní energie jako podmínka stability
4. **Algoritmem Schurovy komplementace** — oddělení bloků (struny / soundboard / akustika) pro výpočetní efektivitu
5. **Numerickými výsledky** — simulace Steinway D, srovnání modelů strun, ověření s experimentem

---

## Principy paperu

### 1. Plný fyzikální PDE systém (rovnice 3a–3g)

Paper modeluje klavír jako **coupled PDE system** zahrnující:

- **Kladívko** (hammer): nelineární kontaktní síla `F_i = k_H * Φ_H(e_i)^p + r_H * dΦ_H/dt`, kde `Φ_H(d) = d^α` pro α > 1
- **Struny** (3 DOF na bod): Timoshenko beam model — příčný posun `u_i`, podélný posun `v_i`, rotace průřezu `θ_i`. Geometricky přesný nelineární model zahrnuje:
  - inharmonicitu (stiff string, EI)
  - podélné vlny (EA)
  - smykové vlny (AG)
  - nelinearity (kvadratické — zdroj phantom partials)
- **Soundboard** (Reissner–Mindlin deska): modální rozklad do M = 2 400 módů, tlumení jako spektrální operátor `f(A_p)`
- **Akustika** (3D): wave equation pro tlak p a rychlost V, PML na hranicích
- **Vazba bridge**: Lagrangeovy multiplikátory `F_ip` (vertikální síla) a `F_ir` (horizontální síla) — přenos příčných i podélných vln

### 2. Prostorová diskretizace

| Subsystém | Metoda | Parametry (Steinway D) |
|---|---|---|
| Struny | 1D Lagrange FEM, stupeň r_s = 4 | Δx = L/200, ~2 400 DOF/strunu |
| Soundboard | Modální (spektrální), 2D Galerkin Q4 | M = 2 400 módů, h = 2 cm |
| Akustika | 3D Gauss–Lobatto hexahedra Q4 | h = mesh size, ~117 milionů DOF |

Klíčové vlastnosti: mass lumping (Gauss–Lobatto body), 4. řád přesnosti, cílový frekvenční rozsah [0, 10] kHz.

### 3. Časová diskretizace strun — θ-schéma s energetickou konzervací

Schéma (87) je **kombinace** tří vrstev:

1. **1/12-θ-schéma** pro d'Alembertovu část `A^{s,D}` — 4. řád přesnosti, podmínka stability `Δt * sp((M_h^s)^{-1} A_h^{s,D})^{1/2} ≤ √3/2`
2. **Implicitní β=1/4 Newmark** pro perturbační část `A^{s,p}` (zahrnuje vlny podélné a smykové) — nepodmíněně stabilní, 2. řád
3. **Conservative approximate gradient** `∇̃_h U` pro nelineární člen — zachovává diskrétní energii přesně (rovnice 83), metodika Strauss–Vasquez

Výsledné schéma (87) je 2. řádu a stabilní pod CFL podmínkou (79), přičemž tato podmínka není zpřísněna nelinearitou.

### 4. Vazba kladívko–struny (schéma 96)

Nelineární kontaktní síla diskretizována přes **konzervativní aproximaci** `Φ_H^+(a,b)` (rovnice 95), čímž se zachovává diskrétní energie interakce kladívko–struny.

### 5. Soundboard — semi-analytické řešení (modální)

Díky modálnímu rozvoji je rovnice soundboardu pro každý mód **analyticky řešena** v každém časovém kroku `[t^n, t^{n+1}]` při zamražení source termů. Schurova komplementace eliminuje `F_ip` (coupling force) výpočtem matice `S_p = I + (Δt/2) R_t C^T_{pp} (M_h^p)^{-1} C_{pp}` (jednou, předem) a skalární konstanty `ρ_{p,h}`.

### 6. Akustika — explicitní leapfrog na staggered grid (schéma 125)

Standardní explicitní schéma; nákladné (117 M DOF), proto paralelizováno (300 jader, 24 h / 1 s zvuku).

### 7. Energetická stabilita (Proposition 1–6)

Diskrétní celková energie `E^{n+1/2}_{s,p,a,h}` je **nerostoucí** (rovnice 131), pokud `β ≥ 1/4` a CFL podmínka (79) je splněna. Toto garantuje stabilitu celého coupled schématu (130).

### 8. Výsledky — 5 fyzikálních cílů

| Cíl | Výsledek |
|---|---|
| Obj. 1: Frekvenčně selektivní útlum | Potvrzen (obr. 7) |
| Obj. 2: Nelinearity oddělují piano/ff | Potvrzen spektrogramem (obr. 7) |
| Obj. 3: Přenos podélných vln na soundboard | Potvrzen precursor vlnou (obr. 6, 9) |
| Obj. 4: Inharmonicita | Potvrzen (srovnání Model 1 vs 3) |
| Obj. 5: Phantom partials (příduchové parciály) | Potvrzen pouze v nelineárním Model 3 |

---

## Srovnání se současnou implementací

| Princip paperu | Stav v C++ synth | Poznámka |
|---|---|---|
| **Inharmonicita** — stiff string freq. `f_k = k * f0 * sqrt(1 + B*k^2)` | **Implementováno** | `NoteParams::B`, `f_hz` předpočítáno v `extract_params.py`; synth přebírá výsledné `f_hz` přímo |
| **Beating strun** — choir detuning Δf mezi 1–3 strunami | **Implementováno** | `PartialParams::beat_hz`, `STRING_SIGNS[n_strings]` v `resonator_voice.cpp`, konfigurováno `beat_scale` |
| **Bi-exponenciální útlum** — `A0*(a1*e^{-t/τ1} + (1-a1)*e^{-t/τ2})` | **Implementováno** | `env1_, env2_, d1_, d2_` v `ResonatorVoice`; `PartialParams::a1, tau1, tau2` |
| **Nelinearity strun** (geometricky přesný Timoshenko model) | **Neimplementováno** | Synth je čistě additivní (oscilátory s fixními frekvencemi); žádná vlnová rovnice ani PDE pro struny |
| **Phantom partials** (příduchové parciály, `f_m ± f_n`) | **Neimplementováno** | Vznikají z kvadratické nelinearity strun; v aditivním modelu by musely být přidány jako extra parciály s vlastními amplitudami a útlumy |
| **Podélné vlny** (`v_i`) a jejich přenos přes bridge | **Neimplementováno** | Přímý fyzikální přenos chybí; podélné frekvence nejsou modelovány jako separátní parciály |
| **Soundboard** — Reissner–Mindlin deska, M = 2 400 módů | **Neimplementováno** | Soundboard zcela chybí jako fyzikální model; jeho efekt je zachycen empiricky skrze spektrální EQ (`BiquadEQ`) z reálných nahrávek |
| **Soundboard damping** — spektrální operátor `f_d(A_p)` | **Parciálně** | Útlumové koeficienty τ1, τ2 extrahované z nahrávek implicitně zahrnují vliv soundboardu; žádná explicitní modální struktura |
| **Akustické záření** — 3D wave equation, PML | **Neimplementováno** | Akustika není modelována; audio výstup jsou přímo oscilátory + EQ |
| **Vazba bridge** (Lagrangeovy multiplikátory `F_ip, F_ir`) | **Neimplementováno** | Vazba struna–soundboard–vzduch chybí celá |
| **Kladívko** — nelineární kontaktní síla `k_H * Φ_H(e)^α` | **Neimplementováno** | Útok je modelován pouze onset rampou (3 ms lineární nárůst); žádná kontaktní mechanika |
| **Nelineární dynamika** — rozdílná spektra pro piano/ff | **Parciálně** | Synth interpoluje různé velocity layers z nahrávek (8 vrstev); fyzikální nelinearita kladívka není přítomna |
| **Energeticky stabilní schéma** (θ-scheme + conservative gradient) | **Irelevantní** | Synth není časový PDE solver; netlumené oscilátory s exponenciálně klesající amplitudou jsou vždy stabilní |
| **Inharmonicita v linearizovaném spektru** (CFL/stability podmínky) | **Irelevantní** | Nepřímý model; frekvence jsou extrahované, ne řešené z PDE |
| **Frekvenčně závislý útlum** (selektivní, Obj. 1) | **Implementováno** | Odlišné τ1, τ2 pro každý parciál zachytí toto chování; vysoké parciály mají kratší τ |
| **Stereo dekortace** (Schroeder all-pass) | **Implementováno** | Specifické pro tento synth, paper toto neřeší |
| **Spektrální EQ** (BiquadEQ z LTASE analýzy) | **Implementováno** | Empirická náhrada za fyzikální soundboard; 8 pásem peaking biquad |
| **3-strunový sbor** (N_s = 1, 2, 3) | **Implementováno** | `n_strings`, panning, `str_norm_` |
| **Velocity dynamika** — `v_H` iniciální rychlost kladívka | **Parciálně** | `vel_gamma` power law; paper má spojité fyzikální kladívko, synth interpoluje diskrétní VL layers |

---

## Doporučení

### Priorita 1 — Phantom partials jako aditivní rozšíření

Paper (Model 3, sekce 6.2) ukazuje, že **phantom partials jsou nutné** pro fyzikální věrnost u nízkých not. Jejich frekvence jsou `f_m ± f_n` inharmonického spektra. V aditivním modelu je lze přidat jako extra parciály s:
- frekvencí `f_p = f_m + f_n` (sum tones) nebo `|f_m - f_n|` (difference tones)
- amplitudou odpovídající kvadratické nelinearitě (malá, ~30–40 dB pod dominantními parciály)
- zvláštním útlumem (dekají jinak než flexurální parciály)

Extrakce těchto frekvencí z nahrávek je možná (`analysis/extract_params.py` by mohl identifikovat peak clustering mimo `k * f0 * sqrt(1+Bk^2)` a přiřadit je jako phantom).

### Priorita 2 — Podélné parciály jako separátní řada

Paper (sekce 6.2, 6.3) dokumentuje **podélné parciály** (`f_n^L`) jako vizuálně odlišné v spektrogramu (delší, tenčí, jiný útlum). Jejich frekvence závisejí pouze na délce, hustotě a Young's modulu — ne na napětí. Pro synth: extrahovat tyto parciály ze sample banky jako separátní skupinu s vlastní τ1^L.

### Priorita 3 — Fyzikálnější model útoku

Paper (sekce 4.3) ukazuje, že kontaktní interakce kladívko–struna trvá **pouze ~5 ms** a je silně nelineární. Současná 3ms lineární rampa je příliš hrubým modelem. Lepší aproximace:
- Kompresní attack tvar (exponenciální nárůst, pak plateau, Herrmann/Giordano model)
- Velocity-závislá délka a tvar útoku (rychlé údery = kratší kontakt = více energie v HF)

### Priorita 4 — Dvojitý útlum (double decay) jako fyzikální cíl

Paper (sekce 6.3, obr. 10) explicitně identifikuje **double decay fenomén** jako klíčový ověřovací benchmark. Bi-exponenciální model v synth (`a1, tau1, tau2`) toto zachycuje, ale parametry musí být extrahované z reálných nahrávek s dostatečnou délkou (>5 s). Ověřit, zda `analysis/extract_params.py` fituje bi-exp správně pro basy.

### Priorita 5 — Soundboard modal coupling jako konvoluce

Kompletní Reissner–Mindlin model je výpočetně nedosažitelný v reálném čase (300 CPU, 24 h/s). Avšak **modální odezva soundboardu** může být aproximována jako LTI filtr (IRF ze soundboard módů). Tento přístup (analogický Paper's modální dekompozici) je kompatibilní s aditivním modelem: soundboard přidává frekvenčně závislý gain a fázi. Současný BiquadEQ to aproximuje, ale chybí časová proměnnost (soundboard modes mají vlastní útlum).

---

## Shrnutí

Paper Chabassier, Duruflé, Joly (2016) je **referenční implementace fyzikálně přesné simulace** klavíru metodou konečných prvků. Jde o jiný přístup než synth v `IthacaCoreResonator` — PDE solver vs. aditivní syntéza.

**Co synth implementuje správně** (ze záměru paperu): inharmonicitu, bi-exponenciální útlum s frekvenčně závislými časovými konstantami, choir detuning pro beating, 1–3 struny s stereo panoramou.

**Co zásadně chybí**: nelinearity strun (phantom partials), fyzikální soundboard (žádná modální struktura, jen empirický EQ), akustické záření, fyzikální model kladívka (jen onset rampa).

**Architektonicky**: paper ukazuje, že pro věrnou reprodukci **phantom partials a podélných partiálů** je nelineárnost nutností — tyto jevy jsou principiálně nedosažitelné čistě lineárním aditivním modelem bez přidání explicitních extra parciálů se správnými frekvencemi a amplitudami. Toto je nejdůležitější gap pro zlepšení autenticity syntézy.

Zároveň paper potvrzuje, že **inharmonicita a beating jsou fyzikálně správně modelovány** v současném synthu — tyto jevy jsou lineárního původu a aditivní model je zachycuje adekvátně.

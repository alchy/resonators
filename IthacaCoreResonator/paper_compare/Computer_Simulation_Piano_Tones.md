# Computer Simulation of Piano Tones and Design of Virtual Piano System

**Zhang, Guo, Pan, Shi — Applied Mathematics and Nonlinear Sciences, 9(1), 2024**
DOI: 10.2478/amns-2024-1424

---

## Hlavní oblasti

Článek pokrývá čtyři témata:

1. **Fyzikální model vibrací strun** — odvození vlnové rovnice, Fourierovy koeficienty, harmonická struktura
2. **Syntéza timbru spektrální obálkovou metodou** — multiplikativní model `T(p,t) = A(p,t) · E(p,t)`, tříúsekový obálkový model
3. **Analýza timbru pomocí STFT + rekonstrukce Cauchyho funkcí** — extrakce feature matice, envelope funkce
4. **Návrh virtuálního výukového systému** — iPad klient, WeChat backend, MySQL databáze, hodnocení žáků

Jádro paperu relevantní pro syntézátor jsou části 2.1 a 2.2. Část 2.3 a kapitola 3 (testování systému, uživatelské role, server load) jsou pro syntézátorový engine irelevantní.

---

## Principy paperu

### 1. Vlnová rovnice strunné vibrace

Paper odvozuje standardní vlnovou rovnici pro ideální strunu bez tuhosti:

```
∂²u/∂t² = a² · ∂²u/∂x²
```

Řešení je Fourierova řada:

```
u(x,t) = Σ_n [Cn·cos(nπa/l·t) + Dn·sin(nπa/l·t)] · sin(nπx/l)
```

Koeficienty Dn závisí na počáteční rychlosti v místě nárazu kladívka x₀. Výsledkem je, že amplituda n-tého parciálu klesá jako **1/n** (piano) oproti **1/n²** (trhanec). Paper explicitně uvádí toto jako důvod bohatšího timbru piana.

**Chybí:** inharmonicita (člen EI·∂⁴u/∂x⁴ pro tuhá vlákna), tlumení, obálkový bi-exponenciální model.

### 2. Tříúsekový obálkový model

Paper navrhuje envelope funkci se třemi fázemi:

```
E(p,t) = { k₁·t,           t ∈ [0, t₁)           — attack
          { k₂·t + b,       t ∈ [t₁, t₂)          — initial decay
          { k₃·exp(-a·t),   t ∈ [t₂, Tc)           — slow decay
```

Tento model je kvazi-fyzikální: attack segment je lineárně rostoucí, decay je exponenciální. Obálkový tvar `f(x) = x²·eˣ` je zmíněn jako alternativní aproximace.

**Chybí:** bi-exponenciální decay (dva oddělené mechanismy útlumu), per-parciální decay konstanty závislé na čísle parciálu k.

### 3. Model timbru: spektrální obálková metoda

Multiplikativní model:

```
T(p,t) = A(p,t) · E(p,t)
```

kde `A(p,t)` je spektrální charakteristika (sinus série s koeficienty c(p,i)), `E(p,t)` je časová obálka. Tento přístup je ekvivalentní additivní syntéze s per-parciálními obálkami.

### 4. STFT analýza a Cauchyho rekonstrukce

Paper používá STFT pro extrakci spektrálních vlastností (okno 30 ms, kompatibilní s nejnižší frekvencí 32 Hz > 25.6 Hz min. frekvence piana). Timbr je rekonstruován váženou Cauchyho funkcí:

```
S_i(jω) = A_i · σ_i² / (σ_i² + (ω - ω_i)²)   pro |ω - ω_i| ≤ Δω_i
```

5 oktáv (17 harmonických) je dostatečných pro ideální eficienci; 50 harmonických překračuje lidský sluch.

### 5. Vibrace strun — typy

Paper klasifikuje 4 typy vibrací: příčná (hlavní), podélná, zdvojená frekvence (neideální ukotvení), torzní. Všechny přispívají k bohatosti harmonické struktury.

---

## Srovnání se současnou implementací

| Princip z paperu | Stav v IthacaCore | Poznámka |
|---|---|---|
| **Inharmonická frekvence parciálů** `f_k = k·f0·√(1+B·k²)` | Implementováno | Paper neobsahuje inharmonicitu — je to nadstavba nad paperem. `PartialParams.f_hz` nese přesnou inharmonickou frekvenci extrahovanou z WAV. |
| **Harmonická amplituda 1/n** (piano vs. 1/n² pro trhanec) | Implementováno implicitně | Amplitudy `A0[k]` jsou extrahovány přímo z reálných nahrávek (ne z 1/n vztahu). Fakticky odpovídá realitě lépe než paperový vzorec. |
| **Tříúsekový obálkový model** (attack + 2× decay) | Částečně / jinak | Synth používá bi-exponenciální model: `env = a1·exp(-t/τ₁) + (1-a1)·exp(-t/τ₂)`. Attack je 3ms lineární ramp pro prevenci kliku, ne fyzikální onset. |
| **Per-parciální exponenciální decay** | Implementováno, přesněji | Každý parciál má vlastní `tau1`, `tau2`, `a1` (bi-exp). Paper má pouze jeden globální `exp(-a·t)` pro celý tón. |
| **Mono vs. bi-exp rozlišení** | Implementováno | `PartialParams.mono` — mono parciály (bez beatingu) používají single-exp, vícestrunnné bi-exp. Paper toto neřeší. |
| **Inter-string beating** | Implementováno, nadstandard | Základní STRING_SIGNS[±0.5] detuning na `beat_hz`. Paper vůbec beating neobsahuje — je to fyzikální rozšíření nad rámec paperu. |
| **Inharmonicita koeficient B** | Implementováno, nadstandard | `NoteParams.B` z reálné extrakce. Paper nemá člen tuhosti (ideální struna). |
| **Spektrální charakteristická funkce A(p,t)** | Implementováno | Additivní součet sinů s `A0[k]` — ekvivalent paperu. |
| **STFT analýza pro extrakci parametrů** | Implementováno v Pythonu | `analysis/extract_params.py` používá STFT/FFT extrakci reálných parametrů z WAV. Přístup analogický paperu. |
| **Cauchyho rekonstrukce timbru** | Neimplementováno | Synth nepoužívá Cauchyho funkce. Místo toho přímá additivní syntéza se změřenými amplitudami — přesnější. |
| **Spektrální EQ korekce** | Implementováno, nadstandard | `BiquadEQ` — 8-pásmový peaking EQ designovaný z 64-bodové křivky z `params.json`. Paper toto nemá. |
| **Šum (noise layer)** | Implementováno, nadstandard | `NoiseParams` — 1-pólový LP šum s decay. Paper šum nemodeluje. |
| **Stereo model** | Implementováno, nadstandard | Per-string equal-power pan, Schroeder all-pass decorrelace, M/S width. Paper je mono. |
| **Velocity dynamika** | Implementováno | `vel_gain = (vel/127)^vel_gamma`. Paper velocity neřeší. |
| **Sustain pedál** | Implementováno | `sustained_notes`, delayed noteOff. Paper neřeší. |
| **Podélné / torzní vibrace strun** | Neimplementováno | Paper je zmiňuje jako příčinu harmonického bohatství, ale nesyntézuje je. Synth tyto složky také neobsahuje. |
| **Vlnová rovnice — neideální ukotvení (frequency-doubling)** | Neimplementováno | Paper zmiňuje oktávovou vibraci způsobenou neideálním ukotvením. V syntu není modelováno. |
| **Rezonance těla nástroje** | Neimplementováno | Paper zmiňuje "resonance system" jako součást timbru. Synth nemá model těla nástroje — EQ je náhrada měřená na výstupním signálu. |

---

## Doporučení

### Co paper přináší relevantního

1. **Poměr 1/n amplitud** — paper formálně odůvodňuje, proč piano má bohatší timbr než trhanec. Aktuální implementace to překračuje (reálně měřené amplitudy), ale porozumění závislosti amplitudy na čísle parciálu je užitečné při debugování.

2. **Tříúsekový obálkový model** — paperový model `k₁·t → k₂·t+b → k₃·exp(-a·t)` popisuje reálný fyzikální onset lépe než aktuální 3ms lineární ramp. Pro vyšší fyzikální věrnost by bylo vhodné implementovat parametrický onset per-note: reálný "rise time" piana se liší podle noty (basy mají pomalejší nástup). Toto je zanedbané.

3. **Klasifikace vibrací** — podélné vibrace produkují sub-harmonické a přidávají hloubku basovému tónu. Torzní vibrace přidávají texturní složky. Obojí lze hrubě aproximovat šumovou vrstvou s laděným spektrálním centroidem — což synth již má. Explicitní modelování by vyžadovalo fyzikální FEM, což je mimo rozsah.

### Slabiny paperu (a jejich implikace pro projekt)

- **Ideální struna bez tuhosti** — paper ignoruje inharmonicitu (klíčovou pro realistický piano tón). Aktuální synth má B-koeficient extrahovaný z dat — to je správné.
- **Jednokomponentní decay** — paper `k₃·exp(-a·t)` je zjednodušení. Bi-exponenciální model v syntu je fyzikálně správnější (Chabassier/Bank papers).
- **Žádný beating** — paper neuznává detuning mezi strunami trichordů/bichordů. To je hlavní příčina "syntetického" zvuku. Synth to řeší.
- **Paper je primárně edukační systém**, ne profesionální syntézátor. Výsledky (94.81% detekce not) jsou metriky pro vzdělávací scoring, ne pro audio kvalitu.
- **Cauchyho rekonstrukce** je zbytečná aproximace v kontextu projektu — přímá additivní syntéza ze změřených parametrů je přesnější a rychlejší.

### Konkrétní akce

| Priorita | Akce | Zdroj |
|---|---|---|
| Střední | Implementovat per-note attack time (real onset rise time z WAV analýzy) místo pevného 3ms rampu | Paper eq. (9), kapitola 2.2.3 |
| Nízká | Prozkoumat, zda šumová vrstva (`NoiseParams`) dobře approximuje podélné+torzní vibrace | Paper sekce 2.1.2 |
| Nízká | Dokumentovat odvození 1/n amplitudového vztahu v `analysis/extract_params.py` jako sanity check | Paper eq. (6) |

---

## Shrnutí

Paper Zhang et al. (2024) je **primárně edukační práce** o virtuálním výukovém systému piana, jejíž fyzikální základ je základní až zjednodušující. Fyzikální model (ideální struna, jeden exponenciální decay, mono výstup, žádný beating, žádná inharmonicita) je podmnožinou toho, co aktuální IthacaCore synth implementuje.

**Aktuální synth překračuje paper ve všech klíčových oblastech:**
- inharmonicita (B-koeficient),
- bi-exponenciální per-parciální decay,
- inter-string beating pro trichord/bichord,
- stereo model s decorrelací a M/S width,
- 8-pásmová spektrální EQ korekce,
- šumová vrstva s LP filtrem.

Hodnota paperu pro projekt spočívá v konsolidaci fyzikální intuice (1/n amplitudy, tříúseková obálka, klasifikace typů vibrací) a potvrzení správnosti additivní syntézní architektury. Konkrétní technický přínos je omezený — paper nenabízí nic, co by vyžadovalo změnu architektury nebo opravilo chybu v současné implementaci.

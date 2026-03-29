# MAESSTRO: A sound synthesis framework for Computer-Aided Design of piano soundboards

**Autoři:** Benjamin Elie, Xavier Boutillon, Juliette Chabassier, Kerem Ege, Bernard Laulagnet, Benjamin Trévisan, Benjamin Cotté, Nicolas Chauvat
**Konference:** ISMA 2019, Detmold, Německo
**Zdroj:** HAL hal-02281818v2

---

## Hlavní oblasti

MAESSTRO je CAD framework zaměřený na **virtuální návrh piánových zvukových desek** (soundboard). Papír řeší celý fyzikální řetězec od kladívka po vyzařovaný zvuk, s důrazem na to, jak mechanické vlastnosti soundboardu ovlivňují výsledný tón. Nejde o real-time syntézu — jeden tón (7 s) trvá výpočtem ~1277 sekund.

---

## Principy paperu

### 1. Modulární fyzikální řetězec (pipeline)

Paper definuje čtyři oddělené výpočetní moduly:

| Modul | Metoda | Vstup → Výstup |
|-------|--------|----------------|
| Modal basis soundboardu | Semi-analytický model ortotropní desky [Trévisan 2017] | Geometrie + materiál → modální báze Φ_j(x,y), frekvence, tlumení |
| Dynamika strun | Nelineární FEM (Montjoie), Timoshenkův model + geometricky exaktní model | Hammer velocity → síla na bridge |
| Dynamika soundboardu | Modální superpozice, viz Eq. (1)–(2) | Síla na bridge → pohyb desky |
| Akustické záření | Rayleighův integrál (baffled plate) | Pohyb desky → tlak p(M,t) na libovolném bodu |

### 2. Soundboardová modální mechanika

Pohyb desky v j-tém módu:

```
Φ_j(x,y) = Σ_{n,m} A^j_{n,m} · sin(nπx/Lx) · sin(mπy/Ly)
```

Modální souřadnice:

```
u_j(x,y,t) = Σ_j q^j_i(t) · Φ_j(x,y)
```

Frekvence módů závisí na tuhosti (∝ h³ pro ohybovou tuhost) a hmotnosti (∝ h) desky. Žebra (ribs) zvyšují tuhost anizotropně.

### 3. Vliv strukturálních modifikací na tón

Papír testuje čtyři konfigurace soundboardu (Steinway D jako reference):
- **RP**: referenční piano (9 mm tloušťka desky)
- **MP1**: dvojnásobná tloušťka desky (18 mm) → vyšší tuhost → pomalejší útlum
- **MP2**: dvojnásobný rozestup žeber → nižší tuhost → rychlejší útlum, per-partial variabilita
- **MP3**: bez žeber → nejnižší tuhost, nejvyšší mobilita → nejrychlejší útlum, ale nejvyšší počáteční tlak

Klíčové pozorování: **mobilita soundboardu na frekvenci parciály přidává tlumení struně**. Pokud je tlumení soundboardem dominantní, útlumový profil napříč parciály kopíruje tvar mobility křivky → různé parciály mají radikálně odlišné doby doznívání.

### 4. Energy Decay Curve (EDC) jako analytický nástroj

Analýza energy decay používá Schroederův integrál:

```
EDC_dB(t) = 10 · log10( ∫_t^∞ p²(τ) dτ )
```

Toto umožňuje kvantifikovat globální i per-parciální útlum.

### 5. Strunová dynamika: Timoshenko + nelinearity

Struna modelována Timoshenkovým modelem (zahrnuje torzní tuhost → inharmonicita) plus geometricky exaktní nelinearity (lokální deformace při silném úderu kladívka). Coupling struna–soundboard přes kontinuitu vertikální rychlosti na bridge.

### 6. Omezení paperu (přiznána autory)

- Nepřesné modelování disipativních jevů v desce a strunách
- Zjednodušený coupling struna–soundboard (pouze vertikální, bez kývání bridge)
- Výpočetní čas neumožňuje real-time použití (~21× real-time pro 7s tón)

---

## Srovnání se současnou implementací

| Princip z paperu | Stav v IthacaCore | Poznámka |
|-----------------|-------------------|----------|
| **Inharmonicita struny** (Timoshenko, B·k² term) | **Implementováno** | `f_k = k·f0·√(1+B·k²)` v `resonator_voice.cpp`; B extrahováno z reálných vzorků v `params.json` |
| **Bi-exponenciální útlum** (per-partial, τ1/τ2) | **Implementováno** | `env1_[k]*d1_[k] + env2_[k]*d2_[k]` v processBlock; parametry z analýzy |
| **Beating strun** (inter-string detuning) | **Implementováno** | `beat_hz * STRING_SIGNS[s]` — ±beat/2 pro 2 struny, ±beat/2 + 0 pro 3 struny |
| **Modální soundboard** (Φ_j, q_j modální superpozice) | **Zcela chybí** | Syntetizér obchází fyziku desky; místo toho používá LTASE spektrální EQ odvozené z nahrávek |
| **Frekvenčně závislý útlum** dle mobility soundboardu | **Nepřítomno** | Τ1/τ2 per parciál jsou fitované k nahrávce, implicitně zahrnují vliv desky, ale bez fyzikálního modelu mobility |
| **Per-partial variabilita útlumu** jako funkce mobility | **Částečně** | τ1/τ2 jsou per-partial parametry z dat; fyzikální příčina (mobilita na f_k) není modelována |
| **Akustické záření** (Rayleighův integrál) | **Zcela chybí** | Stereo výstup je M/S + all-pass decorrelation, nikoli prostorové záření |
| **Coupling struna–soundboard** | **Zcela chybí** | Syntetizér modeluje strunu a desku jako oddělené nezávislé entity |
| **Hammer-string interakce** (nelineární FEM) | **Zcela chybí** | Útok je modelován jako lineární onset ramp (3 ms); žádná kladívková fyzika |
| **Geometrie soundboardu** (GUI, JSON vstup) | **Netýká se** | IthacaCore je sample-based, ne CAD nástroj |
| **Energy Decay Curve analýza** | **Netýká se** | Není součástí syntézního enginu; prováděno v `analysis/` Python skriptech |
| **Spectral EQ korekce** (LTASE z nahrávek) | **Implementováno** | 8-pásmový biquad EQ cascade v `biquad_eq.cpp`; 64-bodová křivka z params.json |
| **Vícestrunnové unisono** (n_strings = 1/2/3) | **Implementováno** | Per-note parametr `n_strings`, panning spread per string |
| **Stereo prostorový obraz** | **Aproximováno** | Schroederův all-pass + M/S width; není pravé akustické záření |
| **Velocity-dependent amplituda** | **Implementováno** | `vel_gain = (vel/127)^vel_gamma`; `vel_gamma=0.7` |
| **Šum při ataku** (percussive noise) | **Implementováno** | LP-filtrovaný šum s `attack_tau_s`, `floor_rms`, `centroid_hz`; zcela chybí v MAESSTRO paperu |

---

## Doporučení

### 1. Frekvencně závislý útlum přes mobilitu (kritické pro autenticitu)

MAESSTRO ukazuje, že **per-partial doba útlumu není volný parametr** — je fyzikálně determinována mobilitou soundboardu na frekvenci dané parciály. Současná implementace slepě fituje τ1/τ2 k nahrávce, ale nemá model, který by predikoval tyto hodnoty z fyziky desky.

**Doporučení:** Přidat do analýzy (`extract_params.py`) výpočet efektivní mobility soundboardu z měřeného útlumového profilu — tj. invertovat relationship `Δτ_k ≈ f(mobility(f_k))`. To by umožnilo smysluplnou **interpolaci mezi nástroji v latentním prostoru**: změna stiffness desky by odpovídala koherentní změně celého τ-profilu přes parciály.

### 2. Nelinearity kladívka při silném úderu

MAESSTRO používá geometricky exaktní nelinearity struny pro velké výchylky. V IthacaCore je útok čistě lineární (onset ramp). Pro forte/fortissimo velocity layers to způsobuje absenci charakteristické "brightness injection" při tvrdém úderu.

**Doporučení:** Zvážit velocity-dependent spectral tilt při ataku — jednoduše `harmonic_brightness` parametr skalovaný podle velocity. Není nutná plná FEM simulace.

### 3. Coupling deska–struna a jeho vliv na beating

Papír nezmiňuje beating explicitně (je to inter-string fenomén, ne soundboard), ale ukazuje, že coupling stringů na bridge je složitější než prostá kontinuita vertikální rychlosti. V IthacaCore je `beat_hz` fitované k nahrávce. Fyzikálně ale beating závisí na ladění kopírování jednotlivých strun v chóru — hodnoty nejsou stabilní a závisí na teplotě, stáří piana atd.

**Doporučení:** Zachovat empirický fitting; nepokoušet se fyzikálně modelovat ladění unisono strun bez přístupu k mechanickým parametrům konkrétního nástroje.

### 4. EDC jako diagnostický nástroj pro syntetizér

MAESSTRO používá EDC (Schroeder integrál) ke srovnání syntetizovaných tónů. Tento nástroj je přímočarý k implementaci v `analysis/` a dal by objektivní srovnání syntetizovaného a reálného tónu per-note, per-partial.

**Doporučení:** Implementovat EDC výpočet v Python analýze pro validaci, že syntetizovaný útlumový profil odpovídá nahrávce. Jednoduchý `np.cumsum(p**2)[::-1]` v log-scale.

### 5. Soundboardová fyzika v latentním prostoru (Fáze 3)

MAESSTRO demonstruje, že jedna skalární změna (tloušťka desky, počet žeber) způsobuje **konzistentní a fyzikálně smysluplnou** změnu celého tónu. Toto je přesně to, co latentní prostor multi-instrument syntezátoru potřebuje: globální fyzikální parametr, který koherentně ovlivňuje celou sadu per-partial τ hodnot.

**Doporučení pro Fázi 3:** Modelovat latentní prostor nikoli jako interpolaci surových param vektorů, ale jako interpolaci fyzikálních parametrů soundboardu (stiffness proxy, mobility shape). Tím by interpolace Steinway ↔ Bösendorfer získala fyzikální základ místo pouhé numerické interpolace dat.

---

## Shrnutí

MAESSTRO a IthacaCore jsou na opačných koncích spektra přístupu k syntéze piana:

**MAESSTRO:** plný fyzikální forward model — geometrie desky → modální báze → strunová dynamika (FEM) → záření → zvuk. Výpočetně nákladné (1277× real-time), ale fyzikálně plně auditovatelné. Cílová skupina: konstruktéři nástrojů.

**IthacaCore:** inverzní přístup — reálné nahrávky → analytická extrakce parametrů → additivní syntéza s physics-informed modelem (inharmonicita, bi-exp útlum, beating). Real-time schopné, ale soundboardová fyzika chybí explicitně a je implicitně zakódována v per-partial τ parametrech.

Největší gap identifikovaný MAESSTRO, který je v IthacaCore otevřený: **fyzikálně podmíněný per-partial útlumový profil jako funkce mobility soundboardu**. Toto je zároveň nejpřirozenější místo pro budoucí propojení obou přístupů — mobility-informed τ fitting by zpřesnil extrakci parametrů a dal fyzikální základ latentnímu prostoru.

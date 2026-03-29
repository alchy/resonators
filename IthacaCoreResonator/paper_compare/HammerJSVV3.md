# Energy based simulation of a Timoshenko beam in non-forced rotation. Application to the flexible piano hammer shank.
**Chabassier & Duruflé, Inria Bordeaux / CNRS, Journal of Sound and Vibration 333(26), 2014. HAL: hal-00918635v2**

---

## Hlavní oblasti

1. Fyzikální model flexibilního dříku kladívka jako vibrujícího Timoshenko paprsku v neznámé rotaci
2. Energeticky konzervativní numerické schéma (implicit Newmark + nestandardní ošetření kinetické energie)
3. Vliv pianistického doteku (legato vs. staccato) na spektrum výsledného zvuku
4. Vliv vzdálenosti odskoku (let-off distance) na interakční sílu kladívko–struna
5. Longitudinální prekurzor způsobený horizontálním pohybem hlavy kladívka

---

## Principy paperu

### 1. Model kladívka jako Timoshenko paprsku
Dřík kladívka je modelován jako ohýbaný paprsek (Timoshenko beam) v makroskopické rotaci s úhlem `θ(t)`. Mikroskopické vibrace jsou malé odchylky `w(s,t)` kolem makroskopické polohy. Parametry modelu:
- délka dříku L = 0.086 m, Youngův modul E = 10.18 GPa, průřez A = 32.38×10⁻⁶ m²
- hustota ρ = 560 kg/m³, moment setrvačnosti I = 83.44×10⁻¹² m⁴
- smykový modul G = 0.64 GPa, smykový korekční faktor κ = 0.85
- hmotnost hlavy kladívka m_H = 10.76 g (D1) / 7.90 g (C5)

Hlava kladívka je tuhé těleso připojené na volný konec paprsku přes Lagrangeovy multiplikátory (podmínka kontinuity polohy).

### 2. Mechanismus jack → dřík → hlava → struna
Systém zahrnuje celý řetězec:
- Jack force `F_jack(t)` působí na dřík v bodě `s = s_jack` kolmo na dřík
- Let-off mechanismus: jakmile vzdálenost hlava–struna klesne pod `d_let-off`, `F_jack` se okamžitě vynuluje
- Kontakt kladívka se strunou: nelineární Hertzianova síla s hysterezí `F_string = K·δ^p + R·(dδ/dt)·δ^(p-1)` (crushing felt law)

### 3. Energeticky zachovávající numerické schéma
Prostorová diskretizace: Galerkin FEM s polynomy stupně `r` na intervalech `[s_k, s_{k+1}]`.

Časová diskretizace:
- Většina členů: implicitní Newmark se θ = 1/4 (bezpodmínečně stabilní)
- Kinetická energie s nelineárními příspěvky `A·ẇ²·θ̇²`: nestandardní schéma odvozené z variační integrace — zachovává diskrétní energii `E^{n+1/2} = E_kin + E_pot`
- Relativní chyba energie v double precision: ~10⁻⁸ po let-off

### 4. Vliv pianistického doteku na zvuk (klíčový výsledek)
Pro notu D1 (jedna struna, f_L ≈ 550 Hz) byly porovnány:
- **Staccato** (S): krátký silný impuls F_jack = 70 N po 7 ms
- **Legato** (L): pomalý nárust F_jack = 30 N po 100 ms

Výsledky při stejné výsledné rychlosti hlavy kladívka (~3.4 m/s):
- Interakční síla kladívko–struna se liší zejm. kolem 600 Hz: staccato −10 dB, legato −25 dB oproti modelu bez dříku
- Ve zvukovém spektru jsou partials 14–19 na různých úrovních: např. partial 16 je −40 dB (legato) vs. −20 dB (staccato) → auditivně zjistitelný rozdíl barvy
- Longitudinální harmonické (2·f_L ≈ 1100 Hz): −26.3 dB (legato) vs. −18.8 dB (staccato) → staccato zní jasněji

**Klíčový závěr:** Rozdíly ve zvuku pocházejí výhradně z vibrací dříku kladívka — žádný šok struktury, žádná změna dopadového bodu, žádný longitudinální tření.

### 5. Vzdálenost let-off
Let-off 0.0 mm / 1.5 mm / 3.0 mm (legato dotyk): rozdíly v interakční síle až 20 dB v pásmu 500–1000 Hz — přímý vliv na zabarvení.

### 6. Horizontální pohyb hlavy (longitudinální prekurzor)
Horizontální rychlost hlavy kladívka při kontaktu dosahuje ±1.5 m/s — zdaleka ne zanedbatelná. Horizontální třecí pohyb by generoval longitudinální vlnu ve struně nezávislou na nelineárním příčně–podélném vazbě. Navržena jako budoucí rozšíření modelu.

---

## Srovnání se současnou implementací

| Princip paperu | Stav v synth | Poznámka |
|---|---|---|
| **Dřík kladívka jako Timoshenko paprsek** | Neimplementováno | Syntezátor je additivní: kladívko neexistuje jako fyzikální objekt. Výsledný zvuk vychází z fitted parametrů extrahovaných ze sample banky. |
| **Nelineární kontakt kladívka se strunou** (Hertzianova síla s hysterezí, `F = K·δ^p + R·dδ/dt·δ^{p-1}`) | Neimplementováno | Kontaktní mechanismus zcela chybí. Útlum je modelován bi-exponenciální obálkou (`env_k = a1·exp(-t/τ1) + (1-a1)·exp(-t/τ2)`), která nezachycuje dynamiku kontaktu. |
| **Jack force a let-off mechanismus** | Neimplementováno | Syntezátor přijímá pouze MIDI velocity. Jack force jako časová funkce, let-off vzdálenost a její vliv na spektrum nejsou modelovány. |
| **Vliv pianistického doteku na spektrum** (legato ≠ staccato pro stejnou hlasitost) | Částečně | MIDI velocity je mapováno na amplitudu přes `vel_gain = (vel/127)^vel_gamma`. Neexistuje model odlišného tvarování kontaktní síly při různých technikách hry — pouze amplitudový rozdíl, ne spektrální. |
| **Bi-exponenciální útlum** (fyzikálně: závisí na materiálu struny, napnutí, tlumení) | Implementováno | `d1 = exp(-1/τ1·sr)`, `d2 = exp(-1/τ2·sr)`, `a1` jako mixing weight. Parametry jsou fitted z dat, nikoli odvozeny z fyzikální rovnice struny. |
| **Inharmonicita parciálů** (`f_k = k·f0·sqrt(1 + B·k²)`) | Implementováno | `f_hz` pro každý partial je uloženo v `PartialParams` (pre-computed při extrakci). Koeficient B je v `NoteParams`. Přesná shoda s fyzikální rovnicí. |
| **Beating strun** (unisono ladění, detuning mezi strunami) | Implementováno | `beat_hz` per partial, STRING_SIGNS = {±0.5, ±0.5, 0} pro 2/3 struny. Beat scale konfigurovatelný přes `SynthConfig::beat_scale`. |
| **Longitudinální vlny** (phantom partials, nelineární vazba příčné–podélné) | Neimplementováno | Žádné phantom partials ani longitudinální harmonické v syntezátoru. Paper ukazuje, že jsou auditivně důležité (2·f_L = 1100 Hz u D1). |
| **Horizontální pohyb hlavy / třecí impuls** (longitudinální prekurzor) | Neimplementováno | Tato rozšíření nebyla ani v paperu plně implementována — slouží jako výhled. |
| **Energeticky konzervativní numerické schéma** | Irelevantní | Syntezátor nepoužívá explicitní integraci pohybových rovnic — vychází z analyticky extrahovaných dat. Energetická konzervace je vlastností simulačního nástroje Chabassier, nikoli nutností RT syntezátoru. |
| **Vliv let-off vzdálenosti na spektrum** | Neimplementováno | Žádný parametr let-off. Spektrální profil závisí výhradně na EQ z params.json (extrahované ze sample banky, tedy implicitně obsahují průměrné podmínky nahrávání). |
| **Pozice dopadu kladívka na strunu** (zonace, needling) | Neimplementováno | Není modelováno. V budoucnu relevantní pro multi-velocity fyzikální model. |
| **Obálka útoku** (nástup zvuku jako důsledek kontaktní dynamiky) | Aproximováno | `onset_ms = 3 ms` lineární ramp pro potlačení clicku. Neodpovídá fyzikálnímu nárůstu interakční síly při kontaktu kladívka se strunou (Hertzianova dynamika, trvání kontaktu ~1–4 ms). |
| **Noise model** (přechodové šumy) | Implementováno | Exponenciálně se rozpadající bílý šum filtrovaný 1-pole LP. Velmi hrubá aproximace mechanického šumu akce. |
| **Stereo model** (vícestrunnové noty) | Implementováno | Pan per string, Schroeder all-pass decorrelation, M/S width. Není fyzikálně odvozeno z geometrie nástroje, ale empiricky kalibrováno. |

---

## Doporučení

### Kritická chybějící fyzika (přímý dopad na autenticitu):

**1. Velocity-dependent spectral shaping (touch model)**
Paper ukazuje, že legato vs. staccato pro stejnou hlasitost generuje rozdíly 10–25 dB v pásmu 600–1100 Hz. Současný syntezátor má velocity pouze jako amplitudový scaler. Minimální vylepšení bez plné fyzikální simulace: fit **velocity-dependent EQ curvy** z multi-velocity sample banky — tj. pro každou MIDI notu spočítat, jak se mění spektrální profil s velocity layer, a interpolovat EQ (nikoli jen amplitudu) podle velocity. Parametry pro to jsou dostupné v sample bance (8 velocity vrstev × 88 not).

**2. Longitudinální harmonické (phantom partials)**
Papery Chabassier (viz i [8] citovaný v textu) zdůrazňují, že longitudinální vlny na frekvencích `n·f_L` přispívají k charakteristickému "zvonivému" barvení basových not. Pro D1 je `2·f_L ≈ 1100 Hz` a paper ukazuje auditivně zjistitelný rozdíl v jeho úrovni. Tyto partials v současném additivním syntezátoru chybí — bylo by možné je přidat jako speciální skupinu parciálů s odlišným útlumovým profilem při extrakci z dat.

**3. Kontaktní útlum jako funkce velocity**
Bi-exponenciální model `τ1, τ2` je fitted jako konstanta pro danou velocity layer. Paper ukazuje, že kontakt trvá jinak dlouho u staccato vs. legato — τ útoku (`a1` mixing weight) by měl záviset na velocity. Stávající interpolace mezi velocity layers toto zachycuje jen přibližně.

**4. Let-off a sustain pedal**
Sustain pedal je implementován v `voice_manager.cpp` jako zpožděné note-off — fyzikálně správně. Ale paper poukazuje, že let-off vzdálenost ovlivňuje spektrum i při plném stisku klávesy (hammmer escapement). Toto je obtížné modelovat bez plné fyzikální simulace — spíše doporučuji ignorovat pro RT syntezátor.

### Střednědobé doporučení:
- Přidat extrakci **longitudinálních harmonických** z FFT (identifikace `f_L` pro bass noty, fit jejich `A0`, `τ`) do `analysis/extract_params.py` a uložit jako speciální partials s příznakem `longitudinal=True`.
- Implementovat **velocity-spectral interpolaci**: místo interpolace pouze `A0` interpolovat i `eq_gains_db` mezi velocity layers.

### Co implementovat nevyplatí:
- Plná Timoshenko beam simulace pro RT syntézu: computationally prohibitive, parametry kalibrace vyžadují rozměrná experimentální měření
- Energeticky konzervativní schéma: relevantní pouze pro offline FEM simulaci, ne pro additivní syntézu
- Horizontální třecí impuls: zatím ani plně validován experimentálně

---

## Shrnutí

Paper Chabassier & Duruflé (2014) poskytuje první kompletní fyzikální simulaci flexibilního dříku kladívka (Timoshenko paprsek) spojené s celým modelem piana. Klíčový výsledek: **pianistický dotek (legato vs. staccato) pro stejnou výslednou hlasitost způsobuje rozdíly 10–25 dB ve spektru kolem 600–1100 Hz**, přičemž tento efekt pochází výhradně z vibrací dříku — nikoli ze šoku struktury nebo jiných mechanismů. Let-off vzdálenost způsobuje podobně velké spektrální rozdíly.

Současná implementace (`resonator_voice.cpp`) je **additivní syntezátor s empiricky extrahovanými parametry** — fyzikálně informovaný, ale bez dynamického modelu kladívka. Správně implementuje inharmonicitu, bi-exponenciální útlum, beating strun a spektrální EQ. Chybí: velocity-spectral shaping (touch model), longitudinální harmonické a fyzikální model kontaktu. Největší okamžitý přínos by mělo přidání **velocity-závislé spektrální interpolace** (EQ per velocity layer) a extrakce longitudinálních harmonických pro bas — obojí je proveditelné v rámci stávající extrakční pipeline bez změny RT architektury.

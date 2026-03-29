# Piano Sounds Synthesis with an emphasis on the modeling of the hammer and the piano wire
**Teng Wei Jian, MSc Acoustic and Music Technology, University of Edinburgh, 2012**

---

## Hlavni oblasti

Paper pokryva tyto oblasti:

1. **Fyzika piana** — mechanismus kladivka, struna jako tuha struna (stiff string), deska a most
2. **Prehled syntezni metod** — FM, Karplus-Strong, vzorkovani, spektralni modely, fyzikalni modely
3. **Model kladivka** — finite-difference implementace nelinearni interakce kladivka a struny
4. **Model struny** — digital waveguide (DWG) s filtry: loss, dispersion (all-pass kaskada), tuning (frakcni delay)
5. **Dalsie aspekty** — vice strun (beating), deska a most (konvoluce s impulsni odezvou tela piana)

---

## Principy paperu

### Kladivko (kapitola 4)
Paper implementuje fyzikalni model kladivka podle Chaigne & Askenfelt (1994). Kladivko je modelovano jako nelinearni pruzina (nonlinear spring) s mocninnym zakonem sily:

```
F_H = K * (x_H - x_S)^p   pro x_H > x_S
F_H = 0                    jinak
```

- `K` = tuhost filce, `p` = exponent tuhost (typicky 2.3–3.5)
- Pohyb kladivka i struny se pocita soucasne metodou konecnych diferencí
- Simulace trva pouze po dobu kontaktu (< 5 ms), pak se vypocet ukoncí
- Vystup: casovy prubeh sily F(t), ktera vstupuje do DWG modelu struny
- Problem: kazda nota vyzaduje presne nameřene hodnoty parametru (hmotnost struny/kladivka, delka, napeti). Bez nich model pracuje jen s typickymi hodnotami z literatury (C2/C4/C7 dle Chaigne & Askenfelt 1994)

### Struna — Digital Waveguide (kapitola 5)
Paper implementuje DWG model pruty s temito komponentami:

**5.1 Travelling wave a DWG zaklad:**
- Reseni vlnove rovnice jako dve protibehy vlny (d'Alembertovo reseni)
- DWG = dve delay lines delky N/2 s reflexi na koncich (gain -1 pro idealni strunu)
- Modely rychlosti struny (ne posuvu), pro snazsi napojeni na kladivkovy vystup
- Delka delay line: `N = round(fs / f0)`

**5.2 Loss filter (5.3):**
- Frekvencne-zavisly utlum: ruzne doby dolehani (decay times) pro ruzne parcialy
- Implementace: jednopovolovy dolnopropustny filtr 1. radu (Valimaki et al. 1996):

```
H_loss(z) = g * (1 + a) / (1 + a*z^-1)
```

- DC gain `g` nastaven dle doby dolehani prvniho partials; pol `a` (zaporne, male) ridi frekvencni zavislost utlumu
- Cela tato struktura ridi decay time per-partial, ale bez namerených dat se `a` nastavi heuristicky (-0.001 az -0.01)

**5.3 Dispersion filter (5.4):**
- Inharmonicita struny vzniká z tuhosti: f_k = k * f0 * sqrt(1 + B*k^2)
- Implementace: kaskada prvorenych all-pass filtru (Van Duyne & Smith 1994):

```
H_ap(z) = (c + z^-1) / (1 + c*z^-1)
```

- Pocet filtru (typicky 16 pro C4) se meni dle frekvence noty; pro vysoke tony je pocet snizen protoze delay line je kratka
- Koeficient `c` nastaven heuristicky; delka delay line se prepocita tak, aby fundamentalni frekvence nesla

**5.4 Tuning filter (5.5):**
- Kompenzace frakcniho delaye: delay line musí mit celociselnou delku, ale skutecna delka (fs/f0) je realna
- Pouziva stejny all-pass filtr 1. radu s koeficientem odvozeny z frakcni casti delky:

```
c_tuning = (1 - D_frac) / (1 + D_frac)
```

**5.5 Vice strun a beating (5.6):**
- Skutecne piano ma 2–3 struny na notu; papir je modeluje jako N paralelních DWG modelu
- Kazdy DWG model je ladeny mirne jinak (mala variace tuning filtru) → vzniká beating
- Paper sam oznacuje tento pristup jako "ad hoc"; spravnejsi by byl model se zpetnou vazbou (coupling) mezi vlnovodem

**5.6 Deska a most (5.6):**
- Deska neni primo modelovana
- Approksimace: vystup DWG je konvolvovan s nahrávkou ucheru do tela piana (impulse response)
- Vysledek: teplejsi zvuk, mene "kovovy"

---

## Srovnani se soucasnou implementaci

| Princip paperu | Stav v soucasne implementaci | Poznamka |
|---|---|---|
| **Inharmonicita: f_k = k·f0·sqrt(1+B·k^2)** | Plne implementovano | `resonator_voice.h` l.9: vzorec identicky; B i f0 extraktovany z realnych samplú per-nota per-vel |
| **Decay — frekvencne-zavisly utlum (DWG loss filter)** | Nahrazeno bi-exponencialním obalkou | DWG loss filter neni; misto nej `tau1/tau2/a1` per-partial z params.json. Fyzikalne ekvivalentni pro syntézu, ale diskretizace jina |
| **Bi-exponencialni decay (dva stupne dolehani)** | Plne implementovano | `d1_[k]`, `d2_[k]`, `a1_[k]`, `a2_[k]` — explicite viz `resonator_voice.cpp` l.154-167 |
| **Beating vice strun (multi-string detune)** | Plne implementovano | `beat_hz/beat_depth` per-partial; STRING_SIGNS[n_str] ±beat/2; n_strings=1/2/3 dle MIDI |
| **Dispersion filter — all-pass kaskada pro inharmonicitu** | Neni (disperzni aspekt ji resi jinak) | Papir pocita inharmonicitu pres all-pass filtry v DWG; soucasna implementace ji modeluje primo jako zazijistene f_k ze samplebanky (pres extrakci peak detection). Fyzikalni efekt je zachycen, pristup odlisny |
| **Tuning filter (frakcni delay)** | Neni potreba | DWG delay-line tuning je specificky problem DWG architektury; additivni syntezer ladí primo pres omega_[k][s] v radianech — frakcni delay je automaticky presny |
| **Kladivko: nelinearni FD model** | Neni implementovano | Soucasna implementace nema model kladivka — vyuziva extrahovane A0 amplitude a noise transient z realnich samplú |
| **Vliv velocity na barvu zvuku (kladivko)** | Castecne implementovano | Pouze amplitude scaling pres vel_gamma (`(vel/127)^vel_gamma`); spektralni zmena timbre pri ruzne sile uderu neni modelovana |
| **Soundboard (deska) — konvoluce s IR** | Neni implementovano | Papir pouziva konvoluci s nahrávkou ucheru; soucasna implementace pouziva spektralni EQ (biquad_eq.cpp) extrahovany z LTASE krivky samplú |
| **Spektralni EQ (barva zvuku)** | Implementovano; pokrocileji nez papir | BiquadEQ: 8-pasmovy peaking EQ z 64-bodove log-spaced krivky; per-nota per-vel z params.json. Papir toto neresi systematicky |
| **Noise (attack transient)** | Implementovano; originalne | `NoiseParams`: 1-pole LP filtered noise s attack_tau_s; L/R nezavisle. Papir attack resi konvolucí s IR; soucasne reseni je lehci a parametrizovatelne |
| **Stereo model** | Implementovano; papir nema | MIDI-dependent pan center, per-string equal-power panning, Schroeder all-pass decorrelace, M/S sirka. Papir je mono |
| **DWG delay-line architektura** | Neni | Soucasna implementace je cisty additivni synthesizer (N parcialu × M strun), ne DWG. Vyhodou je plna kontrola nad parametry extraktovanymi z realnich dat |
| **Coupling strun pres most** | Neni | Papir navrhuje zpetnovazebni coupling; soucasna implementace meri coupling implicitne — beating je extrakovany z realnich samplú, ale fizikalni coupling dynamicky (pri decay) neni modelovan |
| **Parametry z namerených dat vs. extrakce ze samplú** | Zasadne odlisny pristup | Papir spoléha na tabulkové hodnoty (Chaigne & Askenfelt 1994, 3 noty). Soucasna implementace extrakci parametrú z celé 88×8 samplebanky automaticky — kazdá nota má vlastní f0, B, parcialy, tau1/tau2, beat_hz |

---

## Doporuceni

### 1. Dynamicka zmena spektra pri ruznych velocitech (kriticky chybejici prvek)
Papir popisuje kladivkuv model, kde vyssi pocatecni rychlost = ostrejsi vrcholy siloveho pulzu → vice vysokokmitoctnich sl¸zek. Soucasna implementace toto nemoduluje — spectral EQ i parcialy jsou per-velocity extrahovany, ale velocity interopolace (viz `interpolateNoteLayers`) je linearna. **Doporuceni**: proverit, zda `harmonic_brightness` konfigurační parametr je dostatecny, nebo pridat per-velocity spektralni morph.

### 2. Fyzikalni coupling decay pri sdileni tonu (dvojice/trojice strun)
Papir zminuje, ze dve-stupnovy decay vznika z vertikalni/horizontalni polarizace vibraci strojenych strun. Soucasná implementace ma `tau1/tau2` na urovni partials, ale coupling prenos mezi strunami (kdy jedna dekayuje rychleji kvuli brizde) neni dynamicky. **Doporuceni**: zkoumat, zda extrahovane `a1` (mixing weight bi-exp) dostatecne zachycuje tento efekt, nebo zda je treba pridat coupling gain term.

### 3. Nonlinearity pri silnem uderu (forte)
Model kladivka v papiru ma nelinearni mocninnou silu — pri forte uderu je stiffness exponent `p` vyssi, coz produkuje kvalitativne odlisny spurialni obsah. Soucasna implementace ma pouze amplitude scaling (vel_gamma). **Doporuceni**: implementovat jednoduchy nonlinear waveshaping pro vel > 110, nebo extrahovanu per-velocity spektralni krivku dat jako mapa misto linearni interpolace.

### 4. Deska/most — IR pristup vs. EQ
Papir konvolvuje s IR tela piana. Soucasna spectral EQ je lepsi z hlediska parametricke kontroly, ale postrazi casove charakteristiky IR (attack transient). Noise parametry cástecne kompenzuji tuto medzeru (attack_tau_s). Doporucení jsou situacne zavisle — IR by zlepšila autenticitu, ale zvysila latenci a pameTové naroky.

### 5. Presnejsi model beatingh — faze strun
Papir nema coupling, ale navrrhuje zpetnovazebni model. Soucasna implementace ma pevne nastavene `beat_hz` z extrakce. **Potencional**: beat_depth a beat_hz jsou extrakce z LTASE, takze jsou statisticke prumery — pri skutecnem piane se beating meni s casem (protoze tlumení jednotlivych strun neni identicke). Pridani stochastickeho modulace beat_hz v case by zlepšilo zvuk.

---

## Shrnutí

Papir Teng (2012) je pedagogicky vystaveny MSc projekt, ktery prezentuje **digital waveguide (DWG) model** klavirni struny s filtry pro utlum, disperzi a ladeni, a pridruzeny **finite-difference model kladivka**. Jde o klasicky fyzikalni pristup: modelovat mechanismus tvorby zvuku primo, s fyzikalni motivaci kazde komponenty.

**Zasadni rozdil od soucasne implementace**: papir se snazi simulovat vlnovou dynamiku struny "od zacatku" (PDE→DWG), zatimco soucasny syntezator **extrakuje fyzikalni parametry z realnich samplú** a pouziva je v additivnim synthesizeru. Oba pristupy zachycuji stejne fyzikalni jevy (inharmonicita, bi-exp decay, beating), ale soucasny pristup:

- Je vyrazne **presnejsi pro dany nastroj** (parametry jsou namerane, ne odhadnute)
- Ma **plnou pokrytost 88×8 not** (vs. papiru 3 referenci body)
- Neobsahuje chyby DWG (frakcni delay, stabilitni podminky disperzniho filtru)
- Ztrací **fizikalni predikcni schopnost** — nema ze ktereho prvku odvodit chování pri zmene fyzikalnich podminek (teplota, string aging)

Hlavni lekce z papiru pro tento projekt: **velocity-dependent timbre** (papir explicitne resi pres kladivko, soucasna implementace to dela jen amplitude scalingem) a **dynamicky string coupling** jsou oblasti, kde papir nabizi lepsi fyzikalni motivaci pro budouci rozsireni.

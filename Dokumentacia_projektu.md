# Dokumentácia projektu: Kryptoanalýza pomocou veľkých jazykových modelov

## Abstrakt
Tento projekt sa zameriava na výskum a implementáciu kryptoanalytických metód využívajúcich veľké jazykové modely (LLM - Large Language Models) na dešifrovanie klasických šifier. Projekt predstavuje inovatívny prístup k riešeniu kryptografických problémov pomocou moderných metód strojového učenia.

## Štruktúra projektu

### Hlavné komponenty
1. **Serverová aplikácia** (`server.py`)
   - Implementácia REST API pomocou Flask
   - Autentifikácia používateľov (lokálna + Google OAuth)
   - Správa databázy PostgreSQL
   - Integrácia s jazykovými modelmi

2. **Zdrojový kód** (`src/`)
   - `cipher.py`: Implementácia šifrovacích algoritmov
   - `data_gen.py`: Generovanie trénovacích dát
   - `decryptor.py`: Modul pre dešifrovanie
   - `train_model.py`: Trénovanie modelov

3. **Databázová schéma**
   - Tabuľka Users: Správa používateľov
   - Tabuľka Ciphers: Informácie o šifrách
   - Tabuľka Models: Správa modelov
   - Tabuľka Decryption_Attempts: Záznamy o pokusoch
   - Tabuľka Decryption_Results: Výsledky dešifrovania
   - Tabuľka Manual_Corrections: Manuálne korekcie

### Technológie
- **Backend**: Python 3.9+, Flask 3.1.0
- **Databáza**: PostgreSQL
- **ML Framework**: PyTorch 2.6.0, Transformers 4.51.1
- **Autentifikácia**: Flask-Login, Google OAuth
- **Deployment**: Docker, Gunicorn

## Funkcionalita systému

### 1. Kryptoanalytické schopnosti
- Dešifrovanie klasických šifier:
  - Cézarova šifra
  - Monoalfabetická substitučná šifra
  - Vigenèrova šifra
  - Transpozičné šifry

### 2. Používateľské rozhranie
- Webové rozhranie pre nahrávanie šifrovaných textov
- História dešifrovacích pokusov
- Štatistiky úspešnosti
- Správa používateľského profilu

### 3. Analytické nástroje
- Meranie presnosti dešifrovania
- Analýza výkonu modelov
- Vizualizácia výsledkov

## Bezpečnostné opatrenia
- Hašovanie hesiel
- OAuth 2.0 integrácia
- PostgreSQL s SSL pripojením
- Ochrana proti SQL injection
- Rate limiting pre API endpointy

## Metriky hodnotenia
1. **Presnosť dešifrovania**
   - Charakterová presnosť
   - Sémantická podobnosť
   - Čitateľnosť výstupu

2. **Výkonnostné metriky**
   - Čas spracovania
   - Využitie systémových zdrojov
   - Škálovateľnosť

## Budúce vylepšenia
1. Rozšírenie podpory pre ďalšie typy šifier
2. Implementácia pokročilých metód predspracovania
3. Optimalizácia výkonu modelov
4. Rozšírenie analytických nástrojov

## Technická implementácia

### Architektúra systému
- **Microservice-based** architektúra
- RESTful API rozhranie
- Asynchrónne spracovanie požiadaviek
- Škálovateľné databázové riešenie

### Deployment
- Containerizácia pomocou Docker
- CI/CD pipeline
- Monitoring systémových zdrojov
- Automatické zálohovanie dát

## Proces vývoja a história commitov

### Chronologický vývoj projektu

#### Fáza 1: Základná implementácia (Apríl 2025)
- Implementácia základného servera a nastavenie Flask aplikácie
- Vytvorenie používateľského rozhrania
- Integrácia autentifikácie

Kľúčové commity:
- 7e73807 - Aktualizácia nastavení (2025-04-25)
- 41105ee - Aktualizácia server.py (2025-04-25)
- 5fd354e - Aktualizácia transformers (2025-04-25)
- 7b79a03 - Aktualizácia 10 (2025-04-25)
- d686375 - Aktualizácia server2 (2025-04-25)

#### Fáza 2: Vylepšenia a optimalizácia (Koniec Apríla 2025)
- Optimalizácia výkonu servera
- Riešenie problémov s pamäťou (OOM)
- Aktualizácia závislostí a požiadaviek

Kľúčové commity:
- 72b7cc4 - Optimalizácia servera OOM (2025-04-26)
- 497f3e5 - Aktualizácia index stránky (2025-04-26)
- 9dc195c - Aktualizácia 11 (2025-04-26)

#### Fáza 3: Finálne úpravy (Máj 2025)
- Aktualizácia požiadaviek projektu
- Finálne vylepšenia a optimalizácie

Kľúčové commity:
- d1deb46 - Aktualizácia požiadaviek (2025-04-30)
- 83b4147 - Aktualizácia 2 (2025-04-30)
- da39ffe - Aktualizácia 3 (2025-05-01)

### Metodika vývoja
- Iteratívny prístup k vývoju
- Pravidelné aktualizácie a optimalizácie
- Dôraz na výkon a stabilitu systému
- Kontinuálne testovanie a vylepšovanie

## Záver
Projekt predstavuje inovatívne riešenie v oblasti kryptoanalýzy, kombinujúc klasické kryptografické metódy s modernými prístupmi strojového učenia. Systém poskytuje robustnú platformu pre výskum a praktické aplikácie v oblasti bezpečnosti a kryptografie.

---

*Dokumentácia vytvorená: 1. mája 2025*
*Verzia: 1.0* 
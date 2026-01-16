import requests
from bs4 import BeautifulSoup
from deep_translator import GoogleTranslator
import pandas as pd
import time
from urllib.parse import urlparse

# -------------------------
# RAW URL LIST (WITH DUPES)
# -------------------------

RAW_URLS = [
    "https://www.provincie-utrecht.nl/actueel/nieuws/onderzoek-naar-pfas-regenwater-gestart",
    "https://www.provincie-utrecht.nl/onderwerpen/bodem-en-water/bodemverontreiniging/bodemsanering-park-vliegbasis-soesterberg",
    "https://www.rtvutrecht.nl/nieuws/3970001/arnaut-onderzoekt-waarom-pfas-steeds-vaker-in-de-bodem-zit",
    "https://www.rtvutrecht.nl/nieuws/3970376/veel-meer-geld-nodig-voor-opruimen-pfas-bij-vliegbasis-soesterberg",
    "https://www.provincie-utrecht.nl/actueel/nieuws/onderzoek-pesticiden-bron-van-ruim-duizend-kilo-pfas-vervuiling-jaar",
    "https://www.provincie-utrecht.nl/actueel/nieuws/rapportage-grondwaterkwaliteit-toename-van-pfas-grondwater-verwacht",
    "https://www.provincie-utrecht.nl/actueel/nieuws/grondwaterstanden-beinvloeden-saneringsaanpak-park-vliegbasis-soesterberg",
    "https://www.provincie-utrecht.nl/actueel/nieuws/kadernota-2026-doorpakken-op-uitvoering-coalitieakkoord-en-vergroten-realisatiekracht",
    "https://www.rtvutrecht.nl/nieuws/3970001/arnaut-onderzoekt-waarom-pfas-steeds-vaker-in-de-bodem-zit",
    "https://www.rtvutrecht.nl/nieuws/3918065/meten-is-weten-in-nieuw-laboratorium-in-houten-testen-ze-waterkwaliteit",
    "https://www.rtvutrecht.nl/nieuws/3967273/provincie-utrecht-onderzoekt-regenwater-op-aanwezigheid-pfas",
    "https://www.rtvutrecht.nl/nieuws/3915790/te-veel-gif-in-ons-bloed-pfas-laat-ook-utrecht-niet-met-rust",
    "https://www.rtvutrecht.nl/nieuws/3906258/verhoogde-pfas-concentraties-in-afvalwater-amersfoort-bron-onduidelijk",
    "https://www.rtvutrecht.nl/nieuws/3881813/op-eieren-lopen-tijdens-pasen-alles-barst-van-de-pfas-dit-kan-er-ook-nog-wel-bij",
    "https://www.rtvutrecht.nl/nieuws/3881244/rivm-eet-geen-eieren-van-hobbykippen",
    "https://www.rtvutrecht.nl/nieuws/3849040/drinkwatertekort-dreigt-in-utrecht-vitens-zoekt-naar-nieuwe-bronnen",
    "https://www.rtvutrecht.nl/nieuws/3818772/baanbrekend-uu-onderzoek-naar-waterzuivering-door-champignonschimmels",
    "https://www.rtvutrecht.nl/nieuws/3797696/omgevingsdiensten-voldoen-niet-aan-alle-kwaliteitseisen-personeel-niet-altijd-voldoende-opgeleid",
    "https://www.rtvutrecht.nl/nieuws/3760379/zwemmen-dan-hoef-je-je-in-provincie-over-pfas-geen-zorgen-te-maken",
    "https://www.rtvutrecht.nl/nieuws/3731640/124-plekken-in-de-provincie-mogelijk-met-pfas-vervuild",
    "https://www.rtvutrecht.nl/nieuws/3730421/champignon-kan-water-zuiveren-ontdekken-utrechtse-onderzoekers",
    "https://www.rtvutrecht.nl/nieuws/3727817/aanklacht-tegen-nederlandse-staat-voor-vervuiling-en-gezondheidsschade-door-pfas",
    "https://www.rtvutrecht.nl/nieuws/3721231/uiterwaarden-bij-elst-teruggeven-aan-de-natuur-uniek-gebied-met-zeldzame-flora-en-fauna",
    "https://www.rtvutrecht.nl/nieuws/3720450/pfas-in-sloot-utrechtse-wijk-veemarkt",
    "https://www.rtvutrecht.nl/nieuws/3706251/bijensterfte-schokt-imker-uit-groenekan-dit-probleem-wordt-enorm-onderschat",
    "https://www.rtvutrecht.nl/nieuws/3668376/den-haag-betaalt-23-5-miljoen-voor-opruimen-pfas-bij-vliegbasis-soesterberg",
    "https://www.rtvutrecht.nl/nieuws/3646703/pfas-paniek-in-de-provincie-utrecht-dit-is-zo-naar-en-ziekmakend-we-moeten-ermee-stoppen",
    "https://www.rtvutrecht.nl/nieuws/3642705/drie-bedrijven-gebruiken-pfas-in-utrecht-gemeente-ziet-geen-risicos",
    "https://www.rtvutrecht.nl/nieuws/3578975/alarmerend-rapport-over-waterkwaliteit-problemen-zijn-vergelijkbaar-met-stikstofdossier",
    "https://www.rtvutrecht.nl/nieuws/3551428/waterkwaliteit-nog-lang-niet-op-orde-boete-dreigt-we-moeten-echt-aan-de-bak",
    "https://www.rtvutrecht.nl/nieuws/3158880/huizen-steeds-sneller-steeds-duurder-koopprijs-in-utrecht-stijgt-met-15-procent",
    "https://www.rtvutrecht.nl/nieuws/3109709/utrechts-pfas-onderzoek-afgerond-twee-locaties-boven-risiconorm",
    "https://www.rtvutrecht.nl/nieuws/2035679/ondernemers-in-de-knel-starter-na-acht-weken-al-weer-dicht",
    "https://www.rtvutrecht.nl/nieuws/2027290/student-vast-in-italie-nieuwe-besmettingen-in-utrecht-lees-het-in-ons-corona-liveblog",
    "https://www.rtvutrecht.nl/nieuws/2018641/meer-nieuwbouwhuizen-maar-woningnood-blijft-enorm-in-utrecht",
    "https://www.rtvutrecht.nl/nieuws/2005864/hoe-stikstof-en-pfas-ons-dit-jaar-in-zn-greep-hielden",
    "https://www.rtvutrecht.nl/nieuws/2003014/drukke-avondspits-vanwege-boeren-die-terugkeren-van-protestacties",
    "https://www.rtvutrecht.nl/nieuws/2001847/boeren-mogen-van-rechter-protesteren-maar-zien-toch-af-van-acties",
    "https://www.rtvutrecht.nl/nieuws/1993711/tweede-kamer-wil-verbreding-a27-snel-uitvoeren",
    "https://www.rtvutrecht.nl/nieuws/1993492/pfas-norm-verhoogd-groot-deel-bouwprojecten-kan-weer-verder",
    "https://www.rtvutrecht.nl/nieuws/1989549/verbreding-ring-utrecht-loopt-tot-drie-jaar-vertraging-op",
    "https://www.rtvutrecht.nl/nieuws/1985846/reacties-op-verbreding-snelwegen-utrecht-van-belachelijk-tot-van-groot-belang",
    "https://www.rtvutrecht.nl/nieuws/1985244/pfas-en-stikstofcrisis-sijpelt-door-bij-bouwbedrijf-bam-in-bunnik",
    "https://www.rtvutrecht.nl/nieuws/1985216/bezoek-rutte-aan-merwedekanaalzone-in-teken-van-stikstofcrisis-en-boze-bewoners",
    "https://www.rtvutrecht.nl/nieuws/1985189/premier-rutte-bezoekt-merwedekanaalzone-praat-ook-over-polder-rijnenburg",
    "https://www.rtvutrecht.nl/nieuws/1982090/boze-bunschotense-bouwer-kees-op-weg-naar-malieveld-het-is-hopeloos-er-moet-nu-wat-gebeuren",
    "https://www.rtvutrecht.nl/nieuws/1982043/grootste-problemen-voorbij-in-utrecht-na-chaotische-ochtendspits",
    "https://www.rtvutrecht.nl/nieuws/1982033/bouwbedrijven-staken-in-den-haag-tegen-milieuregels"
]

# Remove duplicates
URLS = list(dict.fromkeys(RAW_URLS))

HEADERS = {"User-Agent": "Mozilla/5.0 (Academic use)"}
translator = GoogleTranslator(source="nl", target="en")

def detect_source(url):
    domain = urlparse(url).netloc
    if "provincie-utrecht" in domain:
        return "provincie-utrecht"
    elif "rtvutrecht" in domain:
        return "rtvutrecht"
    return "unknown"

def scrape_article(url):
    print("Scraping:", url)
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    # Title
    h1 = soup.find("h1")
    title = h1.get_text(strip=True) if h1 else (soup.title.get_text(strip=True) if soup.title else "")

    # Date (try <time> or meta first)
    date = ""
    time_tag = soup.find("time")
    if time_tag:
        date = time_tag.get("datetime", time_tag.get_text(strip=True))

    if not date:
        meta = soup.find("meta", attrs={"property": "article:published_time"})
        if meta:
            date = meta.get("content", "")

    # Text
    paragraphs = soup.find_all("p")
    text_nl = "\n".join(p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 40)

    # Translate to English
    text_en = translator.translate(text_nl[:4500]) if text_nl else ""
    title_en = translator.translate(title) if title else ""

    return title_en, date, text_en

rows = []

for url in URLS:
    try:
        source = detect_source(url)
        title, date, text = scrape_article(url)

        rows.append({
            "source": source,
            "date": date,
            "url": url,
            "title_en": title,
            "text_en": text
        })

        time.sleep(1)

    except Exception as e:
        print("❌ Failed:", url, e)

df = pd.DataFrame(rows)
df.to_csv("pfas_news_final.csv", index=False)

print("✅ DONE — saved as pfas_news_final.csv")
print("Articles scraped:", len(df))

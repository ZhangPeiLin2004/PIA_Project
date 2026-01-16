

import json
import requests
from datetime import datetime

# Configuration
SEARCH_KEYWORD = "PFAS"
MAX_RESULTS = 10
OUTPUT_FILE = "pfas_tenders.json"
API_BASE = "https://www.tenderned.nl/papi/tenderned-rs-tns/v2"

def search_tenders_api(keyword, max_results=10):
    
    results = []
    page = 0
    size = 50
    
    print(f"Searching for '{keyword}' tenders via API...")
    
    while len(results) < max_results:
        # API endpoint
        url = f"{API_BASE}/publicaties"
        params = {
            'page': page,
            'size': size,
            'publicatieDatumPreset': 'AF30',  # Lastthrirty days
            'useExperimentalFeature': 'false'
        }
        
        print(f"\n📡 Fetching page {page}...")
        
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if 'content' not in data or not data['content']:
                print("No more tenders found")
                break
            
            tenders = data['content']
            print(f"   Found {len(tenders)} tenders on this page")
            
           
            for tender in tenders:
                if len(results) >= max_results:
                    break
                
                
                tender_text = json.dumps(tender, ensure_ascii=False).upper()
                
                if keyword.upper() in tender_text:
                    print(f"PFAS FOUND in tender {tender.get('publicatieId', 'unknown')}")
                    
                   o
                    tender_data = {
                        'id': tender.get('publicatieId'),
                        'title': tender.get('aanbestedingNaam', ''),
                        'organization': tender.get('opdrachtgeverNaam', ''),
                        'publication_date': tender.get('publicatieDatum', ''),
                        'deadline': tender.get('sluitingsDatum', ''),
                        'url': f"https://www.tenderned.nl/aankondigingen/overzicht/{tender.get('publicatieId', '')}",
                        'description': tender.get('opdrachtBeschrijving', ''),
                        'procedure_type': tender.get('procedure', {}).get('omschrijving', ''),
                        'contract_type': tender.get('typeOpdracht', {}).get('omschrijving', ''),
                        'publication_type': tender.get('typePublicatie', {}).get('omschrijving', ''),
                        'full_data': tender,
                        'scraped_at': datetime.now().isoformat()
                    }
                    
                    
                    pfas_count = tender_text.count(keyword.upper())
                    tender_data['pfas_mentions'] = pfas_count
                    
                    results.append(tender_data)
                    
                    print(f"      📋 {tender_data['title'][:60]}...")
                    print(f"      🏢 {tender_data['organization'][:60]}")
                    print(f"      🔍 PFAS mentioned {pfas_count} times")
            
            # Check if there are more pages
            if data.get('last', True):
                print("Reached last page")
                break
            
            page += 1
            
        except requests.exceptions.RequestException as e:
            print(f"API Error: {e}")
            break
        except Exception as e:
            print(f"Error: {e}")
            break
    
    return results

def main():
    """Main function."""
    
    
    tenders = search_tenders_api(SEARCH_KEYWORD, MAX_RESULTS)
    
    if not tenders:
        print("\n No PFAS tenders found")
        return
    
    
    print(f"\nSaving {len(tenders)} tenders to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(tenders, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*80}")
    print(f"Done Found {len(tenders)} PFAS tenders")
    print(f"Data saved to: {OUTPUT_FILE}")
    print(f"{'='*80}\n")
    
   
    for i, tender in enumerate(tenders, 1):
        print(f"{i}. {tender['title'][:70] if tender['title'] else 'No title'}")
        print(f"{tender['organization'][:70] if tender['organization'] else 'No org'}")
        print(f"Deadline: {tender['deadline']}")
        print(f"PFAS mentions: {tender['pfas_mentions']}")
        print(f"{tender['url']}\n")

if __name__ == "__main__":
    main()

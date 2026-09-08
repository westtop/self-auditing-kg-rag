#!/usr/bin/env python3
"""
Step 1: GPT-4o Triplet Extraction from Wasabi Hydroponic Literature
===================================================================
논문 설계: GPT-4o (temperature=0.0, response_format=JSON)

사용법:
  pip install openai pymupdf
  export OPENAI_API_KEY="sk-..."
  python step1_gpt4o_extraction.py

입력: pdfs/ 폴더 내 PDF 파일들
출력: triplets_gpt4o_extracted.json
"""

import os
import json
import sys
import time

try:
    from openai import OpenAI
except ImportError:
    print("openai 패키지 필요: pip install openai")
    sys.exit(1)

try:
    import fitz  # PyMuPDF
except ImportError:
    print("PyMuPDF 패키지 필요: pip install pymupdf")
    sys.exit(1)

# ============================================================
# Configuration
# ============================================================
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
MODEL = "gpt-4o"
TEMPERATURE = 0.0
MAX_CHARS_PER_CHUNK = 6000  # PDF 텍스트 청크 크기

SYSTEM_PROMPT = """You are an expert agricultural scientist specializing in wasabi (Eutrema japonicum) hydroponic cultivation.

Your task: Extract knowledge triplets from the given text. Each triplet has:
- subject: The parameter or entity (e.g., "water_temp", "DO", "EC", "pH", "AITC", "PPFD", "air_temp", "photoperiod", "VPD", "shading_rate", "rhizome_weight", "root_nitrogen", "glucosinolates")
- predicate: The relationship (e.g., "optimal_range", "measured_value", "set_value", "critical_threshold", "tested_range", "affects", "upper_limit", "minimum")
- object: The value or target
- unit: The measurement unit (e.g., "°C", "mg/L", "dS/m", "μmol/m²/s", "ppm", "mg/kg", "%")
- condition: Experimental condition if specified (e.g., "vermiculite substrate", "summer season", "artificial light")

Rules:
1. Only extract factual, quantitative triplets from the text. Do NOT hallucinate.
2. Include the exact values from the paper.
3. If a condition is stated, include it. If not explicit, set condition to null.
4. Use standardized subject names: water_temp, air_temp, growth_medium_temp, DO, EC, pH, AITC, PPFD, PPF, VPD, photoperiod, shading_rate, rhizome_weight, root_nitrogen, glucosinolates, GBPs, soil_moisture, CO2, altitude, summer_air_temp, WUE
5. Return ONLY valid JSON array of triplets.

Output format:
[
  {"subject": "water_temp", "predicate": "optimal", "object": "13", "unit": "°C", "condition": "indoor artificial light"},
  ...
]
"""

# ============================================================
# PDF Text Extraction
# ============================================================
def extract_text_from_pdf(pdf_path, max_pages=20):
    """Extract text from PDF using PyMuPDF."""
    doc = fitz.open(pdf_path)
    text_parts = []
    for i, page in enumerate(doc):
        if i >= max_pages:
            break
        text_parts.append(page.get_text())
    doc.close()
    return "\n".join(text_parts)


def chunk_text(text, max_chars=MAX_CHARS_PER_CHUNK):
    """Split text into chunks for API calls."""
    paragraphs = text.split("\n\n")
    chunks = []
    current = ""
    for p in paragraphs:
        if len(current) + len(p) > max_chars:
            if current:
                chunks.append(current)
            current = p
        else:
            current += "\n\n" + p if current else p
    if current:
        chunks.append(current)
    return chunks


# ============================================================
# GPT-4o Triplet Extraction
# ============================================================
def extract_triplets_gpt4o(client, text, source_id, paper_info=""):
    """Call GPT-4o to extract triplets from text."""
    chunks = chunk_text(text)
    all_triplets = []

    for i, chunk in enumerate(chunks):
        user_msg = f"""Paper: {paper_info} (source_id: {source_id})
Chunk {i+1}/{len(chunks)}:

{chunk}

Extract all knowledge triplets as JSON array:"""

        try:
            response = client.chat.completions.create(
                model=MODEL,
                temperature=TEMPERATURE,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg}
                ],
                max_tokens=4000
            )

            content = response.choices[0].message.content
            parsed = json.loads(content)

            # Handle both {"triplets": [...]} and [...] formats
            if isinstance(parsed, list):
                triplets = parsed
            elif isinstance(parsed, dict):
                triplets = parsed.get("triplets", parsed.get("data", []))
            else:
                triplets = []

            for t in triplets:
                t["source_id"] = source_id
                t["paper"] = paper_info
            all_triplets.extend(triplets)

            print(f"    Chunk {i+1}/{len(chunks)}: {len(triplets)} triplets extracted")
            time.sleep(0.5)  # Rate limiting

        except Exception as e:
            print(f"    ERROR on chunk {i+1}: {e}")

    return all_triplets


# ============================================================
# Main
# ============================================================
def main():
    if not OPENAI_API_KEY:
        print("ERROR: OPENAI_API_KEY 환경변수를 설정하세요.")
        print("  export OPENAI_API_KEY='sk-...'")
        sys.exit(1)

    client = OpenAI(api_key=OPENAI_API_KEY)

    # PDF 파일 매핑 (파일명 → source_id, paper_info)
    # 사용자의 PDF 파일들에 맞게 수정하세요
    pdf_configs = [
        {"file": "S01-Hoang_2019.pdf",       "source_id": "S01", "paper": "Hoang et al. 2019, Scientia Horticulturae"},
        {"file": "S02-Yamashita_2024.pdf",    "source_id": "S02", "paper": "Yamashita et al. 2024, Soil Sci Plant Nutr"},
        {"file": "S03-Oguni_2005.pdf",        "source_id": "S03", "paper": "Oguni et al. 2005, Environ Control Biol"},
        {"file": "S04-Soffer_Burger_1988.pdf", "source_id": "S04", "paper": "Soffer & Burger 1988, J ASHS"},
        {"file": "S05-Bugbee_2004.pdf",       "source_id": "S05", "paper": "Bugbee 2004, Acta Hort"},
        {"file": "S07-PLANT PRODUCTION IN A CLOSED PLANT FACTORY WITH ARTIFICIAL LIGHTING.pdf",
         "source_id": "S07", "paper": "Goto 2012, Acta Hort"},
        {"file": "S08-Principles of Nutrient and Water Management for Indoor Agriculture.pdf",
         "source_id": "S08", "paper": "Langenfeld et al. 2022, Sustainability"},
        {"file": "S09-Effects of Transmitted Light on Growth and Isocyanate Contents of Wasabi Rhizome Grown under Colored Shading Nets.pdf",
         "source_id": "S09", "paper": "Hisamatsu et al. 2020, Eco-Engineering"},
        {"file": "S10-Study on Cultivation of Japanese Horseradish Using Artificial Light.pdf",
         "source_id": "S10", "paper": "Tanaka et al. 2008, Eco-Engineering"},
        {"file": "S11-Effect-of-environmental-factors-in-nursery-on-growth-of-wasabi-seedlings-and-selection-of-suitable-cultivation-sites-in-and-near-Shizuoka-prefecture-based-on-the-factors.pdf",
         "source_id": "S11", "paper": "Kazaoka et al. 2023, Eco-Engineering"},
        {"file": "S13-A Review of Propagation Techniques and Isothiocyanates Content in Wasabi.pdf",
         "source_id": "S13", "paper": "Kofrankova & Koudela 2019, Acta Univ Agric"},
        {"file": "S13-Hot Pursuit, Searching for the Optimal Wasabi Greenhouse Growing Environment.pdf",
         "source_id": "S14", "paper": "Taylor et al. 2025, HortScience"},
        {"file": "S14-Variation in the main health-promoting compounds and antioxidant activity of different organs of WasabiS13-Hot Pursuit, Searching for the Optimal Wasabi Greenhouse Growing Environment.pdf",
         "source_id": "S15", "paper": "Di et al. 2022, Frontiers in Plant Science"},
    ]

    # PDF 폴더 (현재 디렉토리 또는 pdfs/ 하위 폴더)
    pdf_dir = os.path.join(os.path.dirname(__file__), "pdfs")
    if not os.path.exists(pdf_dir):
        pdf_dir = os.path.dirname(__file__)

    all_triplets = []
    triplet_counter = 0

    print("=" * 60)
    print("Step 1: GPT-4o Triplet Extraction")
    print("=" * 60)

    for cfg in pdf_configs:
        pdf_path = os.path.join(pdf_dir, cfg["file"])
        if not os.path.exists(pdf_path):
            print(f"\n[SKIP] {cfg['file']} — 파일 없음")
            continue

        print(f"\n[{cfg['source_id']}] {cfg['paper']}")
        print(f"  File: {cfg['file']}")

        text = extract_text_from_pdf(pdf_path)
        print(f"  Text length: {len(text)} chars")

        triplets = extract_triplets_gpt4o(client, text, cfg["source_id"], cfg["paper"])

        # Assign IDs
        for t in triplets:
            triplet_counter += 1
            t["id"] = f"GPT-{triplet_counter:03d}"

        all_triplets.extend(triplets)
        print(f"  Total from this paper: {len(triplets)}")

    # Save results
    output = {
        "metadata": {
            "version": "v1.2",
            "extraction_method": "GPT-4o (temperature=0.0, JSON mode)",
            "model": MODEL,
            "total_triplets": len(all_triplets),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "triplets": all_triplets
    }

    out_path = os.path.join(os.path.dirname(__file__), "triplets_gpt4o_extracted.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"총 {len(all_triplets)}개 트리플렛 추출 완료")
    print(f"저장: {out_path}")
    print(f"{'='*60}")

    return output


if __name__ == "__main__":
    main()

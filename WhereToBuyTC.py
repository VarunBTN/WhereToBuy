import os
import json
import re
from pprint import pprint
import mysql.connector
from serpapi.google_search import GoogleSearch
from dotenv import load_dotenv
from openai import OpenAI
from rapidfuzz import fuzz
import logging

# --------------------------
# Load environment variables
# --------------------------
load_dotenv()
SERPAPI_KEY = os.getenv("SERPAPI_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MYSQL_HOST = os.getenv("MYSQL_HOST")
MYSQL_USER = os.getenv("MYSQL_USER")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD")
MYSQL_DB = os.getenv("MYSQL_DB")
MYSQL_VIEW = os.getenv("MYSQL_VIEW")
MYSQL_TABLE = os.getenv("MYSQL_TABLE")
client = OpenAI(api_key=OPENAI_API_KEY)

BASE_IMAGE_URL = "https://static.londonwinecompetition.com/en/submissions/images/h/576/"

# --------------------------
# Setup logging
# --------------------------
LOG_FILE = "verification.log"
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

# --------------------------
# Utility functions
# --------------------------
def clean(text):
    if not text:
        return ""
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()

# --------------------------
# Product Verification Function
# --------------------------
def verify_product(target_name, candidate_name, vintage=None, varietal=None, threshold=85):
    target_clean = clean(target_name)
    candidate_clean = clean(candidate_name)

    # -------------------------
    # 1. Vintage check (strict)
    # -------------------------
    if vintage:
        if str(vintage) not in candidate_name:
            return False, "Rejected", 0, f"Vintage {vintage} not found in candidate"
    
    # -------------------------
    # 2. Fuzzy name match
    # -------------------------
    name_score = fuzz.token_sort_ratio(target_clean, candidate_clean)

    if name_score < threshold:
        return False, "Rejected", name_score, f"Name score too low: {name_score}"

    # -------------------------
    # 3. Varietal check (soft)
    # -------------------------
    confidence = name_score
    status = "Verified"

    if varietal:
        varietal_clean = clean(varietal)
        if varietal_clean and varietal_clean not in candidate_clean:
            status = "Likely"
            confidence = int(confidence * 0.9)  # penalty if varietal missing

    # -------------------------
    # 4. Return result
    # -------------------------
    return True, status, confidence, (
        f"Name matched ({name_score})"
        + (f", vintage {vintage} ok" if vintage else "")
        + (", varietal matched" if varietal and varietal_clean in candidate_clean else "")
    )

# --------------------------
# Google Shopping Search
# --------------------------

def build_query(product_name, producer=None, varietal=None, vintage=None, category=None):
    parts = []

    # Base product name
    parts.append(f'"{product_name}"')

    # Exact match quotes for important fields
    if producer:
        parts.append(f'"{producer}"')
    if varietal:
        parts.append(f'"{varietal}"')
    if vintage:
        parts.append(f'"{vintage}"')

    # Category hints
    if category:
        if "wine" in category.lower():
            parts.append("wine")
        elif "spirit" in category.lower():
            parts.append("spirits")
        elif "beer" in category.lower():
            parts.append("beer")

    return " ".join(parts)

def search_product(product_name, producer=None, varietal=None, vintage=None, category=None, location="London"):
    """
    Search for a product on Google Shopping using SerpAPI
    with an improved query.
    """
    query = build_query(product_name, producer, varietal, vintage, category)

    params = {
        "engine": "google_shopping",
        "q": query,
        "hl": "en",
        "gl": "uk",  # Targeting UK results (adjust as needed)
        "api_key": SERPAPI_KEY
    }

    search = GoogleSearch(params)
    results = search.get_dict()

    extracted_results = []
    for item in results.get("shopping_results", []):
        product_info = {
            "product_name": item.get("title"),
            "store_name": item.get("source"),
            "price": item.get("price"),
            "rating": item.get("rating"),
            "link": item.get("link")
        }
        extracted_results.append(product_info)

    return extracted_results

# --------------------------
# Google Lens Search
# --------------------------
def search_with_google_lens(image_url):
    params = {
        "engine": "google_lens",
        "url": image_url,
        "hl": "en",
        "country": "gb",
        "type": "products",
        "api_key": SERPAPI_KEY,
    }
    search = GoogleSearch(params)
    results = search.get_dict()
    extracted_results = []
    for item in results.get("visual_matches", []):
        product_info = {
            "product_name": item.get("title"),
            "store_name": item.get("source"),
            "link": item.get("link"),
            "thumbnail": item.get("thumbnail")
        }
        extracted_results.append(product_info)
    return extracted_results


# --------------------------
# LLM Fallback Generator
# --------------------------
def query_llm_for_places(product_name):
    prompt = f"""
You are a beverage and drinks expert.
Suggest up to 3 UK-based online or offline retail stores where someone could likely buy "{product_name}".
Always return at least one suggestion.
Return ONLY a JSON array with fields:
    store_name (string),
    url (string, if possible),
    reason (string explaining why the store is relevant)
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        text = response.choices[0].message.content.strip()

        # 🛠 Remove markdown code fences if present
        if text.startswith("```"):
            text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
            text = re.sub(r"```$", "", text).strip()

        llm_places = json.loads(text)
        logging.info(f"LLM fallback results for '{product_name}': {json.dumps(llm_places, indent=2)}")
        return llm_places

    except json.JSONDecodeError:
        logging.error(f"LLM fallback failed to parse JSON (after cleanup): {text}")
        return []
    except Exception as e:
        logging.error(f"LLM fallback failed: {e}")
        return []

# --------------------------
# Hybrid Search
# --------------------------
def hybrid_search(product_name, image_url=None, producer=None, varietal=None, vintage=None,
                  fuzzy_threshold=85):
    """
    Run hybrid search across Google Shopping and Google Lens,
    verifying candidates using fuzzy matching + vintage/varietal rules.
    """
    target_product = f"{product_name} {producer or ''} {varietal or ''} {vintage or ''}".strip()
    verified_results = []

    # --- Google Shopping ---
    shopping_results = search_product(product_name, producer, varietal, vintage)
    logging.info(f"Hybrid Search for '{product_name}' ({len(shopping_results)} shopping candidates)")
    print(f"\n🔎 Google Shopping candidates ({len(shopping_results)}):")

    for result in shopping_results:
        candidate_name = result["product_name"]
        print(f"Checking: {candidate_name}")

        match, status, confidence, reason = verify_product(
            target_product,
            candidate_name,
            vintage=vintage,
            varietal=varietal,
            threshold=fuzzy_threshold,
        )

        verified_status = "ACCEPTED" if match else "REJECTED"
        print(f"  Verified: {verified_status}, Status: {status}, Reason: {reason}, Confidence: {confidence}")
        logging.info(
            f"Candidate: {candidate_name} | Source: Google Shopping | "
            f"Verified: {verified_status} | Status: {status} | "
            f"Confidence: {confidence} | Reason: {reason}"
        )

        if match:
            result.update({
                "verification_reason": reason,
                "source": "Google Shopping",
                "category": status,
                "confidence": confidence,
            })
            verified_results.append(result)

    # --- Google Lens ---
    if image_url:
        lens_results = search_with_google_lens(image_url)
        logging.info(f"Hybrid Search for '{product_name}' ({len(lens_results)} lens candidates)")
        print(f"\n📸 Google Lens candidates ({len(lens_results)}):")

        for result in lens_results:
            candidate_name = result["product_name"]
            print(f"Checking: {candidate_name}")

            match, status, confidence, reason = verify_product(
                target_product,
                candidate_name,
                vintage=vintage,
                varietal=varietal,
                threshold=fuzzy_threshold,
            )

            verified_status = "ACCEPTED" if match else "REJECTED"
            print(f"  Verified: {verified_status}, Status: {status}, Reason: {reason}, Confidence: {confidence}")
            logging.info(
                f"Candidate: {candidate_name} | Source: Google Lens | "
                f"Verified: {verified_status} | Status: {status} | "
                f"Confidence: {confidence} | Reason: {reason}"
            )

            if match:
                result.update({
                    "verification_reason": reason,
                    "source": "Google Lens",
                    "category": status,
                    "confidence": confidence,
                })
                verified_results.append(result)

    # --- Fallback: LLM query ---
    if not verified_results:
        print("\n🤖 No direct matches found. Querying LLM for suggested retailers...")
        llm_results = query_llm_for_places(target_product)
        if llm_results:
            print(f"🤖 LLM fallback suggestions ({len(llm_results)}):")
            for place in llm_results:
                store = place.get("store_name", "Unknown Store")
                url = place.get("url", "")
                reason = place.get("reason", "AI recommendation")

                llm_result = {
                    "product_name": target_product,
                    "store_name": store,
                    "link": url,   # ✅ always use "link" key so it matches Shopping/Lens
                    "reason": reason,
                    "verification_reason": "AI-generated suggestion (no direct match found)",
                    "source": "LLM Fallback",
                    "category": "Suggested"
                }

                print(f"  Store: {store} | URL: {url} | Reason: {reason}")
                verified_results.append(llm_result)

            logging.info(f"LLM fallback results for '{target_product}': {json.dumps(llm_results, indent=2)}")
        else:
            print("🤖 LLM fallback returned no suggestions.")

    # --- Final results (take up to 3) ---
    final_results = verified_results[:3]
    return final_results

# --------------------------
# Unified Results Display
# --------------------------
def display_results(results):
    if not results:
        print("\n❌ No verified matches found.")
        return []

    print(f"\n✅ Found {len(results)} places where you can get the product:")
    print("=" * 80)

    for idx, place in enumerate(results, 1):
        print(f"\n📍 Result {idx}:")
        print(f"  Product: {place.get('product_name')}")
        print(f"  Store Name: {place.get('store_name')}")

        # Source label only for LLM fallback
        if place.get("source") == "LLM Fallback":
            print("  Source: AI Suggested")

        # Optional fields
        if place.get("price"):
            print(f"  Price: {place.get('price')}")
        if place.get("link"):
            print(f"  Link: {place.get('link')}")
        if place.get("rating"):
            print(f"  Rating: {place.get('rating')}")
        if place.get("verification_reason"):
            print(f"  Verification: {place.get('verification_reason')}")
        if place.get("category"):
            print(f"  Confidence: {place.get('category')}")
        if place.get("reason") and place.get("source") == "LLM Fallback":
            print(f"  Reason: {place.get('reason')}")

    return results

# --------------------------
# MySQL Fetch
# --------------------------
def fetch_products_from_mysql():
    conn = mysql.connector.connect(
        host=MYSQL_HOST,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_DB
    )
    cursor = conn.cursor(dictionary=True)
    cursor.execute(f"""
        SELECT competitionSubmissionsID, brandName, producerName, mainCategoryName, 
               variety1Name, vintageName, labelFile 
        FROM {MYSQL_VIEW} 
        LIMIT 5
    """)
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return rows

# --------------------------
# MySQL Update Function
# --------------------------
def update_where_to_buy(product_id, results):
    """
    Updates the MySQL table/view with up to 3 verified places.
    Creates columns if they don't exist.
    """
    conn = mysql.connector.connect(
        host=MYSQL_HOST,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_DB
    )
    cursor = conn.cursor()

    # --- Step 1: Ensure columns exist ---
    columns_to_check = [
        "whereTobuy1Name", "whereTobuy1url",
        "whereTobuy2Name", "whereTobuy2url",
        "whereTobuy3Name", "whereTobuy3url"
    ]
    cursor.execute(f"SHOW COLUMNS FROM {MYSQL_TABLE}")
    existing_columns = [col[0] for col in cursor.fetchall()]

    for col in columns_to_check:
        if col not in existing_columns:
            print(f"Adding missing column: {col}")
            cursor.execute(f"ALTER TABLE {MYSQL_TABLE} ADD COLUMN {col} VARCHAR(500)")

    # --- Step 2: Prepare values for update ---
    values = {}
    for idx in range(3):
        if idx < len(results):
            values[f"whereTobuy{idx+1}Name"] = results[idx].get("store_name")
            values[f"whereTobuy{idx+1}url"] = results[idx].get("link") or ""
        else:
            values[f"whereTobuy{idx+1}Name"] = None
            values[f"whereTobuy{idx+1}url"] = None

    # --- Step 3: Build SET clause dynamically ---
    set_clause = ", ".join(f"{k} = %s" for k in values.keys())
    sql = f"UPDATE {MYSQL_TABLE} SET {set_clause} WHERE competitionSubmissionsID = %s"
    params = list(values.values()) + [product_id]

    cursor.execute(sql, params)
    conn.commit()
    cursor.close()
    conn.close()
    print(f"✅ Updated 'where to buy' info for product ID {product_id}")


# --------------------------
# Main Runner
# --------------------------
if __name__ == "__main__":
    fuzzy_threshold = 85
    products = fetch_products_from_mysql()

    for product in products:
        product_id = product["competitionSubmissionsID"]
        product_name = product["brandName"]
        producer = product["producerName"]
        category = product["mainCategoryName"]
        varietal = product.get("variety1Name") if "wine" in category.lower() else None
        vintage = product.get("vintageName") if "wine" in category.lower() else None
        label_file = product.get("labelFile")
        image_url = f"{BASE_IMAGE_URL}{label_file}" if label_file else None

        print(f"\n🚀 Running hybrid search for: {product_name} ({product_id})")
        print(f"🔍 Search parameters: producer={producer}, varietal={varietal}, vintage={vintage}")
        
        verified_results = hybrid_search(
            product_name, image_url, producer, varietal, vintage, fuzzy_threshold
        )

        display_results(verified_results)

        # --- Upload results back to MySQL ---
        update_where_to_buy(product_id, verified_results)

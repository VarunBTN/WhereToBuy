import os
import json
import re
import mysql.connector
from serpapi.google_search import GoogleSearch
from dotenv import load_dotenv
from pprint import pprint
from openai import OpenAI
from fuzzywuzzy import fuzz

load_dotenv()

# --------------------------
# API KEYS
# --------------------------
SERPAPI_KEY = os.getenv("SERPAPI_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# --------------------------
# MySQL Connection
# --------------------------
DB_CONFIG = {
    "host": os.getenv("MYSQL_HOST", "localhost"),
    "user": os.getenv("MYSQL_USER", "root"),
    "password": os.getenv("MYSQL_PASSWORD", ""),
    "database": os.getenv("MYSQL_DB", "test_db")
}

def get_db_connection():
    return mysql.connector.connect(**DB_CONFIG)

# --------------------------
# Helpers (reuse your existing functions)
# --------------------------
def verify_with_brand_priority(target_product, candidate_product, brand=None, threshold=80):
    def clean(text):
        if not text:
            return ""
        return re.sub(r'\s+', ' ', str(text).lower().strip())
    
    target_clean = clean(target_product)
    candidate_clean = clean(candidate_product)
    
    if brand:
        brand_clean = clean(brand)
        if brand_clean not in candidate_clean:
            return False, f"Brand mismatch (target: {brand_clean})"
    
    ratio = fuzz.ratio(target_clean, candidate_clean)
    partial_ratio = fuzz.partial_ratio(target_clean, candidate_clean)
    token_sort_ratio = fuzz.token_sort_ratio(target_clean, candidate_clean)
    best_score = max(ratio, partial_ratio, token_sort_ratio)
    
    is_match = best_score >= threshold
    reason = f"Fuzzy match score: {best_score} (threshold: {threshold})"
    
    return is_match, reason

def safe_json_extract(text):
    try:
        return json.loads(text)
    except:
        match = re.search(r"\[.*\]", text, re.S)
        if match:
            try:
                return json.loads(match.group())
            except:
                return []
    return []

def search_product(product_name, producer=None, varietal=None, vintage=None, location="London"):
    query_parts = [product_name]
    if producer:
        query_parts.append(producer)
    if varietal:
        query_parts.append(varietal)
    if vintage:
        query_parts.append(str(vintage))
    query = " ".join(query_parts)
    
    params = {
        "engine": "google_shopping",
        "q": query,
        "hl": "en",
        "gl": "uk",
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
            "link": item.get("link"),
            "thumbnail": item.get("thumbnail")
        }
        extracted_results.append(product_info)
    return extracted_results

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

def llm_fallback_search(product_name):
    prompt = f"""
    You are a beverage and drinks expert.
    Suggest up to 3 UK-based online or offline retail stores where someone could likely buy "{product_name}".
    Always return at least one suggestion.
    Return ONLY a JSON array with fields:
        store_name (string),
        url (string, if possible),
        reason (string explaining why the store is relevant)
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.6
    )
    text_output = response.choices[0].message.content
    results = safe_json_extract(text_output)
    if not results:
        results = [{"store_name": "Generic UK Beverage Retailer", "url": "", "reason": f"Suggested fallback store for {product_name}"}]
    return results

def hybrid_search_and_verify(product_name, image_url=None, producer=None, varietal=None, vintage=None, fuzzy_threshold=80):
    target_product = f"{product_name} {producer or ''} {varietal or ''} {vintage or ''}".strip()
    verified_results = []

    shopping_results = search_product(product_name, producer, varietal, vintage)
    for result in shopping_results:
        match, reason = verify_with_brand_priority(target_product, result["product_name"], brand=producer, threshold=fuzzy_threshold)
        if match:
            result["verification_reason"] = reason
            result["source"] = "Google Shopping"
            verified_results.append(result)
    
    if verified_results:
        return verified_results

    if image_url:
        lens_results = search_with_google_lens(image_url)
        for result in lens_results:
            match, reason = verify_with_brand_priority(target_product, result["product_name"], brand=producer, threshold=fuzzy_threshold)
            if match:
                result["verification_reason"] = reason
                result["source"] = "Google Lens"
                verified_results.append(result)
        if verified_results:
            return verified_results
    
    fallback_results = llm_fallback_search(product_name)
    formatted_fallback = []
    for idx, result in enumerate(fallback_results):
        formatted_result = {
            "product_name": product_name,
            "store_name": result.get("store_name", f"Store {idx+1}"),
            "url": result.get("url", ""),
            "reason": result.get("reason", "LLM expert suggestion"),
            "source": "LLM Fallback"
        }
        formatted_fallback.append(formatted_result)
    
    return formatted_fallback

# --------------------------
# Database Integration
# --------------------------
def fetch_products_to_search():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT id, product_name, producer, varietal, vintage, image_url FROM products_to_search WHERE processed=0")
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return rows

def save_search_results(product_id, results):
    conn = get_db_connection()
    cursor = conn.cursor()
    for res in results:
        cursor.execute("""
            INSERT INTO product_search_results 
            (product_id, product_name, store_name, price, rating, link, url, thumbnail, reason, verification_reason, source)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """, (
            product_id,
            res.get("product_name"),
            res.get("store_name"),
            res.get("price"),
            res.get("rating"),
            res.get("link"),
            res.get("url"),
            res.get("thumbnail"),
            res.get("reason"),
            res.get("verification_reason"),
            res.get("source")
        ))
    cursor.execute("UPDATE products_to_search SET processed=1 WHERE id=%s", (product_id,))
    conn.commit()
    cursor.close()
    conn.close()

# --------------------------
# Main Runner
# --------------------------
if __name__ == "__main__":
    products = fetch_products_to_search()
    print(f"Found {len(products)} unprocessed products in DB.")

    for product in products:
        print(f"\n🚀 Processing product: {product['product_name']}")
        results = hybrid_search_and_verify(
            product['product_name'],
            image_url=product.get('image_url'),
            producer=product.get('producer'),
            varietal=product.get('varietal'),
            vintage=product.get('vintage')
        )
        pprint(results[:3])
        save_search_results(product['id'], results)
        print(f"✅ Results saved to DB for product ID {product['id']}")

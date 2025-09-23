import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import mysql.connector
from serpapi.google_search import GoogleSearch
from fuzzywuzzy import fuzz
from dotenv import load_dotenv

load_dotenv()

SERPAPI_KEY = os.getenv("SERPAPI_KEY")

app = FastAPI(title="Drink Availability API")


# --------------------------
# Database helper
# --------------------------
def get_db_connection():
    return mysql.connector.connect(
        host=os.getenv("DB_HOST", "localhost"),
        user=os.getenv("DB_USER", "root"),
        password=os.getenv("DB_PASS", ""),
        database=os.getenv("DB_NAME", "drinks_db")
    )


# --------------------------
# Models
# --------------------------
class DrinkRequest(BaseModel):
    product_id: int


class PlaceInfo(BaseModel):
    product_name: str
    store_name: str
    price: Optional[str]
    rating: Optional[float]
    link: Optional[str]


# --------------------------
# Core functions (search + filter)
# --------------------------
def get_product_from_db(product_id: int):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute(
        "SELECT id, product_name, producer, varietal, vintage, image_url FROM products WHERE id=%s",
        (product_id,)
    )
    row = cursor.fetchone()
    cursor.close()
    conn.close()
    return row


def search_product(product_name, producer=None, varietal=None, vintage=None):
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
    extracted = []
    for item in results.get("shopping_results", []):
        extracted.append({
            "product_name": item.get("title"),
            "store_name": item.get("source"),
            "price": item.get("price"),
            "rating": item.get("rating"),
            "link": item.get("link"),
            "thumbnail": item.get("thumbnail")
        })
    return extracted


def search_product_with_lens(image_url):
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
    extracted = []
    for item in results.get("visual_matches", []):
        extracted.append({
            "product_name": item.get("title"),
            "store_name": item.get("source"),
            "price": item.get("price"),
            "link": item.get("link"),
            "thumbnail": item.get("thumbnail")
        })
    return extracted


def filter_with_fuzzy_matching(results, target_product, threshold=80):
    negative_keywords = [
        "empty", "bottle only", "decanted", "decant", "used",
        "vintage bottle", "old bottle", "collector", "display",
        "ornamental", "set of", "pack of", "lot of", "souvenir", "miniature"
    ]
    matches = []
    for item in results:
        pname = item['product_name'].lower() if item['product_name'] else ""
        if any(k in pname for k in negative_keywords):
            continue
        score = fuzz.token_set_ratio(target_product.lower(), pname)
        if score >= threshold:
            itm = item.copy()
            itm['similarity_score'] = score
            matches.append(itm)
    matches.sort(key=lambda x: x['similarity_score'], reverse=True)
    return matches


def save_places_to_db(product_id, places):
    conn = get_db_connection()
    cursor = conn.cursor()
    # Create columns if they don't exist
    for i in range(1, 4):
        cursor.execute(f"ALTER TABLE products ADD COLUMN IF NOT EXISTS where_to_buy_{i} VARCHAR(255)")
    top_places = [place['store_name'] for place in places[:3]]
    while len(top_places) < 3:
        top_places.append(None)
    cursor.execute(
        f"UPDATE products SET where_to_buy_1=%s, where_to_buy_2=%s, where_to_buy_3=%s WHERE id=%s",
        (*top_places, product_id)
    )
    conn.commit()
    cursor.close()
    conn.close()


# --------------------------
# API Endpoint
# --------------------------
@app.post("/search_drink", response_model=List[PlaceInfo])
def search_drink(request: DrinkRequest):
    product = get_product_from_db(request.product_id)
    if not product:
        raise HTTPException(status_code=404, detail="Product not found in DB")

    results = search_product(
        product["product_name"],
        product["producer"],
        product["varietal"],
        product["vintage"]
    )
    filtered = filter_with_fuzzy_matching(results, product["product_name"], threshold=80)

    # Fallback to image search
    if not filtered and product.get("image_url"):
        lens_results = search_product_with_lens(product["image_url"])
        filtered = filter_with_fuzzy_matching(lens_results, product["product_name"], threshold=60)

    if not filtered:
        raise HTTPException(status_code=404, detail="No matching results found")

    # Save top 3 places to DB
    save_places_to_db(product["id"], filtered)

    # Return results
    return [
        PlaceInfo(
            product_name=r["product_name"],
            store_name=r["store_name"],
            price=r.get("price"),
            rating=r.get("rating"),
            link=r.get("link")
        )
        for r in filtered[:3]
    ]

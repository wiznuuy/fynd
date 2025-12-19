# -*- coding: utf-8 -*-
"""
FYND - Flask Backend API
의류 추천 시스템 웹 API
"""

import os
import json
import uuid
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image
import torch
import sqlite3
import pickle
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
from collections import defaultdict
from typing import List, Dict, Optional, Set

app = Flask(__name__)
CORS(app)

# 설정
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}

# 경로 설정
DB_PATH = "fashion_products.db"
EMBEDDING_DIR = "embeddings"
WISHLIST_PATH = "wishlist.json"

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(EMBEDDING_DIR, exist_ok=True)

# -------------------------
# 카테고리/키워드 매핑
# -------------------------
CATEGORY_GROUPS = {
    "hoodie": ["후드티", "후드 티", "hoodie", "hoody", "후디", "hooded sweatshirt"],
    "hood_coat": ["후드코트", "후드 코트", "hooded coat", "hood coat"],
    "coat": ["코트", "coat", "trench", "트렌치"],
    "jacket": ["자켓", "jacket", "블레이저", "blazer", "점퍼", "jumper"],
    "puffer": ["패딩", "puffer", "다운", "down", "padding", "퍼퍼"],
    "sweatshirt": ["맨투맨", "스웨트셔츠", "sweatshirt", "mtm"],
    "knit": ["니트", "knit", "스웨터", "sweater", "풀오버", "pullover"],
    "cardigan": ["가디건", "cardigan", "카디건"],
    "shirt": ["셔츠", "shirt", "남방"],
    "blouse": ["블라우스", "blouse"],
    "tshirt": ["티셔츠", "t-shirt", "tee", "반팔", "tshirt"],
    "dress": ["원피스", "dress", "드레스"],
    "skirt": ["스커트", "skirt", "치마"],
    "pants": ["팬츠", "pants", "바지", "트라우저", "trousers", "슬랙스", "slacks"],
    "jeans": ["진", "jeans", "청바지", "데님팬츠", "denim pants"],
    "shorts": ["반바지", "shorts", "숏팬츠", "숏츠"],
}

KEYWORD_TO_CATEGORY = {}
for cat, keywords in CATEGORY_GROUPS.items():
    for kw in keywords:
        KEYWORD_TO_CATEGORY[kw.lower()] = cat

DETAIL_KEYWORDS_EXPANDED = {
    "puff": ["퍼프", "퍼프소매", "puff sleeve", "puff", "볼륨소매"],
    "shirring": ["셔링", "shirring", "주름", "개더"],
    "off_shoulder": ["오프숄더", "off shoulder", "off-shoulder"],
    "one_shoulder": ["원숄더", "one shoulder", "one-shoulder"],
    "crop": ["크롭", "crop", "cropped", "짧은기장"],
    "asymmetric": ["언발란스", "asymmetric", "비대칭", "unbalance"],
    "stripe": ["스트라이프", "stripe", "줄무늬", "단가라"],
    "check": ["체크", "check", "plaid", "격자"],
    "floral": ["플로럴", "floral", "꽃무늬"],
    "ruffle": ["러플", "ruffle", "프릴", "frill"],
    "lace": ["레이스", "lace"],
    "eyelet": ["아일렛", "eyelet", "구멍", "펀칭"],
    "pleats": ["플리츠", "pleats", "주름"],
    "backless": ["백리스", "backless", "등트임"],
}

FIT_KEYWORDS_EXPANDED = {
    "oversized": ["오버사이즈", "오버핏", "oversized", "overfit", "루즈핏", "loose fit", "루즈", "박시", "boxy"],
    "slim": ["슬림", "슬림핏", "slim", "slim fit", "스키니", "skinny", "타이트", "tight"],
    "regular": ["레귤러", "레귤러핏", "regular", "regular fit"],
    "relaxed": ["릴렉스", "릴렉스핏", "relaxed", "relaxed fit"],
    "cropped": ["크롭", "cropped", "짧은"],
    "longline": ["롱라인", "longline", "롱핏", "긴기장"],
}

COLOR_KEYWORDS = {
    "black": ["블랙", "black", "검정", "검은"],
    "white": ["화이트", "white", "흰색", "하얀"],
    "gray": ["그레이", "gray", "grey", "회색"],
    "beige": ["베이지", "beige"],
    "brown": ["브라운", "brown", "갈색"],
    "navy": ["네이비", "navy", "남색", "곤색"],
    "blue": ["블루", "blue", "파란", "파랑"],
    "red": ["레드", "red", "빨강", "빨간"],
    "pink": ["핑크", "pink", "분홍"],
    "green": ["그린", "green", "초록", "녹색"],
    "ivory": ["아이보리", "ivory"],
    "cream": ["크림", "cream"],
}


# -------------------------
# 유틸리티 함수
# -------------------------
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def parse_query(query: str) -> Dict:
    query_lower = query.lower()
    result = {"category": None, "fit": [], "detail": [], "color": [], "raw_query": query}
    
    matched_categories = []
    for keyword, category in KEYWORD_TO_CATEGORY.items():
        if keyword in query_lower:
            matched_categories.append((len(keyword), keyword, category))
    if matched_categories:
        matched_categories.sort(reverse=True)
        result["category"] = matched_categories[0][2]
    
    for fit_name, keywords in FIT_KEYWORDS_EXPANDED.items():
        for kw in keywords:
            if kw.lower() in query_lower:
                result["fit"].append(fit_name)
                break
    
    for detail_name, keywords in DETAIL_KEYWORDS_EXPANDED.items():
        for kw in keywords:
            if kw.lower() in query_lower:
                result["detail"].append(detail_name)
                break
    
    for color_name, keywords in COLOR_KEYWORDS.items():
        for kw in keywords:
            if kw.lower() in query_lower:
                result["color"].append(color_name)
                break
    
    return result


def extract_product_category(product_name: str) -> Optional[str]:
    name_lower = product_name.lower()
    matched = []
    for keyword, category in KEYWORD_TO_CATEGORY.items():
        if keyword in name_lower:
            matched.append((len(keyword), category))
    if matched:
        matched.sort(reverse=True)
        return matched[0][1]
    return None


def expand_query(query: str, parsed: Dict) -> str:
    expanded_parts = [query]
    for fit in parsed.get("fit", []):
        if fit in FIT_KEYWORDS_EXPANDED:
            expanded_parts.extend(FIT_KEYWORDS_EXPANDED[fit][:3])
    for detail in parsed.get("detail", []):
        if detail in DETAIL_KEYWORDS_EXPANDED:
            expanded_parts.extend(DETAIL_KEYWORDS_EXPANDED[detail][:3])
    for color in parsed.get("color", []):
        if color in COLOR_KEYWORDS:
            expanded_parts.extend(COLOR_KEYWORDS[color][:2])
    return " ".join(expanded_parts)


def apply_brand_diversity(scores, products, top_k=3, max_per_brand=1, excluded_ids=None):
    if excluded_ids is None:
        excluded_ids = set()
    
    brand_count = defaultdict(int)
    selected_indices = []
    sorted_idx = np.argsort(scores)[::-1]
    
    for idx in sorted_idx:
        if len(selected_indices) >= top_k:
            break
        product = products[idx]
        product_id = product.get('id', idx)
        if product_id in excluded_ids:
            continue
        brand = product.get('brand_name', 'unknown')
        if brand_count[brand] < max_per_brand:
            selected_indices.append(idx)
            brand_count[brand] += 1
    
    return selected_indices


# -------------------------
# 추천 엔진 클래스
# -------------------------
class RecommendationEngine:
    def __init__(self):
        self.text_model = None
        self.clip_model = None
        self.clip_processor = None
        self.device = None
        self.products = []
        self.text_embeddings = None
        self.image_embeddings = None
        self.product_categories = []
        self.is_loaded = False
    
    def load_models(self):
        if self.is_loaded:
            return
        
        print("📦 모델 로딩 중...")
        self.text_model = SentenceTransformer("google/embeddinggemma-300m")
        
        self.clip_model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip")
        self.clip_processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")
        
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        
        self.clip_model = self.clip_model.to(self.device)
        print(f"✅ 모델 로딩 완료! (Device: {self.device})")
        
        self.load_products()
        self.load_or_build_embeddings()
        self.is_loaded = True
    
    def load_products(self):
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT 
                p.id, p.product_code, p.name, p.original_price, p.sale_price,
                p.image_url, p.product_url, p.category,
                b.name as brand_name,
                i.local_path as local_image_path
            FROM products p
            JOIN brands b ON p.brand_id = b.id
            LEFT JOIN images i ON p.id = i.product_id
        ''')
        
        self.products = []
        for row in cursor.fetchall():
            product = dict(row)
            local_path = product.get('local_image_path')
            if local_path and os.path.exists(local_path):
                self.products.append(product)
        
        conn.close()
        
        self.product_categories = [
            extract_product_category(p.get('name', '')) for p in self.products
        ]
        print(f"📊 {len(self.products)}개 상품 로드됨")
    
    def get_clip_image_embedding(self, image_path: str) -> np.ndarray:
        image = Image.open(image_path).convert("RGB")
        inputs = self.clip_processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            features = self.clip_model.get_image_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy().flatten()
    
    def load_or_build_embeddings(self):
        embedding_path = os.path.join(EMBEDDING_DIR, "web_embeddings.pkl")
        
        if os.path.exists(embedding_path):
            print("📂 임베딩 로드 중...")
            with open(embedding_path, 'rb') as f:
                data = pickle.load(f)
                self.text_embeddings = data['text_embeddings']
                self.image_embeddings = data['image_embeddings']
            print("✅ 임베딩 로드 완료!")
        else:
            print("🔨 임베딩 생성 중...")
            from tqdm import tqdm
            
            texts = [f"{p.get('brand_name', '')} {p.get('name', '')}" for p in self.products]
            self.text_embeddings = self.text_model.encode(
                texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True
            )
            
            image_embs = []
            for p in tqdm(self.products):
                try:
                    emb = self.get_clip_image_embedding(p['local_image_path'])
                    image_embs.append(emb)
                except:
                    image_embs.append(np.zeros(512))
            self.image_embeddings = np.array(image_embs)
            
            with open(embedding_path, 'wb') as f:
                pickle.dump({
                    'text_embeddings': self.text_embeddings,
                    'image_embeddings': self.image_embeddings
                }, f)
            print("✅ 임베딩 생성 완료!")
    
    def recommend_text_only(self, query: str, top_k: int = 3, excluded_ids: Set[int] = None) -> List[Dict]:
        """Model 1: 텍스트만으로 추천"""
        parsed = parse_query(query)
        expanded = expand_query(query, parsed)
        
        query_emb = self.text_model.encode(expanded, normalize_embeddings=True)
        similarities = np.dot(self.text_embeddings, query_emb)
        
        if parsed["category"]:
            for i, cat in enumerate(self.product_categories):
                if cat and cat != parsed["category"]:
                    similarities[i] *= 0.5
        
        selected = apply_brand_diversity(
            similarities, self.products, top_k, max_per_brand=1, excluded_ids=excluded_ids
        )
        
        results = []
        for rank, idx in enumerate(selected):
            p = self.products[idx]
            results.append({
                'id': p['id'],
                'name': p.get('name', ''),
                'brand': p.get('brand_name', ''),
                'price': p.get('sale_price', 0) or p.get('original_price', 0),
                'image_path': p.get('local_image_path', ''),
                'product_url': p.get('product_url', ''),
                'score': float(similarities[idx])
            })
        
        return results
    
    def recommend_text_image(self, query: str, image_path: str, top_k: int = 3, 
                            excluded_ids: Set[int] = None, text_weight: float = 0.4,
                            image_weight: float = 0.6) -> List[Dict]:
        """Model 2-2: 텍스트 + 이미지로 추천"""
        parsed = parse_query(query)
        expanded = expand_query(query, parsed)
        
        # 카테고리 필터링
        if parsed["category"]:
            candidate_indices = [
                i for i, cat in enumerate(self.product_categories)
                if cat == parsed["category"] or cat is None
            ]
            if len(candidate_indices) < 10:
                candidate_indices = list(range(len(self.products)))
        else:
            candidate_indices = list(range(len(self.products)))
        
        query_text_emb = self.text_model.encode(expanded, normalize_embeddings=True)
        query_image_emb = self.get_clip_image_embedding(image_path)
        
        candidate_text_embs = self.text_embeddings[candidate_indices]
        candidate_image_embs = self.image_embeddings[candidate_indices]
        
        text_sims = np.dot(candidate_text_embs, query_text_emb)
        image_sims = np.dot(candidate_image_embs, query_image_emb)
        
        combined = text_weight * text_sims + image_weight * image_sims
        
        candidate_products = [self.products[i] for i in candidate_indices]
        selected_local = apply_brand_diversity(
            combined, candidate_products, top_k, max_per_brand=1, excluded_ids=excluded_ids
        )
        
        results = []
        for rank, local_idx in enumerate(selected_local):
            global_idx = candidate_indices[local_idx]
            p = self.products[global_idx]
            results.append({
                'id': p['id'],
                'name': p.get('name', ''),
                'brand': p.get('brand_name', ''),
                'price': p.get('sale_price', 0) or p.get('original_price', 0),
                'image_path': p.get('local_image_path', ''),
                'product_url': p.get('product_url', ''),
                'score': float(combined[local_idx])
            })
        
        return results


# 전역 추천 엔진
engine = RecommendationEngine()


# -------------------------
# 위시리스트 관리
# -------------------------
def load_wishlist():
    if os.path.exists(WISHLIST_PATH):
        with open(WISHLIST_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []


def save_wishlist(items):
    with open(WISHLIST_PATH, 'w', encoding='utf-8') as f:
        json.dump(items, f, ensure_ascii=False, indent=2)


# -------------------------
# 라우트
# -------------------------
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/recommendation')
def recommendation():
    return render_template('recommendation.html')


@app.route('/wishlist')
def wishlist_page():
    return render_template('wishlist.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/mypage')
def mypage():
    return render_template('mypage.html')


@app.route('/images/<path:filename>')
def serve_image(filename):
    """이미지 서빙"""
    return send_from_directory('images', filename)


@app.route('/uploads/<path:filename>')
def serve_upload(filename):
    """업로드된 이미지 서빙"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/api/upload', methods=['POST'])
def upload_image():
    """이미지 업로드"""
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(f"{uuid.uuid4()}_{file.filename}")
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        return jsonify({'filename': filename, 'path': filepath})
    
    return jsonify({'error': 'Invalid file type'}), 400


@app.route('/api/recommend', methods=['POST'])
def recommend():
    """추천 API"""
    engine.load_models()
    
    data = request.json
    query = data.get('query', '')
    image_path = data.get('image_path', '')
    excluded_ids = set(data.get('excluded_ids', []))
    top_k = data.get('top_k', 3)
    
    if not query:
        return jsonify({'error': 'Query is required'}), 400
    
    try:
        if image_path and os.path.exists(image_path):
            results = engine.recommend_text_image(query, image_path, top_k, excluded_ids)
        else:
            results = engine.recommend_text_only(query, top_k, excluded_ids)
        
        return jsonify({'results': results})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/wishlist', methods=['GET'])
def get_wishlist():
    """위시리스트 조회"""
    items = load_wishlist()
    return jsonify({'items': items})


@app.route('/api/wishlist', methods=['POST'])
def add_to_wishlist():
    """위시리스트에 추가"""
    data = request.json
    items = load_wishlist()
    
    # 중복 체크
    for item in items:
        if item.get('id') == data.get('id'):
            return jsonify({'message': 'Already in wishlist'}), 200
    
    items.append({
        'id': data.get('id'),
        'name': data.get('name'),
        'brand': data.get('brand'),
        'price': data.get('price'),
        'image_path': data.get('image_path'),
        'product_url': data.get('product_url'),
        'added_at': datetime.now().isoformat()
    })
    
    save_wishlist(items)
    return jsonify({'message': 'Added to wishlist'})


@app.route('/api/wishlist/<int:item_id>', methods=['DELETE'])
def remove_from_wishlist(item_id):
    """위시리스트에서 제거"""
    items = load_wishlist()
    items = [item for item in items if item.get('id') != item_id]
    save_wishlist(items)
    return jsonify({'message': 'Removed from wishlist'})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
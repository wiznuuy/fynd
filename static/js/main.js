// FYND - Main JavaScript

const API_BASE = '';

// State
let currentQuery = '';
let uploadedImagePath = '';
let currentResults = [];
let excludedIds = new Set();

// Utils
function formatPrice(price) {
    return '₩' + Number(price).toLocaleString();
}

function showToast(message) {
    const toast = document.getElementById('toast');
    if (toast) {
        toast.textContent = message;
        toast.classList.add('active');
        setTimeout(() => toast.classList.remove('active'), 3000);
    }
}

function getGreeting() {
    const hour = new Date().getHours();
    if (hour < 12) return 'Good Morning';
    if (hour < 18) return 'Good Afternoon';
    return 'Good Evening';
}

// Landing Page Functions
function initLandingPage() {
    const greetingEl = document.getElementById('greeting');
    if (greetingEl) {
        greetingEl.textContent = getGreeting() + ',';
    }
    
    const form = document.getElementById('search-form');
    const searchInput = document.getElementById('search-input');
    const addImageBtn = document.getElementById('add-image-btn');
    const imageInput = document.getElementById('image-input');
    const imagePreview = document.getElementById('image-preview');
    
    if (form) {
        form.addEventListener('submit', handleSearchSubmit);
    }
    
    if (addImageBtn && imageInput) {
        addImageBtn.addEventListener('click', () => imageInput.click());
        imageInput.addEventListener('change', handleImageSelect);
    }
}

async function handleImageSelect(e) {
    const file = e.target.files[0];
    if (!file) return;
    
    const addImageBtn = document.getElementById('add-image-btn');
    const imagePreview = document.getElementById('image-preview');
    
    // Upload image
    const formData = new FormData();
    formData.append('image', file);
    
    try {
        const response = await fetch(`${API_BASE}/api/upload`, {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        if (data.path) {
            uploadedImagePath = data.path;
            addImageBtn.classList.add('has-image');
            addImageBtn.innerHTML = `
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M5 13l4 4L19 7"/>
                </svg>
                image added
            `;
            
            // Show preview
            if (imagePreview) {
                imagePreview.src = URL.createObjectURL(file);
                imagePreview.classList.add('active');
            }
        }
    } catch (error) {
        console.error('Image upload error:', error);
        showToast('이미지 업로드 실패');
    }
}

function handleSearchSubmit(e) {
    e.preventDefault();
    
    const searchInput = document.getElementById('search-input');
    const query = searchInput.value.trim();
    
    if (!query) {
        showToast('검색어를 입력해주세요');
        searchInput.focus();
        return;
    }
    
    // Store in sessionStorage
    sessionStorage.setItem('fynd_query', query);
    sessionStorage.setItem('fynd_image_path', uploadedImagePath);
    
    // Navigate to recommendation page
    window.location.href = '/recommendation';
}

// Recommendation Page Functions
function initRecommendationPage() {
    currentQuery = sessionStorage.getItem('fynd_query') || '';
    uploadedImagePath = sessionStorage.getItem('fynd_image_path') || '';
    excludedIds = new Set();
    
    if (!currentQuery) {
        window.location.href = '/';
        return;
    }
    
    loadRecommendations();
    
    const form = document.getElementById('refine-form');
    if (form) {
        form.addEventListener('submit', handleRefineSubmit);
    }
}

async function loadRecommendations() {
    const productsGrid = document.getElementById('products-grid');
    const loadingEl = document.getElementById('loading');
    
    if (loadingEl) loadingEl.style.display = 'flex';
    if (productsGrid) productsGrid.innerHTML = '';
    
    try {
        const response = await fetch(`${API_BASE}/api/recommend`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                query: currentQuery,
                image_path: uploadedImagePath,
                excluded_ids: Array.from(excludedIds),
                top_k: 3
            })
        });
        
        const data = await response.json();
        
        if (loadingEl) loadingEl.style.display = 'none';
        
        if (data.results) {
            currentResults = data.results;
            renderProducts(data.results);
        } else if (data.error) {
            showToast('추천 오류: ' + data.error);
        }
    } catch (error) {
        console.error('Recommendation error:', error);
        if (loadingEl) loadingEl.style.display = 'none';
        showToast('추천을 불러오는데 실패했습니다');
    }
}

function renderProducts(products) {
    const productsGrid = document.getElementById('products-grid');
    if (!productsGrid) return;
    
    productsGrid.innerHTML = '';
    
    products.forEach((product, index) => {
        const card = document.createElement('div');
        card.className = 'product-card-container';
        card.innerHTML = `
            <div class="product-card" data-product-id="${product.id}" data-index="${index}">
                <div class="product-image-container">
                    <img class="product-image" src="/${product.image_path}" alt="${product.name}" 
                         onerror="this.src='/static/images/placeholder.png'">
                </div>
                <div class="product-info">
                    <div class="product-name">${product.name}</div>
                    <div class="product-meta">
                        <span class="product-brand">${product.brand}</span>
                        <span class="product-price">${formatPrice(product.price)}</span>
                    </div>
                </div>
            </div>
            <div class="dislike-container">
                <button class="dislike-btn" data-product-id="${product.id}" title="마음에 안 들어요">
                    ✕
                </button>
            </div>
        `;
        
        productsGrid.appendChild(card);
        
        // Card click -> wishlist modal
        const cardEl = card.querySelector('.product-card');
        cardEl.addEventListener('click', () => showWishlistModal(product));
        
        // Dislike button
        const dislikeBtn = card.querySelector('.dislike-btn');
        dislikeBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            handleDislike(product.id);
        });
    });
}

function handleDislike(productId) {
    excludedIds.add(productId);
    loadRecommendations();
}

function handleRefineSubmit(e) {
    e.preventDefault();
    
    const refineInput = document.getElementById('refine-input');
    const newPrompt = refineInput.value.trim();
    
    if (!newPrompt) return;
    
    // Append to current query
    currentQuery = currentQuery + ' ' + newPrompt;
    sessionStorage.setItem('fynd_query', currentQuery);
    
    refineInput.value = '';
    loadRecommendations();
}

// Wishlist Modal
function showWishlistModal(product) {
    const modal = document.getElementById('wishlist-modal');
    if (!modal) return;
    
    modal.classList.add('active');
    modal.dataset.productId = product.id;
    modal.dataset.product = JSON.stringify(product);
}

function hideWishlistModal() {
    const modal = document.getElementById('wishlist-modal');
    if (modal) {
        modal.classList.remove('active');
    }
}

async function confirmAddToWishlist() {
    const modal = document.getElementById('wishlist-modal');
    const product = JSON.parse(modal.dataset.product);
    
    try {
        const response = await fetch(`${API_BASE}/api/wishlist`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(product)
        });
        
        const data = await response.json();
        hideWishlistModal();
        showToast('위시리스트에 추가되었습니다!');
    } catch (error) {
        console.error('Wishlist error:', error);
        showToast('위시리스트 추가 실패');
    }
}

// Wishlist Page Functions
async function initWishlistPage() {
    await loadWishlist();
}

async function loadWishlist() {
    const productsGrid = document.getElementById('wishlist-grid');
    const emptyEl = document.getElementById('wishlist-empty');
    
    try {
        const response = await fetch(`${API_BASE}/api/wishlist`);
        const data = await response.json();
        
        if (data.items && data.items.length > 0) {
            if (emptyEl) emptyEl.style.display = 'none';
            renderWishlistItems(data.items);
        } else {
            if (emptyEl) emptyEl.style.display = 'block';
            if (productsGrid) productsGrid.innerHTML = '';
        }
    } catch (error) {
        console.error('Wishlist load error:', error);
        showToast('위시리스트를 불러오는데 실패했습니다');
    }
}

function renderWishlistItems(items) {
    const productsGrid = document.getElementById('wishlist-grid');
    if (!productsGrid) return;
    
    productsGrid.innerHTML = '';
    
    items.forEach(item => {
        const card = document.createElement('div');
        card.className = 'product-card';
        card.innerHTML = `
            <div class="product-image-container">
                <img class="product-image" src="/${item.image_path}" alt="${item.name}"
                     onerror="this.src='/static/images/placeholder.png'">
            </div>
            <div class="product-info">
                <div class="product-name">${item.name}</div>
                <div class="product-meta">
                    <span class="product-brand">${item.brand}</span>
                    <span class="product-price">${formatPrice(item.price)}</span>
                </div>
            </div>
        `;
        
        card.addEventListener('click', () => {
            if (item.product_url) {
                window.open(item.product_url, '_blank');
            }
        });
        
        productsGrid.appendChild(card);
    });
}

// Initialize based on page
document.addEventListener('DOMContentLoaded', () => {
    const path = window.location.pathname;
    
    if (path === '/' || path === '/index.html') {
        initLandingPage();
    } else if (path === '/recommendation') {
        initRecommendationPage();
    } else if (path === '/wishlist') {
        initWishlistPage();
    }
});

from fastapi import FastAPI, File, UploadFile, HTTPException
from model_logic import get_vector
import numpy as np
import uvicorn
import firebase_admin
from firebase_admin import credentials, firestore

app = FastAPI()

# --- 1. Firebase 初始化 ---
# 請確保 serviceAccountKey.json 位於 C:\Users\User\pybackend\ 目錄下
try:
    cred = credentials.Certificate("serviceAccountKey.json")
    firebase_admin.initialize_app(cred)
    db = firestore.client()
    print("✅ Firebase Admin SDK 初始化成功")
except Exception as e:
    print(f"❌ Firebase 初始化失敗: {e}")

@app.get("/")
async def root():
    return {"status": "running", "message": "Zenkurenaido AI Backend is online"}

@app.post("/compare")
async def compare(product_id: str, file: UploadFile = File(...)):
    """
    此介面會接收產品 ID，並自動從 Firestore 抓取該產品的 Cloudinary 網址進行比對
    """
    try:
        # --- 2. 從 Firestore 獲取基準產品資料 ---
        # 根據你提供的截圖，集合名稱應為 "products"
        doc_ref = db.collection("products").document(product_id)
        doc = doc_ref.get()
        
        if not doc.exists:
            raise HTTPException(status_code=404, detail=f"找不到 ID 為 {product_id} 的產品")
        
        data = doc.to_dict()
        base_url = data.get("image")  # 這裡會抓到你的 https://res.cloudinary.com/... 網址
        product_name = data.get("name", "未知物品")

        # --- 3. 網址有效性檢查 ---
        if not base_url or not base_url.startswith("http"):
             raise HTTPException(
                 status_code=400, 
                 detail=f"該產品圖片網址無效 (目前為: {base_url})。請確保圖片已成功上傳至 Cloudinary。"
             )

        print(f"🔍 正在比對物品: {product_name}")
        print(f"🌐 基準圖網址: {base_url}")

        # --- 4. 提取基準圖向量 (從 Cloudinary 下載) ---
        base_vector = get_vector(base_url, is_url=True)
        
        # --- 5. 提取目前上傳圖向量 (手機傳來的圖片) ---
        content = await file.read()
        current_vector = get_vector(content, is_url=False)
        
        # --- 6. 計算餘弦相似度 ---
        # 相似度公式: (A · B) / (||A|| * ||B||)
        similarity = np.dot(current_vector, base_vector) / (np.linalg.norm(current_vector) * np.linalg.norm(base_vector))
        
        # --- 7. 設定判定門檻 (建議 0.75 - 0.8) ---
        match_threshold = 0.7
        is_match = bool(similarity > match_threshold)
        
        return {
            "status": "success",
            "match": is_match,
            "score": round(float(similarity), 4),
            "product_name": product_name,
            "threshold": match_threshold
        }

    except Exception as e:
        print(f"⚠️ 伺服器內部錯誤: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # 啟動伺服器，監聽 8000 端口
    uvicorn.run(app, host="0.0.0.0", port=8000)
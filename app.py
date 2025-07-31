import streamlit as st
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# ---------------------------
# Load model từ file
# ---------------------------
def load_model(model_path: str):
    model = joblib.load(model_path)
    return model

# ---------------------------
# LabelEncoder tạm thời (demo)
# ---------------------------
def temporary_label_encoder(df: pd.DataFrame, col: str) -> pd.DataFrame:
    encoder = LabelEncoder()
    df[col] = encoder.fit_transform(df[col])
    return df

# ---------------------------
# Cấu hình App Streamlit
# ---------------------------
def main():
    st.set_page_config(page_title="Dự đoán giá HDB", layout="centered")
    st.title("🏘️ Dự đoán Giá Bất Động Sản (HDB) Singapore")

    st.markdown("Nhập các thông tin dưới đây để hệ thống dự đoán giá bán lại (resale price) của căn hộ HDB.")
    st.markdown("---")

    # ---------------------------
    # Nhập dữ liệu từ người dùng
    # ---------------------------
    year = st.number_input("📅 Năm bán lại", value=2020, step=1)

    town = st.selectbox("🏙️ Khu vực (Town)", [
        "ANG MO KIO", "BEDOK", "BISHAN", "BUKIT BATOK", "BUKIT MERAH",
        "BUKIT PANJANG", "BUKIT TIMAH", "CENTRAL AREA", "CHOA CHU KANG",
        "CLEMENTI", "GEYLANG", "HOUGANG", "JURONG EAST", "JURONG WEST",
        "KALLANG/WHAMPOA", "MARINE PARADE", "PASIR RIS", "PUNGGOL",
        "QUEENSTOWN", "SEMBAWANG", "SENGKANG", "SERANGOON", "TAMPINES",
        "TOA PAYOH", "WOODLANDS", "YISHUN"
    ])

    flat_type = st.selectbox("🏠 Loại căn hộ", [
        "1 ROOM", "2 ROOM", "3 ROOM", "4 ROOM", "5 ROOM", "EXECUTIVE"
    ])

    floor_area_sqm = st.number_input("📐 Diện tích (m²)", value=95.0, step=1.0)

    remaining_lease_years = st.number_input("📜 Thời gian thuê còn lại (năm)", value=74, step=1)

    distance_from_expressway = st.selectbox("🚗 Khoảng cách tới đường cao tốc", [
        "<=50m", "50m-100m", "100m-200m", ">500m"
    ])

    # ---------------------------
    # Nút Dự đoán
    # ---------------------------
    if st.button("🔍 Dự đoán"):
        try:
            model = load_model("random_forest_resale_price_model.joblib")

            # Tạo DataFrame từ input
            data = {
                "year": [year],
                "town": [town],
                "flat_type": [flat_type],
                "floor_area_sqm": [floor_area_sqm],
                "remaining_lease_years": [remaining_lease_years],
                "distance_from_expressway": [distance_from_expressway],
            }
            X_new = pd.DataFrame(data)

            # Mã hoá label tạm thời (chỉ dùng nếu model không có pipeline chuẩn)
            X_new = temporary_label_encoder(X_new, "town")
            X_new = temporary_label_encoder(X_new, "flat_type")
            X_new = temporary_label_encoder(X_new, "distance_from_expressway")

            # Dự đoán
            prediction = model.predict(X_new)[0]

            # Hiển thị kết quả
            st.success(f"💰 Giá ước tính: **{round(prediction, 2):,.0f} SGD**")

        except FileNotFoundError:
            st.error("❌ Không tìm thấy model! Vui lòng chắc chắn file 'random_forest_resale_price_model.joblib' tồn tại trong thư mục.")
        except Exception as e:
            st.error(f"Đã xảy ra lỗi: {e}")

    # ---------------------------
    # Hiển thị dữ liệu mẫu
    # ---------------------------
    st.markdown("---")
    if st.button("📊 Tham khảo giá bán lại trên thị trường"):
        try:
            df_raw = pd.read_csv("resale.csv")  # Đặt đúng tên file CSV
            st.markdown("### 🗂️ Dữ liệu Bán lại HDB trên thị trường hiện nay")
            st.dataframe(df_raw)  # Hiển thị 50 dòng đầu tiên
        except FileNotFoundError:
            st.error("❌ Không tìm thấy file 'resale.csv'!")
        except Exception as e:
            st.error(f"Đã xảy ra lỗi khi tải dữ liệu: {e}")

# ---------------------------
# Chạy app
# ---------------------------
if __name__ == "__main__":
    main()

import streamlit as st
import joblib
import pandas as pd
import numpy as np

# ---------------------------
# Load model từ file
# ---------------------------
def load_model(model_path: str):
    model = joblib.load(model_path)
    return model

# ---------------------------
# Tiền xử lý 'distance_from_expressway'
# ---------------------------
def preprocess_distance(df: pd.DataFrame) -> pd.DataFrame:
    def convert_distance(value):
        if isinstance(value, str):
            value = value.replace('m', '').strip()
            try:
                if '-' in value:
                    # Xử lý khoảng cách, ví dụ: '101-150m'
                    parts = value.split('-')
                    low = float(parts[0])
                    high = float(parts[1])
                    return (low + high) / 2
                elif value.startswith('<='):
                    # Xử lý '<=50m'
                    num = float(value.replace('<=', '').strip())
                    return num
                elif value.startswith('>'):
                    # Xử lý '>500m'
                    num = float(value.replace('>', '').strip())
                    return num
                else:
                    # Xử lý giá trị đơn lẻ, ví dụ: '300m'
                    return float(value)
            except (ValueError, TypeError):
                return np.nan  # Trả về NaN nếu không thể chuyển đổi
        elif np.issubdtype(type(value), np.number):
            return value
        else:
            return np.nan  # Trả về NaN cho các loại dữ liệu khác

    df['distance_from_expressway'] = df['distance_from_expressway'].apply(convert_distance)
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

    storey_range_category = st.selectbox("🏢 Khoảng tầng", [
        "Low (01-06)", "Low-Mid (07-12)", "Mid (13-18)", "High (19-30)", "Very High (>30)"
    ])

    block = st.text_input("🏢 Tòa nhà (Block)", value="101")
    street_name = st.text_input("🛣️ Tên đường (Street Name)", value="ANG MO KIO AVE 1")

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
                "storey_range_category": [storey_range_category],
                "block": [block],
                "street_name": [street_name]
            }
            X_new = pd.DataFrame(data)

            # Tiền xử lý distance_from_expressway
            X_new = preprocess_distance(X_new)

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
            st.dataframe(df_raw.head(50))  # Hiển thị 50 dòng đầu tiên
        except FileNotFoundError:
            st.error("❌ Không tìm thấy file 'resale.csv'!")
        except Exception as e:
            st.error(f"Đã xảy ra lỗi khi tải dữ liệu: {e}")

# ---------------------------
# Chạy app
# ---------------------------
if __name__ == "__main__":
    main()

import requests
import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from PIL import Image
import json

st.set_page_config(page_title="DapoerKito")

class ImageClassifierApp:
    def __init__(self, model_path):
        """Inisialisasi model dan daftar kelas."""
        self.used_recipes = set()  
        self.model = load_model(model_path)
        self.class_labels = ['Daging Ayam', 'Daging Sapi', 'Durian', 'Ikan', 'Tahu', 'Telur', 'Tempe', 'Udang']

    def create_sidebar(self):
        """Create a sidebar with app description and usage instructions."""
        with st.sidebar:
            st.title("DapoerKito")
            
            # App description
            st.markdown("### Tentang Aplikasi")
            st.write("""
            DapoerKito adalah aplikasi pengenal bahan makanan yang memberikan 
            rekomendasi resep masakan khas Sumatera Selatan berdasarkan bahan 
            yang terdeteksi dari gambar yang diunggah.
            """)
            
            # How to use
            st.markdown("### Cara Penggunaan")
            st.write("""
            1. **Unggah Gambar** - Pilih gambar bahan makanan dari perangkat Anda
            2. **Atur Jumlah Resep** - Tentukan berapa banyak resep yang ingin ditampilkan
            3. **Lakukan Prediksi** - Klik tombol untuk mendapatkan hasil klasifikasi dan rekomendasi resep
            """)
            
            # Credits or additional info
            st.markdown("---")
            st.markdown("### Dikembangkan oleh")
            st.write("Farhah Salsabillah")

    def preprocess_image(self, image, target_size=(256, 256)):
        """Preprocess gambar sebelum diklasifikasikan."""
        image = image.resize(target_size)
        img_array = img_to_array(image)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
        return img_array

    def predict(self, image):
        """Melakukan prediksi pada gambar yang diberikan."""
        img_array = self.preprocess_image(image)
        prediction = self.model.predict(img_array)
        predicted_class = self.class_labels[np.argmax(prediction)]
        confidence = np.max(prediction) * 100
        return predicted_class, confidence

    def get_recipes(self, jumlah, input_bahan):
        """Mengambil resep dari API Deepseek dengan streaming."""
        url = "https://api.deepseek.com/chat/completions"
        headers = {"Content-Type": "application/json",
                "Authorization": "Bearer sk-1aaebdb4a21746e3ae27d9288fa95fe5"}
        
        data = {
            "model": "deepseek-chat",
            "messages": [
                {
                    "role": "system",
                    "content": "Anda seorang ibu rumah tangga asal Sumatera Selatan yang hanya bisa berbahasa Indonesia, Anda pandai memasak masakan-masakan yang sering dimasak oleh masyarakat Sumatera Selatan dengan bahan-bahan masakan yang umum digunakan. Resep-resep yang Anda ketahui adalah resep asli namun juga resep umum."
                },
                {
                    "role": "user",
                    "content": f"""Buatkan {jumlah} resep masakan khas Sumatera Selatan yang berbahan dasar {input_bahan}.
                                Sertakan nama makanan, daerah kota asalnya, daftar bahan, langkah-langkah memasak secara detail dari banyaknya bumbu yang digunakan,
                                durasi menunggu, dan sebagainya. 
                                Pastikan resep yang diberikan benar-benar autentik sesuai dengan cita rasa khas daerah asalnya.
                                Jika ada langkah yang tidak perlu, jangan dituliskan. Jika ada langkah yang tidak umum, jangan dituliskan.
                                Langsung berikan daftar resep tanpa keterangan di awal.
                                Buatlah resep tersebut dalam format markdown seperti:
                                # 1. Judul Resep (Asal: Asal Resep)
                                ## Bahan:
                                1. List Bahan
                                ## Cara Membuat:
                                1. List Cara Membuat"""
                }
            ],
            "temperature": 1,
            "stream": True
        }
        
        # Create a placeholder for streaming text
        recipe_placeholder = st.empty()
        full_response = ""
        
        with st.spinner("Menulis resep..."):
            response = requests.post(url, headers=headers, data=json.dumps(data), stream=True)
            
            if response.status_code == 200:
                # Process the streaming response
                for line in response.iter_lines():
                    if line:
                        line_text = line.decode('utf-8')
                        # Skip the "data: " prefix and empty lines
                        if line_text.startswith('data: ') and line_text != 'data: [DONE]':
                            json_str = line_text[6:]  # Remove 'data: ' prefix
                            try:
                                chunk_data = json.loads(json_str)
                                if 'choices' in chunk_data and len(chunk_data['choices']) > 0:
                                    chunk = chunk_data['choices'][0].get('delta', {}).get('content', '')
                                    if chunk:
                                        full_response += chunk
                                        # Update the placeholder with the accumulated text
                                        recipe_placeholder.markdown(full_response)
                            except json.JSONDecodeError:
                                pass
                
                return full_response
            else:
                recipe_placeholder.error("Terjadi kesalahan dalam mengambil data resep.")
                return "Terjadi kesalahan dalam mengambil data resep."

    def display_recipes(self, predicted_class, num_recipes):
        """Menampilkan rekomendasi resep dengan format streaming di Streamlit."""
        st.write("### Rekomendasi Resep:")
        
        # The get_recipes method now handles the display with streaming
        recipes = self.get_recipes(jumlah=num_recipes, input_bahan=predicted_class)
        
        if not recipes or recipes == "Terjadi kesalahan dalam mengambil data resep.":
            st.write("### Maaf, tidak ada resep untuk bahan ini.")
    
    def run(self):
        """Menjalankan aplikasi Streamlit."""

        # Create sidebar
        self.create_sidebar()

        st.markdown("""
            <h2 style='text-align: center; color: #4CAF50;'>Selamat Datang di DapoerKito!</h2>
            <p style='text-align: center; color: #333333; font-size: 18px;'>
            Aplikasi ini membantu Anda menemukan resep masakan khas Sumatera Selatan 
            berdasarkan bahan makanan yang Anda miliki. Unggah gambar atau masukkan 
            bahan utama untuk mendapatkan rekomendasi resep yang lezat dan autentik.
            </p>
            """, unsafe_allow_html=True)
        st.write("Unggah gambar untuk diklasifikasikan oleh model.")

        uploaded_file = st.file_uploader("Pilih gambar", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            # Baca gambar
            image = Image.open(uploaded_file)

            # Layout dengan dua kolom
            col1, col2 = st.columns([3, 2])  # Kolom pertama lebih besar untuk gambar

            with col1:
                # Tampilkan gambar
                st.image(image, caption="Gambar yang diunggah", width=300)

            with col2:
                # Input jumlah resep
                num_recipes = st.number_input("Jumlah resep", min_value=1, max_value=10, value=1, step=1)

                # Tombol prediksi
                prediksi_clicked = st.button("Lakukan Prediksi")

            # prediksi_clicked = st.button("Lakukan Prediksi")

            # Setelah tombol ditekan, hasilnya akan muncul di bawah, tidak dalam kolom
            if prediksi_clicked:
                predicted_class, confidence = self.predict(image)

                # Tampilkan hasil prediksi
                st.write(f"**Prediksi: {predicted_class} ({confidence:.2f}%)**")

                # Tampilkan rekomendasi resep
                self.display_recipes(predicted_class, num_recipes)

# Inisialisasi aplikasi dengan model yang telah dilatih
if __name__ == "__main__":
    app = ImageClassifierApp("update_best_model.keras")
    app.run()

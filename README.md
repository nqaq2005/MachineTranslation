🌍 Hệ thống Dịch máy Song ngữ Anh - Việt (Bidirectional NMT)

Dự án Xử lý Ngôn ngữ Tự nhiên (NLP) xây dựng mô hình Dịch máy hai chiều (Tiếng Anh $\leftrightarrow$ Tiếng Việt) dựa trên kiến trúc Seq2Seq GRU kết hợp Cơ chế Attention toàn cục (Global Attention).

Mô hình được thiết kế chia sẻ trọng số mạng lõi (Encoder/Decoder) để xử lý cả hai chiều ngôn ngữ trên cùng một luồng, tối ưu hóa VRAM cực mạnh thông qua xử lý đồ thị tính toán độc lập.

🚀 Tính năng nổi bật

Dịch đa chiều (Bidirectional): Sử dụng token điều hướng (<Vi2En>, <En2Vi>) để quyết định chiều dịch ngay trong quá trình giải mã.

Tối ưu VRAM đỉnh cao: Phá vỡ đồ thị tính toán (Computation Graph) thành 2 luồng backward độc lập, giảm 50% Peak VRAM.

Tích hợp FastText: Khởi tạo không gian ngữ nghĩa bằng FastText 300d (freeze core, fine-tune special tokens).

Deep Output Layer: Mạng xuất dự đoán sử dụng kiến trúc Deep GELU Block (Linear $\rightarrow$ LayerNorm $\rightarrow$ GELU) để hiểu sâu ngữ pháp.

Mixed Precision & Gradient Accumulation: Tăng tốc huấn luyện bằng torch.autocast (float16) và GradScaler.

🧠 Tổng quan Kiến trúc

Encoder: GRU đa tầng (2 layers), 2 chiều (Bidirectional), hidden_dim = 256.

Decoder: GRU đơn hướng (1 layer).

Attention: Cơ chế Luong Attention (Dot-Product) tính điểm số chú ý.

Tham số: Tổng cộng khoảng 224.8 Triệu tham số (Trainable).

| Thành phần | Số lượng tham số |
| :--- | :--- |
| Encoder parameters | 120,825,696 |
| Decoder parameters | 120,825,696 |
| OutputLayer (VI) params | 51,598,660 |
| OutputLayer (EN) params | 51,598,660 |
| **Total Seq2Seq parameters** | **224,846,312** |

💾 Tập dữ liệu & Tiền xử lý (Data Collating)

Nhóm sử dụng bộ dữ liệu KietReal/Vietnamese-English-translation từ HuggingFace, với 1 triệu dòng được trích xuất cho quá trình huấn luyện.

Để xử lý batching động với độ dài câu khác nhau, mô hình áp dụng kỹ thuật nhúng token điều hướng. Ví dụ với chiều Việt $\rightarrow$ Anh:

Encoder input (Vi): ["tôi", "thích", "học", "máy", "<PAD>", "<PAD>"]

Decoder input (En): ["<Vi2En>", "i", "like", "machine", "learning", "<PAD>"]

Label (En): ["i", "like", "machine", "learning", "<EOS_EN>", "<PAD>"]

⚙️ Cài đặt Môi trường

Bản code được phát triển và tối ưu để chạy trên môi trường có hỗ trợ GPU (như Google Colab hoặc Local Server có CUDA).

# 1. Clone repository
git clone [https://github.com/your-repo/Machine-Translation-Project.git](https://github.com/your-repo/Machine-Translation-Project.git)
cd Machine-Translation-Project

# 2. Cài đặt các thư viện cần thiết
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
pip install datasets fasttext wandb tqdm


🏃 Hướng dẫn chạy Code

Dự án được chia thành 3 giai đoạn độc lập: Xây dựng từ vựng, Huấn luyện và Suy luận.

Bước 1: Xây dựng Từ vựng (Vocabulary)

Tải trước 2 file mô hình FastText (cc.vi.300.bin và cc.en.300.bin) và đặt cùng thư mục.

python make_vocab.py


Kết quả: Hệ thống sẽ sinh ra 2 file vi_vocab_embeddings.pt và en_vocab_embeddings.pt chứa bảng băm (word2idx, idx2word) và ma trận trọng số vector nhúng.

Bước 2: Huấn luyện Mô hình (Training)

Kiểm tra cấu hình đường dẫn dataset_path, save_dir bên trong file trainer.py và chạy lệnh:

python trainer.py


Lưu ý: Trong quá trình huấn luyện, mã nguồn tích hợp sẵn wandb để vẽ biểu đồ Loss, BLEU, chrF theo thời gian thực và có cơ chế tự động Resume bằng file checkpoints_latest.pt nếu bị ngắt kết nối.

Bước 3: Kiểm thử & Dịch thuật (Inference)

Trỏ file Checkpoint tốt nhất vào biến ckpt_path trong file inference.py để sử dụng Chatbot dịch thuật trên Terminal:

python inference.py


Ví dụ khi chạy Terminal:

🤖 CHATBOT DỊCH THUẬT (Gõ 'exit' hoặc 'quit' để thoát)
===================================================
📝 Nhập câu cần dịch: tôi thích học máy
🔄 Hướng dịch (1: Việt -> Anh | 2: Anh -> Việt): 1
🇺🇸 English: I like machine learning

📝 Nhập câu cần dịch: deep learning is very powerful
🔄 Hướng dịch (1: Việt -> Anh | 2: Anh -> Việt): 2
🇻🇳 Tiếng Việt: học sâu rất mạnh mẽ


📈 Metric Đánh giá

SacreBLEU: Đánh giá độ chính xác của cụm n-gram.

chrF: Đánh giá dựa trên mức độ ký tự (character n-gram F-score), đặc biệt hiệu quả đối với các ngôn ngữ có hình thái phong phú như Tiếng Việt.

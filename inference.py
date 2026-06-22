import torch
import torch.nn as nn
import re
import os
import gdown  # Thêm thư viện gdown để tải file từ Google Drive

# --- 1. COPY CÁC HÀM XỬ LÝ TEXT TỪ MAKE_VOCAB ---
def processing_english(text):
    text = text.lower()
    pattern = r"[a-z0-9]+(?:-[a-z0-9]+)*|'s|[.,?]"
    return re.findall(pattern, text)

def processing_vietnamese(text):
    text = text.lower()
    pattern = r"\w+(?:-\w+)*|[.,?]"
    return re.findall(pattern, text, flags=re.UNICODE)

# --- 2. COPY CÁC CLASS MÔ HÌNH VÀ ATTENTION ---
def attention_scores(output_encoder, output_decoder):
    hidden_state_decoder = output_decoder.unsqueeze(-1)
    return torch.bmm(output_encoder, hidden_state_decoder).squeeze(-1)

def attention_weight(attention_scores, mask):
    min_value = torch.finfo(attention_scores.dtype).min
    weight = attention_scores.masked_fill(mask, value=min_value)
    return torch.softmax(weight, dim=1)

def attention_output(attention_weight, output_encoder):
    attention_weight = attention_weight.unsqueeze(-1)
    ct = output_encoder * attention_weight
    ct = torch.sum(ct, dim=1)
    return ct

def attention(output_encoder, output_decoder, mask):
    score   = attention_scores(output_encoder, output_decoder)
    weights = attention_weight(score, mask)
    output  = attention_output(weights, output_encoder)
    return output

class GRU_encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, device, vi_embedding, en_embedding):
        super().__init__()
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim,
                          num_layers=num_layers, device=device, batch_first=True)
        self.vi_embedding = vi_embedding
        self.en_embedding = en_embedding

    def forward(self, Input, lengths, vi2en):
        embedded = self.vi_embedding(Input) if vi2en else self.en_embedding(Input)
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_output, h_n = self.gru(packed_embedded)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        return output, h_n

class GRU_decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, device, vi_embedding, en_embedding):
        super().__init__()
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim,
                          num_layers=num_layers, device=device, batch_first=True)
        self.vi_embedding = vi_embedding
        self.en_embedding = en_embedding

    def forward(self, Input, h_0, vi2en):
        # Đảo ngược ngôn ngữ so với encoder
        embedded = self.en_embedding(Input) if vi2en else self.vi_embedding(Input)
        output, h_n = self.gru(embedded, h_0)
        return output, h_n

class OutputLayer(nn.Module):
    def __init__(self, hidden_dim, vocab_size):
        super().__init__()
        self.layer1 = nn.Linear(in_features=hidden_dim * 2, out_features=hidden_dim, bias=False)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.act1 = nn.GELU()

        self.layer2 = nn.Linear(in_features=hidden_dim, out_features=hidden_dim, bias=False)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.act2 = nn.GELU()

        self.classifier = nn.Linear(in_features=hidden_dim, out_features=vocab_size, bias=True)

    def forward(self, x):
        x = self.act1(self.norm1(self.layer1(x)))
        x = self.act2(self.norm2(self.layer2(x)))
        logits = self.classifier(x)
        return logits


# --- 3. CLASS DỊCH THUẬT (TRANSLATOR) ---
class Translator:
    def __init__(self, checkpoint_path, vi_vocab_path, en_vocab_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Khởi tạo Inference trên: {self.device}")

        # 1. Load Vocab & Embeddings
        vi_data = torch.load(vi_vocab_path, weights_only=False)
        en_data = torch.load(en_vocab_path, weights_only=False)

        self.vi_word2idx = vi_data['word2idx']
        self.vi_idx2word = vi_data['idx2word']
        self.en_word2idx = en_data['word2idx']
        self.en_idx2word = en_data['idx2word']

        vi_embedd = nn.Embedding.from_pretrained(vi_data['embedding'].to(self.device), freeze=True)
        en_embedd = nn.Embedding.from_pretrained(en_data['embedding'].to(self.device), freeze=True)

        vi_vocab_size = len(self.vi_idx2word)
        en_vocab_size = len(self.en_idx2word)

        # Thông số mạng dựa trên setup của bạn
        input_dim = 300
        hidden_dim = 256
        num_layers = 2

        # 2. Khởi tạo Models
        self.encoder = GRU_encoder(input_dim, hidden_dim, num_layers, self.device, vi_embedd, en_embedd).to(self.device)
        self.decoder = GRU_decoder(input_dim, hidden_dim, num_layers, self.device, vi_embedd, en_embedd).to(self.device)
        self.out_vi = OutputLayer(hidden_dim, vi_vocab_size).to(self.device)
        self.out_en = OutputLayer(hidden_dim, en_vocab_size).to(self.device)

        # 3. Load Checkpoint
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Không tìm thấy file {checkpoint_path}")

        print("Đang load trọng số mô hình...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.decoder.load_state_dict(checkpoint['decoder'])
        self.out_vi.load_state_dict(checkpoint['outputlayer_vi'])
        self.out_en.load_state_dict(checkpoint['outputlayer_en'])

        self.encoder.eval()
        self.decoder.eval()
        self.out_vi.eval()
        self.out_en.eval()
        print("Sẵn sàng dịch thuật!")

    def translate(self, text, direction="vi2en", max_len=50):
        # Thiết lập biến theo chiều dịch
        vi2en = (direction == "vi2en")

        if vi2en:
            words = processing_vietnamese(text)
            src_word2idx = self.vi_word2idx
            tgt_idx2word = self.en_idx2word
            bos_idx = self.en_word2idx["<Vi2En>"] # Chiều vào decode là BOS của EN
            eos_idx = self.en_word2idx["<EOS_EN>"]
            unk_idx = self.vi_word2idx["<UNK_VI>"]
        else:
            words = processing_english(text)
            src_word2idx = self.en_word2idx
            tgt_idx2word = self.vi_idx2word
            bos_idx = self.vi_word2idx["<En2Vi>"]
            eos_idx = self.vi_word2idx["<EOS_VI>"]
            unk_idx = self.en_word2idx["<UNK_EN>"]

        if len(words) == 0:
            return ""

        # 1. Mã hóa câu đầu vào
        src_indices = [src_word2idx.get(w, unk_idx) for w in words]
        src_tensor = torch.tensor([src_indices], dtype=torch.long, device=self.device) # Shape: 1 x Seq
        src_length = torch.tensor([len(src_indices)], dtype=torch.int64)

        with torch.inference_mode():
            # 2. Chạy Encoder
            output_encoder, h_n_encoder = self.encoder(src_tensor, src_length, vi2en)
            pad_mask = (src_tensor == 0) # PAD_IDX = 0

            # 3. Chạy Decoder từng bước (Greedy Decoding)
            decoder_input = torch.tensor([[bos_idx]], device=self.device) # Shape: 1 x 1
            h_n_decoder = h_n_encoder
            decoded_words = []

            for _ in range(max_len):
                output_decoder, h_n_decoder = self.decoder(decoder_input, h_n_decoder, vi2en)

                ht = output_decoder[:, 0, :] # Shape: 1 x hidden_dim
                ct = attention(output_encoder, ht, pad_mask) # Shape: 1 x hidden_dim

                combine = torch.cat([ht, ct], dim=1)

                # Phân loại ra từ vựng
                if vi2en:
                    out = self.out_en(combine)
                else:
                    out = self.out_vi(combine)

                # Lấy từ có xác suất cao nhất
                top1_idx = out.argmax(dim=1).item()

                # Nếu gặp thẻ kết thúc câu thì dừng
                if top1_idx == eos_idx:
                    break

                decoded_words.append(tgt_idx2word[top1_idx])

                # Lấy từ vừa sinh ra làm đầu vào cho bước tiếp theo
                decoder_input = torch.tensor([[top1_idx]], device=self.device)

        # 4. Trả về câu đã dịch
        return " ".join(decoded_words).capitalize()

# --- 4. HÀM TẢI FILE TỪ GOOGLE DRIVE ---
def download_file_from_drive(file_id, output_path):
    """Kiểm tra nếu file chưa tồn tại thì dùng gdown để tải từ Google Drive"""
    if not os.path.exists(output_path):
        print(f"Đang tải {output_path} từ Google Drive...")
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, output_path, quiet=False)
    else:
        print(f"File {output_path} đã có sẵn.")

# =====================================================================
# VÍ DỤ SỬ DỤNG PUBLIC (CHO NGƯỜI DÙNG KHÁC)
# =====================================================================
if __name__ == '__main__':
    
    print("\n" + "="*50)
    print("KIỂM TRA VÀ TẢI TÀI NGUYÊN")
    print("="*50)
    
    ckpt_file_id = "https://drive.google.com/drive/folders/1nbopoB4tbgTbenERRlaxG1xL91LJKM5S?usp=drive_link"
    vi_vocab_id  = "https://drive.google.com/file/d/1iSA9CFZAPyfQ2A-sE48kCM38uBAz3s0k/view?usp=drive_link"
    en_vocab_id  = "https://drive.google.com/file/d/1-XwqfMp38RQLkgfnmLt5A_HQBmf6o3L7/view?usp=drive_link"

    # Đường dẫn cục bộ lưu trữ khi người khác chạy file này
    ckpt_path = "checkpoints_best.pt"
    vi_path   = "vi_vocab_embeddings.pt"
    en_path   = "en_vocab_embeddings.pt"

    # Tải 3 file thiết yếu trước khi chạy Chatbot
    try:
        download_file_from_drive(ckpt_file_id, ckpt_path)
        download_file_from_drive(vi_vocab_id, vi_path)
        download_file_from_drive(en_vocab_id, en_path)
    except Exception as e:
        print(f"Lỗi tải file: {e}")
        print("Vui lòng kiểm tra lại Google Drive ID hoặc cài đặt thư viện 'gdown' (pip install gdown).")
        exit()

    try:
        translator = Translator(ckpt_path, vi_path, en_path)

        print("\n" + "="*50)
        print("CHATBOT DỊCH THUẬT (Gõ 'exit' hoặc 'quit' để thoát)")
        print("="*50)

        while True:
            text = input("\n Nhập câu cần dịch: ")
            if text.lower() in ['exit', 'quit']:
                break

            direction = input(" Hướng dịch (1: Việt -> Anh | 2: Anh -> Việt): ")

            if direction == '1':
                res = translator.translate(text, direction="vi2en")
                print(f"English: {res}")
            elif direction == '2':
                res = translator.translate(text, direction="en2vi")
                print(f"Tiếng Việt: {res}")
            else:
                print("Vui lòng chọn 1 hoặc 2.")

    except Exception as e:
        print(f"Lỗi khởi chạy mô hình: {e}")